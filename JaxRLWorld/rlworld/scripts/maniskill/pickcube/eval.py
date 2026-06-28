"""Evaluate a trained ManiSkill PickCube policy (self-contained).

Reloads the config the same way every other sim does -- via
``load_config_from_checkpoint``, which re-runs the saved preset
(``ManiSkillPickCubeConfig``) -- then builds the ManiSkill env, loads the
trained weights, and runs the sim-agnostic eval loop to report success rate +
return. With ``--record_video`` it saves an mp4 via ManiSkill's native
``RecordEpisode`` (SAPIEN renderer).

It uses generic, sim-agnostic framework utilities
(``load_checkpoint_metadata`` / ``load_config_from_checkpoint`` / ``load_runner``
/ ``runner._evaluate_env``) plus the ManiSkill adapter and preset.

Run (JAX-based -> jaxpy to avoid GPU preallocation/OOM):

    jaxpy -m rlworld.scripts.maniskill.pickcube.eval \
        --policy_path outputs/models/.../checkpoint_latest \
        --num_envs 16 --num_episodes 100 --record_video
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rlworld.rl.envs.maniskill_env import ManiSkillEnv
from rlworld.rl.utils.checkpoint import load_checkpoint_metadata, load_config_from_checkpoint, load_runner


def build_eval_env(cfgs, num_envs, seed, video_dir, record_steps):
    """Build the ManiSkill eval env, optionally wrapped for mp4 capture."""
    make_kwargs = dict(cfgs.env.gym_make_kwargs)
    if video_dir is not None:
        # RecordEpisode pulls frames via env.render(); the training env (state
        # obs) is built without a render mode, so enable it for video capture.
        make_kwargs.setdefault("render_mode", "rgb_array")
    base = gym.make(cfgs.env.task_name, num_envs=num_envs, **make_kwargs)
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        base = RecordEpisode(
            base,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            max_steps_per_video=record_steps,
            video_fps=30,
        )
    venv = ManiSkillVectorEnv(base, num_envs, auto_reset=True, ignore_terminations=False)
    return ManiSkillEnv(
        venv,
        env_cfg=cfgs.env,
        scene_cfg=cfgs.scene,
        obs_cfg=cfgs.observation,
        act_cfg=cfgs.action,
        reward_cfg=cfgs.reward,
        command_cfg=cfgs.command,
        seed=seed,
    )


def main():
    p = argparse.ArgumentParser(description="Evaluate a ManiSkill PickCube PPO policy")
    p.add_argument("--policy_path", type=str, required=True, help="checkpoint dir (e.g. .../checkpoint_latest)")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--record_video", action="store_true", help="save mp4 via ManiSkill RecordEpisode")
    p.add_argument("--video_dir", type=str, default=None, help="default: <policy_path>/eval_videos")
    p.add_argument("--record_steps", type=int, default=500, help="max steps per recorded video")
    args = p.parse_args()

    # Standard reload: re-run the saved preset (same path as every other sim).
    metadata = load_checkpoint_metadata(args.policy_path)
    cfgs = load_config_from_checkpoint(metadata)

    # Eval-time overrides.
    cfgs.env.num_envs = args.num_envs
    cfgs.runner.eval_num_episodes = args.num_episodes
    cfgs.runner.eval_deterministic = True

    video_dir = None
    if args.record_video:
        video_dir = args.video_dir or os.path.join(args.policy_path, "eval_videos")

    env = build_eval_env(cfgs, args.num_envs, args.seed, video_dir, args.record_steps)

    # Load weights into the appropriate runner (cfgs passed -> no re-derive).
    runner = load_runner(env=env, checkpoint_path=args.policy_path, cfgs=cfgs, use_wandb=False)

    stats = runner._evaluate_env(env)

    print("\n" + "=" * 60)
    print(f"ManiSkill eval: {cfgs.env.task_name}  (num_envs={args.num_envs})")
    print("=" * 60)
    print(f"  episodes      : {stats.get('eval/num_episodes', 0)}")
    print(f"  mean return   : {stats.get('eval/mean_return', 0.0):.3f} +/- {stats.get('eval/std_return', 0.0):.3f}")
    print(f"  episode length: {stats.get('eval/mean_episode_length', 0.0):.1f}")
    if "eval/success_rate" in stats:
        print(f"  success rate  : {stats['eval/success_rate'] * 100:.1f}%")
    else:
        print("  success rate  : (not reported by env)")

    if video_dir is not None:
        # Flush RecordEpisode buffers to disk.
        env.gym_env.close()
        print(f"\n  mp4 saved under: {video_dir}")


if __name__ == "__main__":
    main()
