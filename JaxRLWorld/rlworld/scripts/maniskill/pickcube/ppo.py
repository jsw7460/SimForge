"""PPO training on ManiSkill PickCube-v1 via the ManiSkill adapter.

Config lives entirely in the preset ``ManiSkillPickCubeConfig`` (so eval can
re-run it for a faithful reload). This script just builds it, constructs the
ManiSkill env from ``cfgs.env.gym_make_kwargs`` (single source of truth),
and injects it into ``OnPolicyRunner`` with ``env=``. No Genesis/MuJoCo import.

Run (JAX-based -> jaxpy to avoid GPU preallocation/OOM):

    jaxpy -m rlworld.scripts.maniskill.pickcube.ppo --max_iterations 400
"""

from __future__ import annotations

import argparse

import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rlworld.rl.configs.presets.maniskill.pickcube import ManiSkillPickCubeConfig
from rlworld.rl.envs.maniskill_env import ManiSkillEnv
from rlworld.rl.runners import OnPolicyRunner


def build_env(cfgs, num_envs, seed):
    """Build the ManiSkill env from the config's gym_make_kwargs."""
    base = gym.make(cfgs.env.task_name, num_envs=num_envs, **cfgs.env.gym_make_kwargs)
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
    p = argparse.ArgumentParser(description="PPO on ManiSkill PickCube-v1 (baseline-aligned)")
    p.add_argument("--task", type=str, default="PickCube-v1")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--obs_mode", type=str, default="state")
    p.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    p.add_argument("--max_iterations", type=int, default=400)
    p.add_argument("--eval_interval", type=int, default=25, help="0 disables in-training eval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    cfgs = ManiSkillPickCubeConfig(
        task=args.task,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        max_iterations=args.max_iterations,
        eval_interval=args.eval_interval,
        seed=args.seed,
    ).build()

    # --- Train env (adapter, injected via env=). ---
    train_env = build_env(cfgs, cfgs.env.num_envs, args.seed)
    runner = OnPolicyRunner(env=train_env, cfgs=cfgs, use_wandb=not args.no_wandb, seed=args.seed)

    # --- Eval env (adapter) injected so the lazy creator skips the
    #     public dispatch that would otherwise rebuild a different env. ---
    if cfgs.runner.eval_interval > 0:
        runner._eval_env = build_env(cfgs, cfgs.runner.eval_num_envs, args.seed + 10_000)

    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
