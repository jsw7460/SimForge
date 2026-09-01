"""Train state-based PPO on a ManiSkill task.

Hyperparameters come from the ManiSkillStatePPOConfig preset + the per-task
STATE_PPO_TASKS table (ManiSkill's official baseline recipe). Builds the
ManiSkill env from cfgs.env.gym_make_kwargs and injects it into OnPolicyRunner
with env=. No Genesis/MuJoCo import.

Run (JAX-based -> jaxpy to avoid GPU preallocation/OOM):

    jaxpy -m jaxrlworld.scripts.maniskill.train --task PickCube-v1
    jaxpy -m jaxrlworld.scripts.maniskill.train --task PushT-v1
"""

from __future__ import annotations

import argparse

import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from jaxrlworld.rl.configs.presets.maniskill.base import STATE_PPO_TASKS, get_config
from jaxrlworld.rl.envs.maniskill_env import ManiSkillEnv
from jaxrlworld.rl.runners import OnPolicyRunner


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
    p = argparse.ArgumentParser(description="Train state-based PPO on a ManiSkill task")
    p.add_argument("--task", type=str, default="PickCube-v1", choices=sorted(STATE_PPO_TASKS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    cfgs = get_config(args.task, seed=args.seed)

    train_env = build_env(cfgs, cfgs.env.num_envs, args.seed)
    runner = OnPolicyRunner(env=train_env, cfgs=cfgs, use_wandb=not args.no_wandb, seed=args.seed)

    # Eval env (same adapter) injected so the lazy creator skips the public
    # dispatch; only when in-training eval is enabled.
    if cfgs.runner.eval_interval > 0:
        runner._eval_env = build_env(cfgs, cfgs.runner.eval_num_envs, args.seed + 10_000)

    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
