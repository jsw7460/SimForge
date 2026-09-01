"""PPO on Gymnasium ``Swimmer-v5``.

Mirrors ``ppo/halfcheetah.py`` (config-driven GymnasiumEnv path: the runner
builds a vectorised ``gym.make`` env from ``cfgs.env.task_name``). Swimmer-v5
obs is already a flat Box, so no FlattenObservation / action-repeat is needed.

Note: Swimmer is a notoriously hard task for PPO (long effective horizon /
discounting); SAC / TD3 / TDMPC2 typically score much higher. Bump
``NUM_ENVS`` for more parallel rollouts if you have the throughput.

Run (GPU box):
    python jaxrlworld/scripts/benchmark/ppo/swimmer.py
"""

import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from jaxrlworld.rl.configs.algorithms.ppo import PPOConfig
from jaxrlworld.rl.configs.common_config_classes import (
    Activation,
    DefaultInit,
    DistributionType,
    MLPActorCfg,
    MLPCriticCfg,
)
from jaxrlworld.rl.configs.presets.go2.mlp import get_config
from jaxrlworld.rl.runners import BaseRunner

TASK_NAME = "Swimmer-v5"
NUM_ENVS = 64  # tunable: more parallel rollouts -> better PPO, slower per-iter on CPU gym


def main():
    cfgs_for_run = get_config(sim="genesis")
    cfgs_for_run.runner.run_name = "Swimmer_PPO"

    cfgs_for_run.env.num_envs = NUM_ENVS
    cfgs_for_run.env.env_name = "GymnasiumEnv"
    cfgs_for_run.env.task_name = TASK_NAME
    cfgs_for_run.nn.policy.actor = MLPActorCfg(
        hidden_dims=[256, 128, 64],
        activation=Activation.RELU,
        init=DefaultInit(),
    )
    cfgs_for_run.nn.policy.critic = MLPCriticCfg(
        hidden_dims=[256, 128, 64],
        activation=Activation.RELU,
        init=DefaultInit(),
    )
    cfgs_for_run.nn.policy.distribution_type = DistributionType.GAUSSIAN

    cfgs_for_run.algorithm = PPOConfig(
        actor_lr=3e-4,
        critic_lr=3e-4,
    )

    cfgs_for_run.runner.log_interval = 10
    cfgs_for_run.runner.max_iterations = 1000000
    cfgs_for_run.runner.save_interval = 100000
    cfgs_for_run.runner.eval_interval = 5000

    runner = BaseRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
