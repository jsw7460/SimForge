"""SAC on Gymnasium ``Swimmer-v5``.

Mirrors ``sac/sac_halfcheetah.py`` (manual ``gym.make`` + FlattenObservation
+ SyncVectorEnv -> GymnasiumEnv -> OffPolicyRunner), pointed at the real
gymnasium Swimmer-v5 task.

``ACTION_REPEAT=1`` is the standard gym Swimmer-v5 setting. Swimmer is one of
the tasks where a larger action repeat (e.g. 4) is a common trick if learning
stalls — change the constant below if needed.

Run (GPU box):
    python jaxrlworld/scripts/benchmark/sac/swimmer.py
"""

import os

os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.wrappers import FlattenObservation

from jaxrlworld.rl.configs import SACPolicyConfig
from jaxrlworld.rl.configs.algorithms import SACConfig
from jaxrlworld.rl.configs.presets.go2.mlp import get_config
from jaxrlworld.rl.envs import GymnasiumEnv
from jaxrlworld.rl.envs.gymnasium import ActionRepeatWrapper
from jaxrlworld.rl.runners import OffPolicyRunner

TASK_NAME = "Swimmer-v5"
MAX_EPISODE_STEPS = 1000
ACTION_REPEAT = 1  # standard gym Swimmer-v5; try 4 if learning stalls


def main():
    cfgs_for_run = get_config(sim="genesis")
    cfgs_for_run.runner.run_name = "Swimmer_SAC"

    cfgs_for_run.env.num_envs = 1
    cfgs_for_run.env.env_name = "GymnasiumEnv"
    cfgs_for_run.env.seed = 0
    cfgs_for_run.env.task_name = TASK_NAME

    cfgs_for_run.algorithm = SACConfig(
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=2500,
    )
    cfgs_for_run.nn.policy = cfgs_for_run.nn.policy.to(SACPolicyConfig)
    cfgs_for_run.runner.log_interval = 500
    cfgs_for_run.runner.max_iterations = 1000000
    cfgs_for_run.runner.save_interval = 100000
    cfgs_for_run.runner.eval_interval = 5000

    def make_env(seed):
        def _init():
            env = gym.make(TASK_NAME, max_episode_steps=MAX_EPISODE_STEPS)
            env = ActionRepeatWrapper(env, repeat=ACTION_REPEAT)
            env = FlattenObservation(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return _init

    env_gym = SyncVectorEnv(
        [make_env(i) for i in range(cfgs_for_run.env.num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env = GymnasiumEnv(
        env_gym,
        env_cfg=cfgs_for_run.env,
        scene_cfg=cfgs_for_run.scene,
        obs_cfg=cfgs_for_run.observation,
        act_cfg=cfgs_for_run.action,
        reward_cfg=cfgs_for_run.reward,
        command_cfg=cfgs_for_run.command,
        seed=cfgs_for_run.env.seed,
    )

    runner = OffPolicyRunner(env=env, cfgs=cfgs_for_run, use_wandb=True, seed=cfgs_for_run.env.seed)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
