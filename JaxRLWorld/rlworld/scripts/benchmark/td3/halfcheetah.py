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

from rlworld.rl.configs import GenesisConfigsForRun, TD3PolicyConfig
from rlworld.rl.configs.algorithms import TD3Config
from rlworld.rl.configs.presets.go2_flat.mlp import get_config
from rlworld.rl.envs import GymnasiumEnv
from rlworld.rl.runners import OffPolicyRunner


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat action for n steps, accumulate reward."""

    def __init__(self, env, repeat=2):
        super().__init__(env)
        self._repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def main():
    # Get complete config from preset
    configs_dict = get_config(sim="genesis")

    configs_dict["runner"]["run_name"] = "HalfCheetah_TD3"

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    cfgs_for_run.env.num_envs = 1
    cfgs_for_run.env.env_name = "GymnasiumEnv"
    cfgs_for_run.env.seed = 0
    cfgs_for_run.env.task_name = "dm_control/cheetah-run-v0"
    sac_config = TD3Config(
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=2500,
    )
    cfgs_for_run.algorithm = sac_config
    cfgs_for_run.nn.policy = cfgs_for_run.nn.policy.to(TD3PolicyConfig)
    cfgs_for_run.runner.log_interval = 500
    cfgs_for_run.runner.max_iterations = 1000000
    cfgs_for_run.runner.save_interval = 100000

    def make_env(seed):
        def _init():
            env = gym.make(cfgs_for_run.env.task_name, max_episode_steps=1000)
            env = ActionRepeatWrapper(env, repeat=2)
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

    # runner = BaseRunner.create_with_env(cfgs_for_run)
    runner = OffPolicyRunner(env=env, cfgs=cfgs_for_run, use_wandb=True, seed=cfgs_for_run.env.seed)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
