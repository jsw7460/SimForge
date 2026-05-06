import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.presets.go2_flat.mlp import get_config
from rlworld.rl.runners import BaseRunner


def main():
    # Get complete config from preset
    configs_dict = get_config(sim="genesis")

    configs_dict["runner"]["run_name"] = "HalfCheetah_PPO"

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    cfgs_for_run.env.num_envs = 1024
    cfgs_for_run.env.env_name = "GymnasiumEnv"
    cfgs_for_run.env.task_name = "HalfCheetah-v4"
    cfgs_for_run.nn.policy.actor_kwargs.update(
        {"hidden_dims": [256, 128, 64], "activation": "relu", "ortho_init": False, "output_gain": 0.1}
    )
    cfgs_for_run.nn.policy.critic_kwargs.update(
        {"hidden_dims": [256, 128, 64], "activation": "relu", "ortho_init": False, "output_gain": 0.1}
    )
    cfgs_for_run.nn.policy.distribution_type = "gaussian"

    ppo_config = PPOConfig(
        actor_lr=3e-4,
        critic_lr=3e-4,
    )
    cfgs_for_run.algorithm = ppo_config

    cfgs_for_run.runner.log_interval = 10
    cfgs_for_run.runner.max_iterations = 1000000
    cfgs_for_run.runner.save_interval = 100000

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
