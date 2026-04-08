import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.algorithms import TD3Config
from rlworld.rl.configs import TD3PolicyConfig
from rlworld.rl.configs.presets.go2_flat.mlp import get_config


def main():
    # Get complete config from preset
    cfgs_for_run = get_config(sim="genesis").with_cli_overrides()

    cfgs_for_run.env.num_envs = 4096
    td3_config = TD3Config(
        batch_size=4096 * 2,
        buffer_size=4096 * 1100,
        learning_starts=10000,
        num_gradient_steps=16,
        obs_normalization=True,
        target_noise_clip=0.1
    )
    cfgs_for_run.algorithm = td3_config
    cfgs_for_run.nn.policy = cfgs_for_run.nn.policy.to(TD3PolicyConfig)

    cfgs_for_run.action.clip_actions = "joint_limit"
    cfgs_for_run.action.action_scale = 1.0

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
