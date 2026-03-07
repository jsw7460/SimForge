import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.go1.genesis.mlp import get_config
from rlworld.rl.configs.algorithms import FastTD3Config

large = [512, 256, 128]


# extreme_large = [2000, 2000, 2000]

def main():
    cfgs_for_run = get_config().with_cli_overrides()

    scale_param = 1.0
    # cfgs_for_run.action.action_scale = cfgs_for_run.action.action_scale / scale_param
    cfgs_for_run.action.action_scale = 1.0
    cfgs_for_run.action.clip_actions = (-scale_param, scale_param)

    # FastTD3 architecture: Actor [512, 256, 128], Critic [1024, 512, 256]
    cfgs_for_run.nn.policy["actor_kwargs"].update({
        "hidden_dims": [512, 256, 128],
        "activation": "relu",
    })
    cfgs_for_run.nn.policy["critic_kwargs"].update({
        "hidden_dims": [1024, 512, 256],
        "activation": "relu",
    })
    cfgs_for_run.runner.max_iterations = 25000

    fast_td3_config = FastTD3Config(
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.97,
        tau=0.005,
        batch_size=32768,
        buffer_size=4096000,
        learning_starts=100,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        num_atoms=101,
        v_min=-20.0,
        v_max=50.0,
        noise_min=0.2,
        noise_max=0.8,
        is_squashed=True,
        use_cdq=True,
        utd_ratio=8,
        obs_normalization=True
    )
    cfgs_for_run.algorithm = fast_td3_config

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
