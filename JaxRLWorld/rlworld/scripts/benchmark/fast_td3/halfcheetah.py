import os

os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs.algorithms import FastTD3Config
from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.go2_flat.genesis.mlp import get_config


def main():
    # Get complete config from preset
    configs_dict = get_config()

    configs_dict["runner"]["run_name"] = "HalfCheetah_TD3"

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    cfgs_for_run.env.num_envs = 1
    cfgs_for_run.env.env_name = "GymnasiumEnv"
    cfgs_for_run.env.task_name = "HalfCheetah-v4"
    cfgs_for_run.nn.policy.actor_kwargs.update({
        "hidden_dims": [256, 128, 128],
        "activation": "relu",
        "ortho_init": False,
        "output_gain": 0.1
    })
    cfgs_for_run.nn.policy.critic_kwargs.update({
        "hidden_dims": [256, 128, 128],
        "activation": "relu",
        "ortho_init": False,
        "output_gain": 0.1
    })

    fast_td3_config = FastTD3Config(
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.97,
        tau=0.005,
        batch_size=32768,
        buffer_size=4096000,
        learning_starts=100,
        policy_delay=2,
        num_atoms=151,
        v_min=0.0,
        v_max=4000.0,
        is_squashed=True,
        obs_normalization=True,
        use_cdq=True,
        num_gradient_steps=1,
    )
    cfgs_for_run.algorithm = fast_td3_config

    cfgs_for_run.runner.log_interval = 500
    cfgs_for_run.runner.max_iterations = 1000000
    cfgs_for_run.runner.save_interval = 100000

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
