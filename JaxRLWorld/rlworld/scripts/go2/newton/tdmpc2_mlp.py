import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs.algorithms import TDMPC2Config
from rlworld.rl.configs.presets.go2_flat.mlp import get_config
from rlworld.rl.runners import BaseRunner


def main():
    # Get complete config from preset
    cfgs_for_run = get_config(sim="newton").with_cli_overrides()

    tdmpc2_config = TDMPC2Config(
        vmin=-5.0,
        vmax=5.0,
        num_bins=101,
        num_samples=512,
        num_pi_trajs=24,
        num_elites=64,
        num_iterations=6,
        buffer_size=5_000_000,
        num_gradient_steps=8,
        batch_size=10000,
        learning_starts=5000,
    )
    cfgs_for_run.env.num_envs = 1024
    cfgs_for_run.runner.max_iterations = 100000
    cfgs_for_run.action.clip_actions = "joint_limit"
    cfgs_for_run.action.action_scale = 1.0
    # cfgs_for_run.action.clip_actions = (-5.0, 5.0)
    cfgs_for_run.algorithm = tdmpc2_config
    cfgs_for_run.runner.run_name = "Newton_Go2_MLP_TDMPC2"

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
