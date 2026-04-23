"""Train G1 motion-tracking policy in Genesis.

Usage:
    jaxpy JaxRLWorld/rlworld/scripts/g1_tracking/genesis/mlp.py \\
        env.num_envs=4096 runner.max_iterations=10000

Motion source is set in ``G1TrackingConfig.motion_files`` (tuple of NPZ
paths — length-1 for single-clip, length >= 2 for multi-motion).
"""
from rlworld.rl.configs.presets.g1_tracking.base import G1TrackingConfig
from rlworld.rl.runners import BaseRunner


def main():
    cfgs_for_run = G1TrackingConfig(sim_type="genesis").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
