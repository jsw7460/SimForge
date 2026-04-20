"""Train G1 motion-tracking policy in Newton.

Usage:
    jaxpy JaxRLWorld/rlworld/scripts/g1_tracking/newton/mlp.py \\
        env.num_envs=64 runner.max_iterations=5 \\
        command.terms.motion.motion_file=/path/to/motion.npz

The preset's ``motion_file`` default points at the Gangnam Style NPZ;
override via ``command.terms.motion.motion_file=<path>`` or by editing
``G1TrackingConfig.motion_file`` directly.
"""
from rlworld.rl.configs.presets.g1_tracking.base import G1TrackingConfig
from rlworld.rl.runners import BaseRunner


def main():
    cfgs_for_run = G1TrackingConfig(sim_type="newton").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
