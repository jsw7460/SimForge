"""Train T1 motion-tracking policy in MuJoCo (mjlab).

Usage:
    jaxpy JaxRLWorld/rlworld/scripts/t1_tracking/mujoco/mlp.py \\
        env.num_envs=4096 runner.max_iterations=10000

Motion source is set in ``T1TrackingConfig.motion_files`` (tuple of NPZ
paths — length-1 for single-clip, length >= 2 for multi-motion). Edit
the preset or subclass it to target a different clip set.
"""
from rlworld.rl.configs.presets.t1_tracking.base import T1TrackingConfig
from rlworld.rl.runners import BaseRunner


def main():
    cfgs_for_run = T1TrackingConfig(sim_type="mujoco").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
