"""Train T1 motion-tracking policy in Newton.

Usage:
    uv run python -m rlworld.scripts.t1_tracking.newton.mlp \\
        --motion-file /path/to/motion.npz \\
        --num-envs 4096 --max-iterations 10000
"""
import tyro

from rlworld.rl.configs.presets.t1_tracking.base import T1TrackingConfig
from rlworld.rl.runners import BaseRunner


def main(cfg: T1TrackingConfig):
    cfg.sim_type = "newton"
    cfgs_for_run = cfg.build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    tyro.cli(main)
