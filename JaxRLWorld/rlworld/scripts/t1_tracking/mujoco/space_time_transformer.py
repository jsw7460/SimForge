"""Train T1 motion tracking with the SpaceTimeTransformer in MuJoCo (mjlab).

See ``newton/space_time_transformer.py`` for details.
"""
from rlworld.rl.configs.presets.t1_tracking.transformer import (
    T1TrackingTransformerConfig,
)
from rlworld.rl.runners import BaseRunner


def main():
    cfgs_for_run = (
        T1TrackingTransformerConfig(sim_type="mujoco")
        .build()
        .with_cli_overrides()
    )
    runner = BaseRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
