"""Train T1 motion tracking with the SpaceTimeTransformer in Newton.

Usage:
    jaxpy JaxRLWorld/rlworld/scripts/t1_tracking/newton/space_time_transformer.py \\
        env.num_envs=4096 runner.max_iterations=10000

Identical wiring to ``mlp.py`` but uses
:class:`T1TrackingTransformerConfig` (factorized space x time transformer
actor/critic, future-motion-reference window obs, NPMP-style information
bottleneck). See ``rl/configs/presets/t1_tracking/transformer.py`` for
the architecture-specific config fields.
"""

from rlworld.rl.configs.presets.t1_tracking.transformer import (
    T1TrackingTransformerConfig,
)
from rlworld.rl.runners import BaseRunner


def main():
    cfgs_for_run = (
        T1TrackingTransformerConfig(sim_type="newton", pe_type="traversal", use_relational_bias=False)
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
