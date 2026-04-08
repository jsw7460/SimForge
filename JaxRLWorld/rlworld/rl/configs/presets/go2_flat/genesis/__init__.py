"""Go2 flat terrain locomotion configs for Genesis simulator.

Phase A migration: ``Go2FlatGenesisConfig`` is now a thin compatibility
shim defined in ``presets/go2_flat/base.py`` (subclass of the unified
``Go2FlatConfig`` with ``sim_type="genesis"``). It is re-exported here
for variants (``gait_conditioned``, ``scaffolded_tdmpc2``, etc.) that
still inherit from it.
"""

from rlworld.rl.configs.presets.go2_flat.base import Go2FlatGenesisConfig
from rlworld.rl.configs.presets.go2_flat.mlp import get_config as _get_unified_config


def get_mlp_config():
    """Backward-compatible entry point — delegates to ``mlp.get_config(sim="genesis")``."""
    return _get_unified_config(sim="genesis")


__all__ = [
    "Go2FlatGenesisConfig",
    "get_mlp_config",
]
