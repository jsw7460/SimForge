"""Go2 flat terrain locomotion configs for Newton simulator.

Phase A migration: ``Go2FlatNewtonConfig`` is now a thin compatibility
shim defined in ``presets/go2_flat/base.py`` (subclass of the unified
``Go2FlatConfig`` with ``sim_type="newton"``). It is re-exported here
for variants (``gait_conditioned``, etc.) that still inherit from it.
"""

from rlworld.rl.configs.presets.go2_flat.base import Go2FlatNewtonConfig
from rlworld.rl.configs.presets.go2_flat.mlp import get_config as _get_unified_config


def get_mlp_config():
    """Backward-compatible entry point — delegates to ``mlp.get_config(sim="newton")``."""
    return _get_unified_config(sim="newton")


__all__ = [
    "Go2FlatNewtonConfig",
    "get_mlp_config",
]
