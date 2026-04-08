"""Go2 flat terrain locomotion configs for MuJoCo (mjlab) simulator.

Phase A migration: ``Go2FlatMujocoConfig`` is now a thin compatibility
shim defined in ``presets/go2_flat/base.py`` (subclass of the unified
``Go2FlatConfig`` with ``sim_type="mujoco"``). It is re-exported here
for variants (``gait_conditioned``, etc.) that still inherit from it.
"""

from rlworld.rl.configs.presets.go2_flat.base import Go2FlatMujocoConfig
from rlworld.rl.configs.presets.go2_flat.mlp import get_config as _get_unified_config


def get_mlp_config():
    """Backward-compatible entry point — delegates to ``mlp.get_config(sim="mujoco")``."""
    return _get_unified_config(sim="mujoco")


__all__ = [
    "Go2FlatMujocoConfig",
    "get_mlp_config",
]
