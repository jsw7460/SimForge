"""Compat shim: delegates to the unified ``presets/go2_flat/mlp.py``.

Phase A migration: kept so existing scripts that import from
``presets.go2_flat.newton.mlp`` continue to work. Phase B will update
those callers and remove this shim.
"""

from rlworld.rl.configs.presets.go2_flat.mlp import get_config as _get_unified_config


def get_config():
    return _get_unified_config(sim="newton")
