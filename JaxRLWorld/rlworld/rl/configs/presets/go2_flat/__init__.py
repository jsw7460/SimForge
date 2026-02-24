"""
Go2 flat terrain locomotion configs.

Supports multiple simulator backends:
- genesis: Genesis physics engine
- newton: Newton physics engine

Usage (import only what you need):
    from rlworld.rl.configs.presets.go2_flat.genesis.mlp import get_config
    # or
    from rlworld.rl.configs.presets.go2_flat.newton.mlp import get_config
"""

# No automatic imports - use explicit imports to avoid loading both backends
__all__ = ["genesis", "newton"]
