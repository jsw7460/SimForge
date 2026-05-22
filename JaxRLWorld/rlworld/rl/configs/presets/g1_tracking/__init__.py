"""G1 motion tracking preset (Mjlab-faithful port)."""

from rlworld.rl.configs.presets.g1_tracking.base import G1TrackingConfig
from rlworld.rl.configs.presets.g1_tracking.transformer import (
    G1TrackingTransformerConfig,
)

__all__ = ["G1TrackingConfig", "G1TrackingTransformerConfig"]
