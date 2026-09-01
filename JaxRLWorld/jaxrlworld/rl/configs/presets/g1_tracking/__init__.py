"""G1 motion tracking preset (Mjlab-faithful port)."""

from jaxrlworld.rl.configs.presets.g1_tracking.base import G1TrackingConfig
from jaxrlworld.rl.configs.presets.g1_tracking.transformer import (
    G1TrackingTransformerConfig,
)

__all__ = ["G1TrackingConfig", "G1TrackingTransformerConfig"]
