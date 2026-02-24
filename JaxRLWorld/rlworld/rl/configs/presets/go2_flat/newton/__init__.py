"""Go2 flat terrain locomotion configs for Newton simulator."""

from .mlp import get_config as get_mlp_config
from .base import Go2FlatNewtonConfig

__all__ = [
    "Go2FlatNewtonConfig",
    "get_mlp_config",
]
