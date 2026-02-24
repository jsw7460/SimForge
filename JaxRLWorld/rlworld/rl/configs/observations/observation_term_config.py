from dataclasses import dataclass, field
from typing import Callable, Any

import torch

from rlworld.rl.configs.observations.noise import NoiseConfig


@dataclass
class ObservationTermConfig:
    """Configuration for an observation term."""
    func: Callable[..., torch.Tensor]
    history_length: int = 0
    flatten_history_dim: bool = True
    clip: tuple[float, float] | None = None
    scale: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    noise: NoiseConfig | None = None
