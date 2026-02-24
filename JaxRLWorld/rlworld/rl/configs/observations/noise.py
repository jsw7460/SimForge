from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class NoiseConfig:
    """Base noise configuration."""
    pass


@dataclass
class UniformNoiseConfig(NoiseConfig):
    """Uniform distribution noise."""
    n_min: float = 0.0
    n_max: float = 0.0
    operation: Literal["add", "scale", "abs"] = "add"


@dataclass
class GaussianNoiseConfig(NoiseConfig):
    """Gaussian distribution noise."""
    mean: float = 0.0
    std: float = 1.0
    operation: Literal["add", "scale", "abs"] = "add"


def apply_noise(obs: torch.Tensor, noise_cfg: NoiseConfig) -> torch.Tensor:
    """Apply noise to observation tensor.

    Args:
        obs: Observation tensor of shape (num_envs, obs_dim).
        noise_cfg: Noise configuration.

    Returns:
        Noised observation tensor.
    """
    if isinstance(noise_cfg, UniformNoiseConfig):
        noise = torch.empty_like(obs).uniform_(noise_cfg.n_min, noise_cfg.n_max)
    elif isinstance(noise_cfg, GaussianNoiseConfig):
        noise = torch.empty_like(obs).normal_(noise_cfg.mean, noise_cfg.std)
    else:
        return obs

    if noise_cfg.operation == "add":
        return obs + noise
    elif noise_cfg.operation == "scale":
        return obs * noise
    elif noise_cfg.operation == "abs":
        return noise
    return obs