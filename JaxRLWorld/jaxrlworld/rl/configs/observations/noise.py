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


_NOISE_REGISTRY: dict[str, type] = {}


def _register_noise_classes() -> None:
    """Populate registry after class definitions."""
    for cls in (UniformNoiseConfig, GaussianNoiseConfig):
        _NOISE_REGISTRY[cls.__qualname__] = cls


def noise_config_from_dict(d: dict) -> NoiseConfig:
    """Reconstruct a NoiseConfig subclass from a serialized dict."""
    _register_noise_classes()
    d = dict(d)
    type_name = d.pop("_type", None)
    if type_name is None:
        raise ValueError("Missing '_type' in noise config dict")
    cls = _NOISE_REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown noise type: {type_name}")
    return cls(**d)


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
