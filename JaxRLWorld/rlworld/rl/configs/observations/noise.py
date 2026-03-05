from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
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


def apply_noise(obs, noise_cfg: NoiseConfig):
    """Apply noise to observation tensor (supports both torch and JAX).

    Args:
        obs: Observation tensor/array of shape (num_envs, obs_dim).
        noise_cfg: Noise configuration.

    Returns:
        Noised observation tensor/array.
    """
    if isinstance(obs, jax.Array):
        return _apply_noise_jax(obs, noise_cfg)
    return _apply_noise_torch(obs, noise_cfg)


def _apply_noise_torch(obs: torch.Tensor, noise_cfg: NoiseConfig) -> torch.Tensor:
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


# Global JAX PRNG key for observation noise
_noise_rng_key = jax.random.PRNGKey(0)


def _apply_noise_jax(obs: jax.Array, noise_cfg: NoiseConfig) -> jax.Array:
    global _noise_rng_key
    _noise_rng_key, subkey = jax.random.split(_noise_rng_key)

    if isinstance(noise_cfg, UniformNoiseConfig):
        noise = jax.random.uniform(
            subkey, shape=obs.shape, minval=noise_cfg.n_min, maxval=noise_cfg.n_max
        )
    elif isinstance(noise_cfg, GaussianNoiseConfig):
        noise = noise_cfg.mean + noise_cfg.std * jax.random.normal(subkey, shape=obs.shape)
    else:
        return obs

    if noise_cfg.operation == "add":
        return obs + noise
    elif noise_cfg.operation == "scale":
        return obs * noise
    elif noise_cfg.operation == "abs":
        return noise
    return obs