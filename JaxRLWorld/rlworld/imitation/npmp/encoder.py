"""NPMP encoder — q(z_t | z_{t-1}, x_t).

Two-layer MLP with ELU activations whose final layer feeds twin linear
heads producing the mean and log-std of a diagonal Gaussian over z_t.

Input layout (concatenated along the feature axis):
    [z_{t-1} (D_z,)] ++ [x_t (D_x,)]    →    (D_z + D_x,)

``x_t`` is the future motion-reference window in robot-anchor frame,
flattened to ``(T_future * num_tracked_bodies * 9,)`` by the caller —
the encoder treats it as an opaque feature vector.

The log-std output is clipped to ``[log_std_min, log_std_max]`` so
that gradient steps cannot collapse the posterior to a delta or blow
it up; the same trick is used by NPMP and most subsequent imitation
work (e.g. OmniH2O).
"""

from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["NPMPEncoder"]


class NPMPEncoder(eqx.Module):
    layers: tuple
    mean_head: eqx.nn.Linear
    log_std_head: eqx.nn.Linear

    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    x_dim: int = eqx.field(static=True)

    def __init__(
        self,
        x_dim: int,
        latent_dim: int,
        hidden: Sequence[int] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        *,
        key: jax.Array,
    ):
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        keys = jax.random.split(key, len(hidden) + 2)
        layers = []
        in_dim = latent_dim + x_dim
        for h, k in zip(hidden, keys[:-2]):
            layers.append(eqx.nn.Linear(in_dim, h, key=k))
            in_dim = h
        self.layers = tuple(layers)
        self.mean_head = eqx.nn.Linear(in_dim, latent_dim, key=keys[-2])
        self.log_std_head = eqx.nn.Linear(in_dim, latent_dim, key=keys[-1])

    def __call__(
        self,
        z_prev: jax.Array,
        x_t: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Single-element forward (unbatched). Returns (mean, log_std)."""
        h = jnp.concatenate([z_prev, x_t], axis=-1)
        for layer in self.layers:
            h = jax.nn.elu(layer(h))
        mean = self.mean_head(h)
        log_std = jnp.clip(
            self.log_std_head(h),
            self.log_std_min,
            self.log_std_max,
        )
        return mean, log_std
