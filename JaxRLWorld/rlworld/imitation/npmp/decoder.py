"""NPMP decoder — π(a_t | s_t, z_t).

Three-layer MLP that maps the proprioceptive state plus the latent z
to an action *mean*. The action standard deviation is a state-
independent learnable per-action-dim parameter (init 0.0 → unit std);
the full Gaussian policy is therefore (mean, exp(log_std)).

Distillation-time training optimises the Gaussian negative log-
likelihood between the decoder mean and the expert mean; ``log_std``
is included in the parameter set so the loss can adapt the scale to
match the noise level of the BC labels.
"""

from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["NPMPDecoder"]


class NPMPDecoder(eqx.Module):
    layers: tuple
    head: eqx.nn.Linear
    log_std: jax.Array  # (action_dim,) — learnable, state-independent

    s_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(
        self,
        s_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (512, 256, 128),
        log_std_init: float = 0.0,
        *,
        key: jax.Array,
    ):
        self.s_dim = s_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        keys = jax.random.split(key, len(hidden) + 1)
        layers = []
        in_dim = s_dim + latent_dim
        for h, k in zip(hidden, keys[:-1]):
            layers.append(eqx.nn.Linear(in_dim, h, key=k))
            in_dim = h
        self.layers = tuple(layers)
        self.head = eqx.nn.Linear(in_dim, action_dim, key=keys[-1])

        self.log_std = jnp.full((action_dim,), log_std_init, dtype=jnp.float32)

    def mean(self, s_t: jax.Array, z_t: jax.Array) -> jax.Array:
        """Return action mean only (used by inference path)."""
        h = jnp.concatenate([s_t, z_t], axis=-1)
        for layer in self.layers:
            h = jax.nn.elu(layer(h))
        return self.head(h)

    def __call__(
        self,
        s_t: jax.Array,
        z_t: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return (action_mean, action_log_std). log_std is broadcast-ready."""
        return self.mean(s_t, z_t), self.log_std
