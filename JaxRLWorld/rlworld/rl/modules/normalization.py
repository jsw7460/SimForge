from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


class EmpiricalNormalization(eqx.Module):
    """Running mean/std normalization for observations."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array
    epsilon: float = eqx.field(static=True)
    shape: Tuple[int, ...] = eqx.field(static=True)

    def __init__(self, shape: int, epsilon: float = 1e-2):
        self.shape = (shape,)
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = jnp.array(1e-4)
        self.epsilon = epsilon

    def update(self, x: jax.Array) -> "EmpiricalNormalization":
        """Update running statistics with new batch."""
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        return eqx.tree_at(
            lambda s: (s.mean, s.var, s.count),
            self,
            (new_mean, new_var, total_count),
        )

    def normalize(self, x: jax.Array) -> jax.Array:
        """Normalize input using running statistics."""
        std = jnp.sqrt(self.var)
        return (x - self.mean) / (std + self.epsilon)  # mjlab 방식

    def unnormalize(self, x: jax.Array) -> jax.Array:
        """Inverse of :meth:`normalize` — recovers raw-space values from
        a normalized input. Used by the PPO value normalizer to convert
        critic outputs (normalized space) back to raw return space for
        GAE / storage / bootstrap.
        """
        std = jnp.sqrt(self.var)
        return x * (std + self.epsilon) + self.mean

    def __call__(self, x: jax.Array) -> jax.Array:
        """Alias for normalize."""
        return self.normalize(x)
