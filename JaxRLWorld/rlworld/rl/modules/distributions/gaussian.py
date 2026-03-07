
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


class GaussianDistribution(eqx.Module):
    """
    Diagonal Gaussian distribution for continuous actions.
    """
    mean: jax.Array
    std: jax.Array
    is_squashed: bool = eqx.field(static=True, default=False)

    def __init__(self, mean: jax.Array, std: jax.Array):
        self.mean = mean
        self.std = std

    @property
    def variance(self) -> jax.Array:
        return self.std ** 2

    @property
    def stddev(self) -> jax.Array:
        return self.std

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample from the distribution."""
        noise = jax.random.normal(key, shape=self.mean.shape)
        return self.mean + self.std * noise

    def rsample(self, key: jax.Array) -> jax.Array:
        """Reparameterized sample (same as sample for Gaussian)."""
        return self.sample(key)

    def log_prob(self, actions: jax.Array) -> jax.Array:
        """
        Compute log probability of actions.

        Returns sum over action dimensions.
        """
        var = self.variance
        log_scale = jnp.log(self.std)
        log_prob = -0.5 * (
            ((actions - self.mean) ** 2) / var
            + 2 * log_scale
            + jnp.log(2 * jnp.pi)
        )
        return log_prob.sum(axis=-1)

    def entropy(self) -> jax.Array:
        """
        Compute entropy of the distribution.

        Returns sum over action dimensions.
        """
        return 0.5 * (1 + jnp.log(2 * jnp.pi) + 2 * jnp.log(self.std)).sum(axis=-1)

    def kl_divergence(self, other: "GaussianDistribution") -> jax.Array:
        """KL divergence KL(self || other)."""
        var_self = self.variance
        var_other = other.variance

        kl = (
            jnp.log(other.std / self.std)
            + (var_self + (self.mean - other.mean) ** 2) / (2 * var_other)
            - 0.5
        )
        return kl.sum(axis=-1)


class SquashedGaussianDistribution(eqx.Module):
    """
    Gaussian distribution with tanh squashing for bounded actions.

    Used in SAC and similar algorithms.
    """
    mean: jax.Array
    std: jax.Array
    _eps: float = eqx.field(static=True, default=1e-4)
    is_squashed: bool = eqx.field(static=True, default=True)

    def __init__(self, mean: jax.Array, std: jax.Array, eps: float = 1e-4):
        self.mean = mean
        self.std = std
        self._eps = eps

    @property
    def stddev(self) -> jax.Array:
        return self.std

    def sample_raw(self, key: jax.Array) -> jax.Array:
        """Sample pre-tanh (raw) action. Use for storage in PPO."""
        noise = jax.random.normal(key, shape=self.mean.shape)
        return self.mean + self.std * noise

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample and squash through tanh."""
        return jnp.tanh(self.sample_raw(key))

    def rsample_with_log_prob(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Reparameterized sample with log probability.

        More numerically stable than separate sample + log_prob.
        """
        noise = jax.random.normal(key, shape=self.mean.shape)
        pre_tanh = self.mean + self.std * noise
        action = jnp.tanh(pre_tanh)

        # Log prob with tanh correction
        log_prob = self._log_prob_from_pre_tanh(pre_tanh)

        return action, log_prob

    def log_prob_raw(self, raw_actions: jax.Array) -> jax.Array:
        """Compute log prob from pre-tanh (raw) actions. No arctanh needed.

        This is the numerically stable path used during PPO updates,
        matching Brax's approach of always operating on pre-tanh actions.
        """
        return self._log_prob_from_pre_tanh(raw_actions)

    def log_prob(self, actions: jax.Array) -> jax.Array:
        """
        Compute log probability of squashed actions.

        WARNING: This uses arctanh inversion which is numerically unstable
        near ±1. Prefer log_prob_raw() with pre-tanh actions when possible.
        """
        # Inverse tanh (atanh)
        actions_clipped = jnp.clip(actions, -1 + self._eps, 1 - self._eps)
        pre_tanh = jnp.arctanh(actions_clipped)

        return self._log_prob_from_pre_tanh(pre_tanh)

    def _log_prob_from_pre_tanh(self, pre_tanh: jax.Array) -> jax.Array:
        """Compute log prob given pre-tanh values."""
        # Gaussian log prob
        var = self.std ** 2
        log_scale = jnp.log(self.std)
        gaussian_log_prob = -0.5 * (
            ((pre_tanh - self.mean) ** 2) / var
            + 2 * log_scale
            + jnp.log(2 * jnp.pi)
        )

        # Tanh correction: log|det(d tanh / d pre_tanh)|
        # = sum(log(1 - tanh^2(pre_tanh)))
        # Numerically stable version: 2 * (log(2) - pre_tanh - softplus(-2 * pre_tanh))
        log_det_jacobian = 2 * (
            jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2 * pre_tanh)
        )

        return (gaussian_log_prob - log_det_jacobian).sum(axis=-1)

    def entropy(self) -> jax.Array:
        """
        Approximate entropy (Gaussian entropy, ignoring squashing).

        Exact entropy of squashed Gaussian has no closed form.
        """
        return 0.5 * (1 + jnp.log(2 * jnp.pi) + 2 * jnp.log(self.std)).sum(axis=-1)


# ==================== Utility Functions ====================

def sample_action(
    dist: GaussianDistribution | SquashedGaussianDistribution,
    key: jax.Array,
    deterministic: bool = False,
) -> jax.Array:
    """
    Sample action from distribution.

    Args:
        dist: Action distribution
        key: JAX random key
        deterministic: If True, return mean action

    Returns:
        Sampled or mean action
    """
    if deterministic:
        if isinstance(dist, SquashedGaussianDistribution):
            return jnp.tanh(dist.mean)
        return dist.mean
    return dist.sample(key)
