"""AR(1) prior over latent z for the NPMP module.

Implements the prior from Merel et al. ICLR 2019, eq. (2):

    z_t = alpha * z_{t-1} + sqrt(1 - alpha^2) * eps,    eps ~ N(0, I)

so that marginally z_t ~ N(0, I). At an episode boundary (no z_{t-1}
history) the prior collapses to the standard normal N(0, I) — the
caller passes ``episode_start=True`` for that step.

The prior carries no learnable parameters; ``alpha`` is a fixed
hyperparameter (paper default 0.95).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["AR1Prior"]


class AR1Prior(eqx.Module):
    """First-order autoregressive Gaussian prior on the latent."""

    alpha: float = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, latent_dim: int, alpha: float = 0.95):
        self.latent_dim = latent_dim
        self.alpha = alpha

    @property
    def ar_std(self) -> float:
        """Per-dim std of the AR(1) innovation, sqrt(1 - alpha^2)."""
        return float((1.0 - self.alpha**2) ** 0.5)

    def mean_std(
        self,
        z_prev: jax.Array,
        episode_start: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return (mean, std) of p_z(z_t | z_{t-1}) per latent dim.

        Shapes: ``z_prev`` is ``(D_z,)`` (unbatched — caller vmaps as
        needed). ``episode_start`` is a scalar bool/array; when truthy
        the prior is N(0, I).
        """
        ar_mean = self.alpha * z_prev
        ar_std = jnp.full_like(z_prev, self.ar_std)
        ep = episode_start.astype(z_prev.dtype)
        mean = jnp.where(ep > 0.5, jnp.zeros_like(z_prev), ar_mean)
        std = jnp.where(ep > 0.5, jnp.ones_like(z_prev), ar_std)
        return mean, std

    def log_prob(
        self,
        z_t: jax.Array,
        z_prev: jax.Array,
        episode_start: jax.Array,
    ) -> jax.Array:
        """Diagonal Gaussian log-prob, summed over latent dim."""
        mean, std = self.mean_std(z_prev, episode_start)
        var = std**2
        return -0.5 * jnp.sum(
            ((z_t - mean) ** 2) / var + jnp.log(2.0 * jnp.pi * var),
            axis=-1,
        )

    def sample(
        self,
        z_prev: jax.Array,
        episode_start: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        mean, std = self.mean_std(z_prev, episode_start)
        return mean + std * jax.random.normal(key, z_prev.shape)
