"""
TD-MPC2 math utilities (JAX).

Implements discrete regression (two-hot encoding), symmetric log/exp transforms,
SimNorm, and Gumbel-Softmax sampling used throughout TD-MPC2.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ==================== Symmetric Log/Exp ====================


def symlog(x: jax.Array) -> jax.Array:
    """Symmetric logarithmic transform. Adapted from DreamerV3."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    """Symmetric exponential transform (inverse of symlog)."""
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


# ==================== Discrete Regression (Two-Hot) ====================


class TwoHotConfig(NamedTuple):
    """Configuration for two-hot encoding."""

    num_bins: int
    vmin: float
    vmax: float
    bin_size: float


def make_two_hot_config(num_bins: int, vmin: float, vmax: float) -> TwoHotConfig:
    """Create two-hot configuration.

    Discrete regression needs at least two bins: with one, ``soft_ce`` is
    the log-softmax of a single logit, identically zero, so the reward
    and value heads would receive no gradient at all.
    """
    if num_bins < 2:
        raise ValueError(f"two-hot regression needs num_bins >= 2, got {num_bins}")
    if not vmax > vmin:
        raise ValueError(f"two-hot regression needs vmax > vmin, got vmin={vmin}, vmax={vmax}")
    bin_size = (vmax - vmin) / (num_bins - 1)
    return TwoHotConfig(num_bins=num_bins, vmin=vmin, vmax=vmax, bin_size=bin_size)


def two_hot(x: jax.Array, cfg: TwoHotConfig) -> jax.Array:
    """
    Convert scalars to soft two-hot encoded targets for discrete regression.

    Args:
        x: Scalar values [batch_size, 1] or [batch_size]
        cfg: Two-hot configuration

    Returns:
        Soft two-hot encoded targets [batch_size, num_bins]
    """
    x_squeezed = x.squeeze(-1) if x.ndim > 1 else x
    x_clamped = jnp.clip(symlog(x_squeezed), cfg.vmin, cfg.vmax)

    # The lower bin is clipped to num_bins - 2 so the upper bin exists,
    # and the offset is measured from the clipped index: at the upper
    # bound the position is num_bins - 1, the lower bin num_bins - 2 and
    # the offset 1, so the whole mass lands on the last bin. Measuring
    # the offset from the unclipped index put it on the second-to-last
    # bin for every target at or above vmax.
    position = (x_clamped - cfg.vmin) / cfg.bin_size
    bin_idx = jnp.clip(jnp.floor(position).astype(jnp.int32), 0, cfg.num_bins - 2)
    bin_offset = position - bin_idx.astype(jnp.float32)

    batch_size = x_clamped.shape[0]
    soft_two_hot = jnp.zeros((batch_size, cfg.num_bins), dtype=x.dtype)

    # Scatter lower bin
    rows = jnp.arange(batch_size)
    soft_two_hot = soft_two_hot.at[rows, bin_idx].set(1.0 - bin_offset)
    soft_two_hot = soft_two_hot.at[rows, bin_idx + 1].set(bin_offset)

    return soft_two_hot


def two_hot_inv(x: jax.Array, cfg: TwoHotConfig) -> jax.Array:
    """
    Convert soft two-hot encoded vectors back to scalars.

    Args:
        x: Logits [batch_size, num_bins] or [num_q, batch_size, num_bins]
        cfg: Two-hot configuration

    Returns:
        Scalar values [batch_size, 1] or [num_q, batch_size, 1]
    """
    dreg_bins = jnp.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, dtype=x.dtype)
    probs = jax.nn.softmax(x, axis=-1)
    val = jnp.sum(probs * dreg_bins, axis=-1, keepdims=True)
    return symexp(val)


# ==================== Cross Entropy Loss ====================


def soft_ce(pred: jax.Array, target: jax.Array, cfg: TwoHotConfig) -> jax.Array:
    """
    Cross entropy loss between predictions and soft two-hot targets.

    Args:
        pred: Logits [batch_size, num_bins]
        target: Scalar targets [batch_size, 1] or [batch_size]
        cfg: Two-hot configuration

    Returns:
        Loss [batch_size, 1]
    """
    log_pred = jax.nn.log_softmax(pred, axis=-1)
    target_encoded = two_hot(target, cfg)
    return -(target_encoded * log_pred).sum(axis=-1, keepdims=True)


# ==================== Gaussian Policy ====================


def log_std_transform(x: jax.Array, log_std_min: float, log_std_dif: float) -> jax.Array:
    """Transform raw network output to bounded log standard deviation."""
    return log_std_min + 0.5 * log_std_dif * (jnp.tanh(x) + 1.0)


def gaussian_logprob(eps: jax.Array, log_std: jax.Array) -> jax.Array:
    """
    Compute Gaussian log probability.

    Args:
        eps: Noise samples (z-scores)
        log_std: Log standard deviation

    Returns:
        Log probability [batch_size, 1]
    """
    residual = -0.5 * eps**2 - log_std
    log_prob = residual - 0.9189385175704956  # -0.5 * log(2*pi)
    return log_prob.sum(axis=-1, keepdims=True)


def squash(mu: jax.Array, pi: jax.Array, log_pi: jax.Array):
    """
    Apply tanh squashing function to actions.

    Returns:
        (squashed_mu, squashed_pi, adjusted_log_pi)
    """
    mu = jnp.tanh(mu)
    pi = jnp.tanh(pi)
    # Correction for tanh squashing
    squash_correction = jnp.log(jax.nn.relu(1.0 - pi**2) + 1e-6)
    log_pi = log_pi - squash_correction.sum(axis=-1, keepdims=True)
    return mu, pi, log_pi


# ==================== Gumbel-Softmax ====================


def gumbel_softmax_sample(
    logits: jax.Array,
    key: jax.Array,
    temperature: float = 1.0,
) -> jax.Array:
    """
    Sample from Gumbel-Softmax distribution and return hard index.

    Args:
        logits: Log-probabilities or scores [num_samples]
        key: JAX random key
        temperature: Temperature parameter

    Returns:
        Sampled index (scalar)
    """
    # Gumbel(0,1) noise
    u = jax.random.uniform(key, logits.shape, minval=1e-8, maxval=1.0)
    gumbels = -jnp.log(-jnp.log(u))
    y = (jnp.log(logits + 1e-8) + gumbels) / temperature
    return jnp.argmax(y, axis=-1)
