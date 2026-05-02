"""ELBO loss for NPMP distillation.

Maximises (Merel et al. ICLR 2019, eq. 3)::

    E_q[ Σ_t  log π(μ_E_t | s_t, z_t)
              + β · ( log p_z(z_t | z_{t-1}) − log q(z_t | z_{t-1}, x_t) ) ]

equivalently minimises::

    L = −log π(μ_E | s, z)            (Gaussian NLL)
        + β · KL( q(z | z_prev, x)  ‖  p_z(z | z_prev) )

The KL is computed in closed form between two diagonal Gaussians (no
sampling needed for the KL term — the reparameterised z_t enters only
through the decoder NLL). Both terms are summed over the trailing
feature dim and averaged over the (batch, time) axes.

The BC target is the noiseless expert mean ``μ_E(s_t)`` logged during
rollout — DART convention. ``s_t`` here is the decoder proprio
observation; ``x_t`` is the encoder's future motion-reference window.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rlworld.imitation.npmp.module import NPMPModule, NPMPStepOutput


__all__ = ["NPMPBatch", "NPMPLossInfo", "npmp_elbo_loss"]


_LOG_2PI = math.log(2.0 * math.pi)


# ── Data structures ─────────────────────────────────────────────────


class NPMPBatch(NamedTuple):
    """One mini-batch of (B, T, ...) trajectory data."""

    s: jax.Array               # (B, T, D_s)   decoder proprio
    x: jax.Array               # (B, T, D_x)   encoder future-window (flattened)
    mu_E: jax.Array            # (B, T, A)     expert mean target
    episode_starts: jax.Array  # (B, T)        bool — True at episode/rollover start


class NPMPLossInfo(NamedTuple):
    """Auxiliary metrics returned alongside the loss for logging."""

    loss: jax.Array            # scalar — total loss minimised
    recon: jax.Array           # scalar — mean Gaussian NLL across (B, T)
    kl: jax.Array              # scalar — mean KL(q ‖ p_z) across (B, T)
    decoder_log_std: jax.Array # (A,)   — current decoder log std (per-action-dim)


# ── Helpers ─────────────────────────────────────────────────────────


def _gaussian_nll(
    target: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    """Per-element Gaussian NLL, summed over the trailing feature axis.

    Args:
        target:  ``(..., A)`` BC target μ_E.
        mean:    ``(..., A)`` decoder mean.
        log_std: ``(..., A)`` decoder log std (broadcast-compatible).

    Returns:
        Tensor of shape ``(...,)``.
    """
    var = jnp.exp(2.0 * log_std)
    nll = log_std + 0.5 * _LOG_2PI + 0.5 * ((target - mean) ** 2) / var
    return jnp.sum(nll, axis=-1)


def _diag_gaussian_kl(
    mu_q: jax.Array,
    log_std_q: jax.Array,
    mu_p: jax.Array,
    log_std_p: jax.Array,
) -> jax.Array:
    """Closed-form ``KL(N(μ_q, σ_q² I) ‖ N(μ_p, σ_p² I))``.

    Per-dim formula::

        KL = log(σ_p/σ_q) + (σ_q² + (μ_q − μ_p)²) / (2 σ_p²) − 1/2

    Summed over the trailing feature axis.
    """
    var_q = jnp.exp(2.0 * log_std_q)
    var_p = jnp.exp(2.0 * log_std_p)
    kl = (
        log_std_p
        - log_std_q
        + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p)
        - 0.5
    )
    return jnp.sum(kl, axis=-1)


# ── Loss ────────────────────────────────────────────────────────────


def npmp_elbo_loss(
    module: NPMPModule,
    batch: NPMPBatch,
    beta: float,
    key: jax.Array,
) -> tuple[jax.Array, NPMPLossInfo]:
    """Compute ELBO loss for a (B, T) trajectory batch.

    The forward pass runs ``module.encode_decode_trajectory`` along
    each env's time axis (via :func:`jax.lax.scan` inside the module),
    vmapped across the batch axis. The returned ``NPMPStepOutput`` has
    a ``(B, T, ...)`` shape per field. Reconstruction NLL and KL are
    then computed elementwise and averaged over (B, T).
    """
    B = batch.s.shape[0]
    keys = jax.random.split(key, B)

    outputs: NPMPStepOutput = jax.vmap(module.encode_decode_trajectory)(
        batch.s, batch.x, batch.episode_starts, keys,
    )

    # Reconstruction: −log π(μ_E | μ̂, σ_a). Decoder log_std is
    # state-independent (A,) but scan stacks it to (T, A) per env, then
    # vmap to (B, T, A). Already broadcast-compatible with action_mean.
    recon = _gaussian_nll(
        batch.mu_E, outputs.action_mean, outputs.action_log_std,
    )

    kl = _diag_gaussian_kl(
        outputs.q_mean, outputs.q_log_std,
        outputs.p_mean, outputs.p_log_std,
    )

    recon_mean = jnp.mean(recon)
    kl_mean = jnp.mean(kl)
    loss = recon_mean + beta * kl_mean

    info = NPMPLossInfo(
        loss=loss,
        recon=recon_mean,
        kl=kl_mean,
        decoder_log_std=module.decoder.log_std,
    )
    return loss, info
