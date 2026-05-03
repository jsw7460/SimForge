"""NPMPModule — encoder + AR(1) prior + decoder bundled together.

Provides three forward modes:

* :meth:`step` — one reparameterised encode-decode step (unbatched).
  All quantities needed by the ELBO loss are returned in one shot:
  posterior parameters, prior parameters, sampled latent, decoder
  output. Caller typically vmaps this across a batch axis.

* :meth:`encode_decode_trajectory` — runs :meth:`step` along a
  length-``T`` trajectory via :func:`jax.lax.scan`, threading
  ``z_{t-1}`` through time. Used during training: the resulting
  per-step outputs feed directly into the ELBO. Episode boundaries
  inside the trajectory are handled by ``episode_starts`` — at those
  positions the encoder sees ``z_prev = 0`` and the prior is N(0, I).

* :meth:`act` — deterministic inference (encoder mean, decoder mean,
  no sampling). Used at deploy time when running the distilled motor
  primitive in the environment.

The encoder/decoder/prior are stored as plain ``eqx.Module`` fields,
so :func:`equinox.partition` / :func:`equinox.combine` slice the
trainable parameters cleanly when wiring up an optimiser.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.imitation.npmp.decoder import NPMPDecoder
from rlworld.imitation.npmp.encoder import NPMPEncoder
from rlworld.imitation.npmp.prior import AR1Prior


__all__ = ["NPMPModule", "NPMPStepOutput"]


class NPMPStepOutput(NamedTuple):
    """Per-step bundle of outputs needed by the ELBO loss."""

    z_t: jax.Array            # sampled latent       (D_z,)
    q_mean: jax.Array         # encoder mean         (D_z,)
    q_log_std: jax.Array      # encoder log std      (D_z,)
    p_mean: jax.Array         # AR(1) prior mean     (D_z,)
    p_log_std: jax.Array      # AR(1) prior log std  (D_z,)
    action_mean: jax.Array    # decoder mean         (A,)
    action_log_std: jax.Array # decoder log std      (A,)


class NPMPModule(eqx.Module):
    encoder: NPMPEncoder
    decoder: NPMPDecoder
    prior: AR1Prior

    def __init__(
        self,
        s_dim: int,
        x_dim: int,
        action_dim: int,
        latent_dim: int = 60,
        encoder_hidden: Sequence[int] = (256, 256),
        decoder_hidden: Sequence[int] = (512, 256, 128),
        ar1_alpha: float = 0.95,
        decoder_log_std_init: float = 0.0,
        *,
        key: jax.Array,
    ):
        k_enc, k_dec = jax.random.split(key)
        self.encoder = NPMPEncoder(
            x_dim=x_dim,
            latent_dim=latent_dim,
            hidden=encoder_hidden,
            key=k_enc,
        )
        self.decoder = NPMPDecoder(
            s_dim=s_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden=decoder_hidden,
            log_std_init=decoder_log_std_init,
            key=k_dec,
        )
        self.prior = AR1Prior(latent_dim=latent_dim, alpha=ar1_alpha)

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def action_dim(self) -> int:
        return self.decoder.action_dim

    # ------------------------------------------------------------------
    # Single-step (unbatched) forward
    # ------------------------------------------------------------------

    def step(
        self,
        z_prev: jax.Array,
        s_t: jax.Array,
        x_t: jax.Array,
        episode_start: jax.Array,
        key: jax.Array,
    ) -> NPMPStepOutput:
        """One reparameterised encode + decode step (unbatched).

        At an episode boundary the encoder sees ``z_prev = 0`` and the
        prior is N(0, I) — the AR(1) recursion only kicks in within a
        single contiguous motion trajectory.
        """
        z_prev_used = jnp.where(
            episode_start, jnp.zeros_like(z_prev), z_prev,
        )

        # Prior parameters (no learnable params).
        p_mean, p_std = self.prior.mean_std(z_prev_used, episode_start)
        p_log_std = jnp.log(p_std)

        # Encoder posterior.
        q_mean, q_log_std = self.encoder(z_prev_used, x_t)

        # Reparameterised sample.
        eps = jax.random.normal(key, q_mean.shape)
        z_t = q_mean + jnp.exp(q_log_std) * eps

        # Decoder.
        action_mean, action_log_std = self.decoder(s_t, z_t)

        return NPMPStepOutput(
            z_t=z_t,
            q_mean=q_mean,
            q_log_std=q_log_std,
            p_mean=p_mean,
            p_log_std=p_log_std,
            action_mean=action_mean,
            action_log_std=action_log_std,
        )

    # ------------------------------------------------------------------
    # Trajectory forward — used by the training loss.
    # ------------------------------------------------------------------

    def encode_decode_trajectory(
        self,
        s_seq: jax.Array,           # (T, D_s)
        x_seq: jax.Array,           # (T, D_x)
        episode_starts: jax.Array,  # (T,)
        key: jax.Array,
    ) -> NPMPStepOutput:
        """Run encode + decode along a length-T trajectory via ``lax.scan``.

        Returns an :class:`NPMPStepOutput` whose fields each carry a
        leading time axis of length ``T``. Caller is responsible for
        vmapping across an outer batch axis.
        """
        T = s_seq.shape[0]
        keys = jax.random.split(key, T)
        z_init = jnp.zeros((self.latent_dim,), dtype=s_seq.dtype)

        def scan_fn(z_prev, inputs):
            s_t, x_t, ep, k = inputs
            out = self.step(z_prev, s_t, x_t, ep, k)
            return out.z_t, out

        _, outputs = jax.lax.scan(
            scan_fn, z_init, (s_seq, x_seq, episode_starts, keys),
        )
        return outputs

    # ------------------------------------------------------------------
    # Deterministic inference path — used at deploy time.
    # ------------------------------------------------------------------

    def act_step_deterministic(
        self,
        z_prev: jax.Array,
        s_t: jax.Array,
        x_t: jax.Array,
        episode_start: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Stateful single-step deterministic inference (env-stepping path).

        Encoder uses its mean (no sampling), decoder uses its mean (no
        action noise). Episode boundary zeros ``z_prev`` so the AR(1)
        chain restarts from the prior origin. Caller threads ``z_t``
        forward as the next step's ``z_prev``.

        Returns ``(z_t, action_mean)``. Used by the viser policy
        wrapper and any non-jit-fused per-step deployment loop. When
        you also need encoder log-std diagnostics, see
        :meth:`eval_step`.
        """
        z_prev_used = jnp.where(
            episode_start, jnp.zeros_like(z_prev), z_prev,
        )
        q_mean, _ = self.encoder(z_prev_used, x_t)
        z_t = q_mean
        action_mean, _ = self.decoder(s_t, z_t)
        return z_t, action_mean

    def eval_step(
        self,
        z_prev: jax.Array,
        s_t: jax.Array,
        x_t: jax.Array,
        episode_start: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Diagnostic-rich variant of :meth:`act_step_deterministic`.

        Returns ``(z_t, action_mean, q_log_std)`` so the in-training
        evaluator can track encoder posterior spread alongside the
        latent norm and action output. ``q_log_std`` is the encoder's
        per-dim log-std at this step.
        """
        z_prev_used = jnp.where(
            episode_start, jnp.zeros_like(z_prev), z_prev,
        )
        q_mean, q_log_std = self.encoder(z_prev_used, x_t)
        z_t = q_mean
        action_mean, _ = self.decoder(s_t, z_t)
        return z_t, action_mean, q_log_std

    def act(
        self,
        s_seq: jax.Array,           # (T, D_s)
        x_seq: jax.Array,           # (T, D_x)
        episode_starts: jax.Array,  # (T,)
    ) -> jax.Array:
        """Deterministic per-step action mean along a trajectory.

        Encoder mean is used for ``z`` (no sampling); decoder mean is
        returned as the action. Episode boundaries reset ``z_prev`` to
        zero so the AR(1) chain always starts from the prior origin
        when a new motion begins.
        """
        z_init = jnp.zeros((self.latent_dim,), dtype=s_seq.dtype)

        def scan_fn(z_prev, inputs):
            s_t, x_t, ep = inputs
            z_prev_used = jnp.where(ep, jnp.zeros_like(z_prev), z_prev)
            q_mean, _ = self.encoder(z_prev_used, x_t)
            z_t = q_mean
            action_mean, _ = self.decoder(s_t, z_t)
            return z_t, action_mean

        _, actions = jax.lax.scan(
            scan_fn, z_init, (s_seq, x_seq, episode_starts),
        )
        return actions
