"""Tokenizer for :class:`SpaceTimeTransformer`.

Converts the flat observation vector
``[proprio (D_p,), future_window (T * B_tracked * D_ref,)]`` into a
token grid of shape ``(T + 1, B_all, D_embed)`` where:

* ``t = 0`` row holds per-body projections of the full proprio vector
  (one learned Linear per kinematic-tree body, but stored as a single
  *stacked* Linear so the forward compiles to one batched matmul).
* ``t = 1..T`` rows hold per-body projections of the future-reference
  features for the *tracked* subset of bodies. Untracked bodies stay
  as zero tokens at future time steps and only receive information via
  spatial attention in the encoder.

When ``num_future_frames == 0`` the future branch is skipped entirely
and the tokenizer returns a ``(1, B_all, D)`` grid — the whole stack
then reduces to a pure body-axis transformer, so the architecture
works for non-tracking presets as well.

The implementation uses :func:`equinox.filter_vmap` over the constructor
to build a single Linear whose ``weight`` and ``bias`` carry an extra
leading body dimension; the forward applies ``jax.vmap(lin)(...)`` over
that dimension. This avoids unrolling a Python ``for`` loop over bodies
at trace time, which is what made the previous implementation produce
a multi-GB XLA HLO graph for B_all > ~10.
"""
from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


__all__ = ["SpaceTimeTokenizer"]


def _make_orthogonal_linear(
    in_dim: int, out_dim: int, key: jax.Array,
) -> eqx.nn.Linear:
    """Linear with QR-orthogonal weight init and zero bias."""
    k_lin, k_ortho = jax.random.split(key)
    lin = eqx.nn.Linear(in_dim, out_dim, key=k_lin)
    max_dim = max(in_dim, out_dim)
    q, _ = jnp.linalg.qr(jax.random.normal(k_ortho, shape=(max_dim, max_dim)))
    new_weight = q[:out_dim, :in_dim]
    lin = eqx.tree_at(lambda l: l.weight, lin, new_weight)
    lin = eqx.tree_at(lambda l: l.bias, lin, jnp.zeros_like(lin.bias))
    return lin


class SpaceTimeTokenizer(eqx.Module):
    proprio_proj: eqx.nn.Linear
    ref_proj: eqx.nn.Linear

    num_bodies_all: int = eqx.field(static=True)
    num_bodies_tracked: int = eqx.field(static=True)
    tracked_body_indices: jax.Array
    proprio_dim: int = eqx.field(static=True)
    num_future_frames: int = eqx.field(static=True)
    ref_feature_dim: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_bodies_all: int,
        tracked_body_indices: Sequence[int],
        proprio_dim: int,
        num_future_frames: int,
        ref_feature_dim: int,
        embed_dim: int,
        hidden_dim: int | None = None,
        *,
        key: jax.Array,
    ):
        if hidden_dim is not None:
            raise NotImplementedError(
                "tokenizer_hidden_dim is currently unsupported by the "
                "vmap-stacked tokenizer (single Linear per body is hardcoded). "
                "Pass hidden_dim=None until this lands."
            )

        self.num_bodies_all = num_bodies_all
        self.num_bodies_tracked = len(tracked_body_indices)
        self.tracked_body_indices = jnp.array(
            list(tracked_body_indices), dtype=jnp.int32,
        )
        self.proprio_dim = proprio_dim
        self.num_future_frames = num_future_frames
        self.ref_feature_dim = ref_feature_dim
        self.embed_dim = embed_dim

        key_p, key_r = jax.random.split(key)

        # Stacked Linear: B_all separate (proprio_dim → embed_dim) Linears
        # whose weights are stored in a single tensor of shape
        # (B_all, embed_dim, proprio_dim). Forward applies via vmap.
        @eqx.filter_vmap
        def make_proprio(k):
            return _make_orthogonal_linear(proprio_dim, embed_dim, k)

        self.proprio_proj = make_proprio(
            jax.random.split(key_p, num_bodies_all),
        )

        # Stacked ref Linear: B_tracked separate (ref_feature_dim → embed_dim)
        # Linears. Build at least one even when no future window so the
        # field has a concrete shape; the forward path skips it when
        # num_future_frames == 0.
        n_tracked_for_init = max(self.num_bodies_tracked, 1)

        @eqx.filter_vmap
        def make_ref(k):
            return _make_orthogonal_linear(ref_feature_dim, embed_dim, k)

        self.ref_proj = make_ref(jax.random.split(key_r, n_tracked_for_init))

    def __call__(self, observation: jax.Array) -> jax.Array:
        """Tokenize a single flat observation.

        Args:
            observation: ``(proprio_dim + T * B_tracked * D_ref,)`` — unbatched.

        Returns:
            tokens: ``(T + 1, num_bodies_all, embed_dim)``. When
            ``num_future_frames == 0``, returns
            ``(1, num_bodies_all, embed_dim)``.
        """
        proprio = observation[: self.proprio_dim]

        # Apply each per-body Linear to the SAME proprio vector via vmap
        # over the stacked Linear's body axis. Compiles to one batched
        # matmul: (B_all, D, D_p) · (D_p,) → (B_all, D).
        proprio_tokens = jax.vmap(lambda lin: lin(proprio))(self.proprio_proj)

        if self.num_future_frames == 0:
            return proprio_tokens[None]

        ref = observation[self.proprio_dim:].reshape(
            self.num_future_frames, self.num_bodies_tracked, self.ref_feature_dim,
        )
        # (T, B_tracked, D_ref) → (B_tracked, T, D_ref) so the body axis
        # aligns with the stacked Linear's leading dim.
        ref_bt = jnp.transpose(ref, (1, 0, 2))

        def per_body(lin, body_feats):
            # body_feats: (T, D_ref) → (T, D_embed)
            return jax.vmap(lin)(body_feats)

        # vmap over body: stacked Linear (B_tracked) and ref_bt (B_tracked, T, D_ref)
        tracked_tokens_bt = jax.vmap(per_body)(self.ref_proj, ref_bt)
        tracked_tokens = jnp.transpose(tracked_tokens_bt, (1, 0, 2))

        ref_tokens = jnp.zeros(
            (self.num_future_frames, self.num_bodies_all, self.embed_dim),
            dtype=proprio_tokens.dtype,
        )
        ref_tokens = ref_tokens.at[:, self.tracked_body_indices, :].set(
            tracked_tokens,
        )

        return jnp.concatenate([proprio_tokens[None], ref_tokens], axis=0)
