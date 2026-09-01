"""Tokenizer for :class:`SpaceTimeTransformer`.

Converts the flat observation vector
``[proprio (D_p,), future_window (T * B_tracked * D_ref,)]`` into a
token grid of shape ``(T + 1, B_all, D_embed)`` where:

* ``t = 0`` row holds **one shared** projection of the full proprio
  vector, broadcast across every body. All bodies start from the same
  proprio embedding; body identity is then injected by the encoder's
  positional embedding (``LearnedPE`` or ``TraversalPE``) before the
  first attention layer.
* ``t = 1..T`` rows hold a **shared** projection of each future-reference
  feature vector, scattered into the tracked-body slots. Untracked
  bodies stay as zero tokens at future time steps and only receive
  information via spatial attention.

Processes a single observation per call; ``jax.vmap`` is applied at
the actor/critic level when batched input is needed.
"""

from __future__ import annotations

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["SpaceTimeTokenizer"]


def _make_orthogonal_linear(
    in_dim: int,
    out_dim: int,
    key: jax.Array,
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
                "shared-Linear tokenizer (single Linear per kind is hardcoded). "
                "Pass hidden_dim=None until this lands."
            )

        self.num_bodies_all = num_bodies_all
        self.num_bodies_tracked = len(tracked_body_indices)
        self.tracked_body_indices = jnp.array(
            list(tracked_body_indices),
            dtype=jnp.int32,
        )
        self.proprio_dim = proprio_dim
        self.num_future_frames = num_future_frames
        self.ref_feature_dim = ref_feature_dim
        self.embed_dim = embed_dim

        key_p, key_r = jax.random.split(key)
        self.proprio_proj = _make_orthogonal_linear(proprio_dim, embed_dim, key_p)
        self.ref_proj = _make_orthogonal_linear(ref_feature_dim, embed_dim, key_r)

    def __call__(self, observation: jax.Array) -> jax.Array:
        """Tokenize a single (unbatched) flat observation.

        Args:
            observation: ``(proprio_dim + T * B_tracked * D_ref,)``.

        Returns:
            tokens: ``(T + 1, num_bodies_all, embed_dim)`` (or ``(1, ...)``
            when ``num_future_frames == 0``).
        """
        proprio = observation[: self.proprio_dim]

        # Single proprio embedding shared across bodies; body identity
        # comes from the encoder's positional embedding.
        proprio_token = self.proprio_proj(proprio)  # (D,)
        proprio_tokens = jnp.broadcast_to(
            proprio_token,
            (self.num_bodies_all, self.embed_dim),
        )

        if self.num_future_frames == 0:
            return proprio_tokens[None]

        ref = observation[self.proprio_dim :].reshape(
            self.num_future_frames,
            self.num_bodies_tracked,
            self.ref_feature_dim,
        )
        # Apply shared ref Linear to every (t, b) ref vector.
        tracked_tokens = jax.vmap(jax.vmap(self.ref_proj))(ref)  # (T, B_tracked, D)

        ref_tokens = jnp.zeros(
            (self.num_future_frames, self.num_bodies_all, self.embed_dim),
            dtype=proprio_tokens.dtype,
        )
        ref_tokens = ref_tokens.at[:, self.tracked_body_indices, :].set(
            tracked_tokens,
        )

        return jnp.concatenate([proprio_tokens[None], ref_tokens], axis=0)
