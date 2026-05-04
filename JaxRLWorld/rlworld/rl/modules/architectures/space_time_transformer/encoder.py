"""Factorized (space × time) transformer encoder.

TimeSformer / ViViT-style encoder that alternates attention between the
body axis (spatial) and the time axis (temporal), given a token grid of
shape ``(T + 1, B_all, D)`` produced by :class:`SpaceTimeTokenizer`.

Two structural priors can be plugged into the body axis:

* **Positional embedding** for body identity. Either
  :class:`LearnedPositionalEmbedding` (single learnable ``(B, D)`` table,
  no structural prior) or :class:`TraversalPositionalEmbedding`
  (SWAT-style; concatenates pre / in / post-order DFS lookups so bodies
  near each other in the tree end up at nearby indices in at least one
  ordering).
* **Spatial attention bias** via :class:`GraphRelationalEmbedding`.
  Static graph features (Laplacian / SPD / PPR) are projected per-head
  to a learnable ``(H, B, B)`` tensor and added to spatial attention
  scores. Soft, head-specific generalization of the old binary
  ``adjacency_mask`` (which is still supported and can be combined).

When a relational bias is configured, spatial attention is implemented
manually (still using the ``eqx.nn.MultiheadAttention`` module's
projection weights) because ``eqx.nn.MultiheadAttention`` only accepts
boolean masks. Temporal attention always goes through the eqx call
because it has no graph structure to inject.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.architectures.space_time_transformer.embeddings import (
    GraphRelationalEmbedding,
    LearnedPositionalEmbedding,
    TraversalPositionalEmbedding,
)

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


__all__ = [
    "FactorizedAttentionBlock",
    "JointAttentionBlock",
    "SpaceTimeTransformerEncoder",
]


def _spatial_self_attention_with_bias(
    mha: eqx.nn.MultiheadAttention,
    tokens: jax.Array,
    re_bias: jax.Array | None,
    bool_mask: jax.Array | None,
) -> jax.Array:
    """Manual self-attention reusing ``mha``'s projection weights.

    Adds ``re_bias`` (continuous, ``(H, B, B)``) and/or applies the
    boolean ``bool_mask`` (``(B, B)``, True = can attend) to the
    attention scores. Mirrors the pre-softmax math of
    ``eqx.nn.MultiheadAttention.__call__`` so weights behave identically
    when both ``re_bias`` and ``bool_mask`` are ``None``.

    Args:
        mha: parameter-holding attention module.
        tokens: ``(B, D)`` (unbatched single-time-step token grid).
        re_bias: ``(H, B, B)`` or ``None``.
        bool_mask: ``(B, B)`` or ``None``.

    Returns:
        ``(B, D)``.
    """
    H = mha.num_heads
    qk = mha.qk_size
    B, D = tokens.shape

    # Linear projections — apply per-row (eqx.nn.Linear is unbatched).
    q = jax.vmap(mha.query_proj)(tokens).reshape(B, H, qk)
    k = jax.vmap(mha.key_proj)(tokens).reshape(B, H, qk)
    v = jax.vmap(mha.value_proj)(tokens).reshape(B, H, qk)

    # scores: (H, B, B)
    scale = jnp.asarray(qk, dtype=q.dtype) ** -0.5
    scores = jnp.einsum("she,She->hsS", q, k) * scale

    if re_bias is not None:
        scores = scores + re_bias
    if bool_mask is not None:
        scores = jnp.where(bool_mask, scores, jnp.finfo(scores.dtype).min)

    weights = jax.nn.softmax(scores, axis=-1)
    # attn: (B, H, qk)
    attn = jnp.einsum("hsS,She->she", weights, v)
    attn = attn.reshape(B, H * qk)
    return jax.vmap(mha.output_proj)(attn)


class FactorizedAttentionBlock(eqx.Module):
    spatial_attn: eqx.nn.MultiheadAttention
    spatial_norm: eqx.nn.LayerNorm
    temporal_attn: eqx.nn.MultiheadAttention
    temporal_norm: eqx.nn.LayerNorm
    ffn_linear1: eqx.nn.Linear
    ffn_linear2: eqx.nn.Linear
    ffn_norm: eqx.nn.LayerNorm
    has_temporal: bool = eqx.field(static=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        has_temporal: bool = True,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 4)
        self.spatial_attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout,
            key=keys[0],
        )
        self.spatial_norm = eqx.nn.LayerNorm(embed_dim)
        self.temporal_attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout,
            key=keys[1],
        )
        self.temporal_norm = eqx.nn.LayerNorm(embed_dim)
        self.ffn_linear1 = eqx.nn.Linear(embed_dim, dim_feedforward, key=keys[2])
        self.ffn_linear2 = eqx.nn.Linear(dim_feedforward, embed_dim, key=keys[3])
        self.ffn_norm = eqx.nn.LayerNorm(embed_dim)
        self.has_temporal = has_temporal

    def __call__(
        self,
        tokens: jax.Array,
        spatial_mask: jax.Array | None,
        spatial_re_bias: jax.Array | None,
        *,
        key: jax.Array,
    ) -> jax.Array:
        """Apply one factorized block.

        Args:
            tokens: ``(T, B, D)`` token grid (unbatched).
            spatial_mask: ``(B, B)`` boolean mask; True = can attend.
                ``None`` = full attention (no hard locality constraint).
            spatial_re_bias: ``(H, B, B)`` continuous attention bias from
                a :class:`GraphRelationalEmbedding`. ``None`` = no bias.
                When provided, spatial attention is computed manually
                using the eqx-MHA's projection weights (so the same
                parameters are exercised either way).
            key: JAX PRNG key.

        Returns:
            ``(T, B, D)`` updated tokens.
        """
        T = tokens.shape[0]
        key_s, key_t, key_ffn = jax.random.split(key, 3)

        spatial_keys = jax.random.split(key_s, T)

        if spatial_re_bias is None:
            # Original eqx path — preserves exact pre-existing behavior
            # when no relational bias is requested.
            def spatial_step(t_tokens: jax.Array, k: jax.Array) -> jax.Array:
                a = self.spatial_attn(
                    query=t_tokens,
                    key_=t_tokens,
                    value=t_tokens,
                    mask=spatial_mask,
                    inference=False,
                    key=k,
                )
                return jax.vmap(self.spatial_norm)(t_tokens + a)
        else:
            # Manual path — we need to inject the continuous bias into
            # attention scores, which the eqx MHA's bool-only mask
            # interface can't express.
            def spatial_step(t_tokens: jax.Array, k: jax.Array) -> jax.Array:
                del k  # dropout=0 in this codebase
                a = _spatial_self_attention_with_bias(
                    self.spatial_attn, t_tokens, spatial_re_bias, spatial_mask,
                )
                return jax.vmap(self.spatial_norm)(t_tokens + a)

        tokens = jax.vmap(spatial_step)(tokens, spatial_keys)

        if self.has_temporal and T > 1:
            tokens_bt = jnp.transpose(tokens, (1, 0, 2))
            B = tokens_bt.shape[0]
            temporal_keys = jax.random.split(key_t, B)

            def temporal_step(b_tokens: jax.Array, k: jax.Array) -> jax.Array:
                a = self.temporal_attn(
                    query=b_tokens,
                    key_=b_tokens,
                    value=b_tokens,
                    mask=None,
                    inference=False,
                    key=k,
                )
                return jax.vmap(self.temporal_norm)(b_tokens + a)

            tokens_bt = jax.vmap(temporal_step)(tokens_bt, temporal_keys)
            tokens = jnp.transpose(tokens_bt, (1, 0, 2))

        def ffn(x: jax.Array) -> jax.Array:
            h = self.ffn_linear1(x)
            h = jax.nn.elu(h)
            return self.ffn_linear2(h)

        shape = tokens.shape
        flat = tokens.reshape(-1, shape[-1])
        ffn_out = jax.vmap(ffn)(flat)
        flat = jax.vmap(self.ffn_norm)(flat + ffn_out)
        return flat.reshape(shape)


class JointAttentionBlock(eqx.Module):
    """Single self-attention over the flattened ``(T*B,)`` sequence.

    Each ``(t, b)`` token attends to every other ``(t', b')`` token in
    one pass. More expressive than the factorized variant (cross-time-
    and-space dependencies are direct, not mediated by stacking layers)
    and on the GPU it usually wins on wall time too: factorized runs
    ``T + B`` separate small attentions per layer (poor fusion), while
    joint runs one batched attention of seq length ``T*B``.

    Same parameter set as :class:`FactorizedAttentionBlock`'s spatial
    half (one MHA + one FFN); the temporal MHA is dropped because there
    is no second attention pass.

    Mask / RE bias are accepted in the *body* shape ``(B, B)`` /
    ``(H, B, B)`` and broadcast across all time pairs at forward time
    (so the existing kinematic mask and graph-relational embedding plug
    in unchanged from the factorized path).
    """

    attn: eqx.nn.MultiheadAttention
    norm: eqx.nn.LayerNorm
    ffn_linear1: eqx.nn.Linear
    ffn_linear2: eqx.nn.Linear
    ffn_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout,
            key=keys[0],
        )
        self.norm = eqx.nn.LayerNorm(embed_dim)
        self.ffn_linear1 = eqx.nn.Linear(embed_dim, dim_feedforward, key=keys[1])
        self.ffn_linear2 = eqx.nn.Linear(dim_feedforward, embed_dim, key=keys[2])
        self.ffn_norm = eqx.nn.LayerNorm(embed_dim)

    def __call__(
        self,
        tokens: jax.Array,
        body_mask: jax.Array | None,
        body_re_bias: jax.Array | None,
        *,
        key: jax.Array,
    ) -> jax.Array:
        """Apply one joint-attention block.

        Args:
            tokens: ``(T, B, D)`` token grid (unbatched).
            body_mask: ``(B, B)`` boolean body-pair mask; broadcast across
                time pairs. ``None`` = full attention.
            body_re_bias: ``(H, B, B)`` continuous body-pair bias;
                broadcast across time pairs. ``None`` = no bias.
            key: JAX PRNG key.

        Returns:
            ``(T, B, D)``.
        """
        T, B, D = tokens.shape
        S = T * B
        flat = tokens.reshape(S, D)

        joint_mask = (
            jnp.tile(body_mask, (T, T)) if body_mask is not None else None
        )
        joint_re_bias = (
            jnp.tile(body_re_bias, (1, T, T))
            if body_re_bias is not None
            else None
        )

        if joint_re_bias is None:
            attn_out = self.attn(
                query=flat, key_=flat, value=flat,
                mask=joint_mask, inference=False, key=key,
            )
        else:
            attn_out = _spatial_self_attention_with_bias(
                self.attn, flat, joint_re_bias, joint_mask,
            )
        flat = jax.vmap(self.norm)(flat + attn_out)

        def ffn(x: jax.Array) -> jax.Array:
            h = self.ffn_linear1(x)
            h = jax.nn.elu(h)
            return self.ffn_linear2(h)

        ffn_out = jax.vmap(ffn)(flat)
        flat = jax.vmap(self.ffn_norm)(flat + ffn_out)
        return flat.reshape(T, B, D)


def _layer_call(
    layer,
    tokens: jax.Array,
    body_mask: jax.Array | None,
    body_re_bias: jax.Array | None,
    key: jax.Array,
) -> jax.Array:
    """Polymorphic layer dispatch — same signature works for both
    :class:`FactorizedAttentionBlock` and :class:`JointAttentionBlock`.
    """
    return layer(tokens, body_mask, body_re_bias, key=key)


# Gradient-checkpointed wrapper: forward stores only inputs, backward
# recomputes the layer's intermediate activations. Trades ~1.5x extra
# forward compute for ~50% lower activation memory, which is what lets
# us keep the PPO mini-batch large without OOM-ing the attention scores
# tensor at high num_envs.
_layer_call_ckpt = eqx.filter_checkpoint(_layer_call)


class SpaceTimeTransformerEncoder(eqx.Module):
    body_pe_module: LearnedPositionalEmbedding | TraversalPositionalEmbedding
    time_pe: jax.Array
    layers: tuple
    adjacency_mask: jax.Array | None
    relational_embedding: GraphRelationalEmbedding | None

    num_bodies_all: int = eqx.field(static=True)
    num_time_tokens: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    use_kinematic_mask: bool = eqx.field(static=True)
    use_checkpoint: bool = eqx.field(static=True)
    attention_mode: str = eqx.field(static=True)

    def __init__(
        self,
        num_bodies_all: int,
        num_time_tokens: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        kinematic_tree: "KinematicTree | None" = None,
        use_kinematic_mask: bool = True,
        use_checkpoint: bool = False,
        pe_type: str = "learned",
        use_relational_bias: bool = False,
        re_use_laplacian: bool = True,
        re_use_spd: bool = True,
        re_use_ppr: bool = True,
        re_ppr_alpha: float = 0.15,
        attention_mode: str = "factorized",
        *,
        key: jax.Array,
    ):
        if attention_mode not in ("factorized", "joint"):
            raise ValueError(
                f"Unknown attention_mode={attention_mode!r}. "
                "Expected 'factorized' or 'joint'."
            )
        self.num_bodies_all = num_bodies_all
        self.num_time_tokens = num_time_tokens
        self.embed_dim = embed_dim
        self.use_kinematic_mask = use_kinematic_mask
        self.use_checkpoint = use_checkpoint
        self.attention_mode = attention_mode

        key_pe, key_re, key_layers, key_tpe = jax.random.split(key, 4)

        # Body positional embedding (learned plain or SWAT-style).
        if pe_type == "learned":
            self.body_pe_module = LearnedPositionalEmbedding(
                num=num_bodies_all, embed_dim=embed_dim, key=key_pe,
            )
        elif pe_type == "traversal":
            if kinematic_tree is None:
                raise ValueError(
                    "pe_type='traversal' requires kinematic_tree to compute "
                    "pre/in/post-order DFS indices."
                )
            self.body_pe_module = TraversalPositionalEmbedding(
                kinematic_tree=kinematic_tree, embed_dim=embed_dim, key=key_pe,
            )
        else:
            raise ValueError(
                f"Unknown pe_type={pe_type!r}. Expected 'learned' or 'traversal'."
            )

        # Time positional embedding stays a simple learned (T, D) table —
        # the time axis has no graph structure to encode.
        self.time_pe = (
            jax.random.normal(key_tpe, (num_time_tokens, embed_dim)) * 0.02
        )

        # Spatial relational bias (optional, additive to attn scores).
        if use_relational_bias:
            if kinematic_tree is None:
                raise ValueError(
                    "use_relational_bias=True requires kinematic_tree."
                )
            self.relational_embedding = GraphRelationalEmbedding(
                kinematic_tree=kinematic_tree,
                num_heads=num_heads,
                use_laplacian=re_use_laplacian,
                use_spd=re_use_spd,
                use_ppr=re_use_ppr,
                ppr_alpha=re_ppr_alpha,
                key=key_re,
            )
        else:
            self.relational_embedding = None

        # Hard adjacency mask (kept as a separate, optional locality
        # constraint — orthogonal to the soft RE bias and combinable).
        if use_kinematic_mask and kinematic_tree is not None:
            adj = kinematic_tree.get_adjacency_matrix()
            adj = jnp.asarray(adj)
            mask = (adj + jnp.eye(num_bodies_all)) != 0
            self.adjacency_mask = mask
        else:
            self.adjacency_mask = None

        has_temporal = num_time_tokens > 1
        layer_keys = jax.random.split(key_layers, num_layers)
        if attention_mode == "factorized":
            self.layers = tuple(
                FactorizedAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    has_temporal=has_temporal,
                    key=layer_keys[i],
                )
                for i in range(num_layers)
            )
        else:  # "joint"
            self.layers = tuple(
                JointAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    key=layer_keys[i],
                )
                for i in range(num_layers)
            )

    def __call__(
        self,
        tokens: jax.Array,
        *,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Encode a single (unbatched) token grid.

        Args:
            tokens: ``(T + 1, B_all, D)`` from :class:`SpaceTimeTokenizer`.
            key: JAX PRNG key; used only when dropout > 0.

        Returns:
            ``(T + 1, B_all, D)`` encoded tokens.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        body_pe = self.body_pe_module()  # (B_all, D)
        tokens = tokens + body_pe[None, :, :] + self.time_pe[:, None, :]

        re_bias = (
            self.relational_embedding()
            if self.relational_embedding is not None
            else None
        )

        layer_keys = jax.random.split(key, len(self.layers))
        call = _layer_call_ckpt if self.use_checkpoint else _layer_call
        for layer, k in zip(self.layers, layer_keys):
            tokens = call(layer, tokens, self.adjacency_mask, re_bias, k)
        return tokens
