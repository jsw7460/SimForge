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

All self-attention (spatial, temporal, joint) routes through
``jax.nn.dot_product_attention`` via the local :func:`_flash_self_attention`
helper. With ``implementation="cudnn"`` (default) this fires cuDNN's
fused FlashAttention kernel (the torch SDPA equivalent), reusing the
``eqx.nn.MultiheadAttention`` modules' projection weights — so the
parameterization is unchanged, only the attention-compute path is
faster. cuDNN flash requires bf16 / fp16 Q/K/V, so the helper casts
those tensors at the attention boundary and casts the output back to
the input dtype; weights and every other tensor stay fp32 (mixed
precision pattern, no master-weights wrapper needed). ``re_bias``
(from :class:`GraphRelationalEmbedding`) and the kinematic ``bool_mask``
are passed through to the fused kernel directly, so the previous
"manual softmax when bias is set" branch is gone.
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


def _flash_self_attention(
    mha: eqx.nn.MultiheadAttention,
    tokens: jax.Array,
    re_bias: jax.Array | None = None,
    bool_mask: jax.Array | None = None,
    implementation: str = "cudnn",
) -> jax.Array:
    """Self-attention via ``jax.nn.dot_product_attention``.

    With ``implementation="cudnn"`` this routes through cuDNN's fused
    FlashAttention kernel (the torch SDPA equivalent). cuDNN flash
    requires bf16/fp16 Q/K/V, so the helper casts inputs (and the bias
    if any) to bfloat16 at the attention boundary and casts the output
    back to the caller's dtype. The mha module's projection weights and
    every other tensor in the model stay fp32 — standard
    mixed-precision pattern, no master-weights wrapper needed.

    With ``implementation="xla"`` the attention runs in the input dtype
    (fp32 in our setup) through the standard XLA matmul + softmax path.
    No fusion, no dtype restriction — safe fallback when cuDNN's flash
    can't accept the shape on a given GPU.

    Reuses ``mha``'s Q/K/V/output projection weights — drop-in for the
    eqx ``mha(query=t, key_=t, value=t, mask=..., inference=...)`` call.

    Args:
        mha: parameter-holding attention module.
        tokens: ``(S, D)`` token grid (unbatched).
        re_bias: ``(H, S, S)`` additive attention bias, or ``None``.
        bool_mask: ``(S, S)`` boolean mask, True = can attend, or
            ``None`` for full attention.
        implementation: ``"cudnn"`` (default, flash) or ``"xla"``.

    Returns:
        ``(S, D)``.
    """
    H = mha.num_heads
    qk = mha.qk_size
    S, _ = tokens.shape
    out_dtype = tokens.dtype

    # Project per-row using the mha module's existing weights.
    q = jax.vmap(mha.query_proj)(tokens).reshape(S, H, qk)
    k = jax.vmap(mha.key_proj)(tokens).reshape(S, H, qk)
    v = jax.vmap(mha.value_proj)(tokens).reshape(S, H, qk)

    if implementation == "cudnn":
        compute_dtype = jnp.bfloat16
        q = q.astype(compute_dtype)
        k = k.astype(compute_dtype)
        v = v.astype(compute_dtype)
        if re_bias is not None:
            re_bias = re_bias.astype(compute_dtype)

    # jax.nn.dot_product_attention expects (B, T, N, H_per_head); add the
    # batch axis. Bias/mask shapes follow ``(B, N, T, S)`` broadcastable;
    # ``re_bias[None]`` → ``(1, H, S, S)``, ``bool_mask[None, None]`` →
    # ``(1, 1, S, S)`` (broadcast over heads).
    bias = re_bias[None] if re_bias is not None else None
    mask = bool_mask[None, None] if bool_mask is not None else None

    out = jax.nn.dot_product_attention(
        q[None],
        k[None],
        v[None],
        bias=bias,
        mask=mask,
        implementation=implementation,
    )  # (1, S, H, qk)

    attn = out[0].astype(out_dtype).reshape(S, H * qk)
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
        del key  # no dropout anywhere in this codebase; cuDNN flash has no dropout arg

        # Spatial attention via cuDNN FlashAttention (drop-in for the
        # previous eqx-MHA / manual-bias split). The flash helper
        # supports both ``re_bias`` and ``bool_mask`` natively, so the
        # two-branch logic collapses to one call.
        def spatial_step(t_tokens: jax.Array) -> jax.Array:
            a = _flash_self_attention(
                self.spatial_attn,
                t_tokens,
                re_bias=spatial_re_bias,
                bool_mask=spatial_mask,
            )
            return jax.vmap(self.spatial_norm)(t_tokens + a)

        tokens = jax.vmap(spatial_step)(tokens)

        if self.has_temporal and T > 1:
            tokens_bt = jnp.transpose(tokens, (1, 0, 2))

            def temporal_step(b_tokens: jax.Array) -> jax.Array:
                a = _flash_self_attention(self.temporal_attn, b_tokens)
                return jax.vmap(self.temporal_norm)(b_tokens + a)

            tokens_bt = jax.vmap(temporal_step)(tokens_bt)
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
        del key  # no dropout; cuDNN flash has no dropout arg

        joint_mask = jnp.tile(body_mask, (T, T)) if body_mask is not None else None
        joint_re_bias = jnp.tile(body_re_bias, (1, T, T)) if body_re_bias is not None else None

        attn_out = _flash_self_attention(
            self.attn,
            flat,
            re_bias=joint_re_bias,
            bool_mask=joint_mask,
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
        kinematic_tree: KinematicTree | None = None,
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
            raise ValueError(f"Unknown attention_mode={attention_mode!r}. Expected 'factorized' or 'joint'.")
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
                num=num_bodies_all,
                embed_dim=embed_dim,
                key=key_pe,
            )
        elif pe_type == "traversal":
            if kinematic_tree is None:
                raise ValueError(
                    "pe_type='traversal' requires kinematic_tree to compute pre/in/post-order DFS indices."
                )
            self.body_pe_module = TraversalPositionalEmbedding(
                kinematic_tree=kinematic_tree,
                embed_dim=embed_dim,
                key=key_pe,
            )
        else:
            raise ValueError(f"Unknown pe_type={pe_type!r}. Expected 'learned' or 'traversal'.")

        # Time positional embedding stays a simple learned (T, D) table —
        # the time axis has no graph structure to encode.
        self.time_pe = jax.random.normal(key_tpe, (num_time_tokens, embed_dim)) * 0.02

        # Spatial relational bias (optional, additive to attn scores).
        if use_relational_bias:
            if kinematic_tree is None:
                raise ValueError("use_relational_bias=True requires kinematic_tree.")
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

        re_bias = self.relational_embedding() if self.relational_embedding is not None else None

        layer_keys = jax.random.split(key, len(self.layers))
        call = _layer_call_ckpt if self.use_checkpoint else _layer_call
        for layer, k in zip(self.layers, layer_keys):
            tokens = call(layer, tokens, self.adjacency_mask, re_bias, k)
        return tokens
