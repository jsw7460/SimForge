"""Factorized (space × time) transformer encoder.

TimeSformer / ViViT-style encoder that alternates attention between the
body axis (spatial) and the time axis (temporal), given a token grid of
shape ``(T + 1, B_all, D)`` produced by :class:`SpaceTimeTokenizer`.

Spatial attention optionally uses a kinematic adjacency mask (only
connected bodies attend to each other, Body-Transformer style) — this is
opt-in so non-morphology tasks can fall back to full spatial attention.

Temporal attention is currently unmasked (full attention over time) and
is a no-op when the grid has only one time token, which is the
degenerate case covering non-tracking presets.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


__all__ = ["FactorizedAttentionBlock", "SpaceTimeTransformerEncoder"]


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
        *,
        key: jax.Array,
    ) -> jax.Array:
        """Apply one factorized block.

        Args:
            tokens: ``(T, B, D)`` token grid (unbatched).
            spatial_mask: ``(B, B)`` boolean mask; True = can attend. None = full.
            key: JAX PRNG key.

        Returns:
            ``(T, B, D)`` updated tokens.
        """
        T = tokens.shape[0]
        key_s, key_t, key_ffn = jax.random.split(key, 3)

        spatial_keys = jax.random.split(key_s, T)

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


def _layer_call(
    layer: FactorizedAttentionBlock,
    tokens: jax.Array,
    spatial_mask: jax.Array,
    key: jax.Array,
) -> jax.Array:
    return layer(tokens, spatial_mask=spatial_mask, key=key)


# Gradient-checkpointed wrapper: forward stores only inputs, backward
# recomputes the layer's intermediate activations. Trades ~1.5x extra
# forward compute for ~50% lower activation memory, which is what lets
# us keep the PPO mini-batch large without OOM-ing the attention scores
# tensor at high num_envs.
_layer_call_ckpt = eqx.filter_checkpoint(_layer_call)


class SpaceTimeTransformerEncoder(eqx.Module):
    body_pe: jax.Array
    time_pe: jax.Array
    layers: tuple
    adjacency_mask: jax.Array

    num_bodies_all: int = eqx.field(static=True)
    num_time_tokens: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    use_kinematic_mask: bool = eqx.field(static=True)
    use_checkpoint: bool = eqx.field(static=True)

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
        *,
        key: jax.Array,
    ):
        self.num_bodies_all = num_bodies_all
        self.num_time_tokens = num_time_tokens
        self.embed_dim = embed_dim
        self.use_kinematic_mask = use_kinematic_mask
        self.use_checkpoint = use_checkpoint

        key_pe, key_layers = jax.random.split(key)
        key_bpe, key_tpe = jax.random.split(key_pe)
        self.body_pe = (
            jax.random.normal(key_bpe, (num_bodies_all, embed_dim)) * 0.02
        )
        self.time_pe = (
            jax.random.normal(key_tpe, (num_time_tokens, embed_dim)) * 0.02
        )

        if use_kinematic_mask and kinematic_tree is not None:
            adj = kinematic_tree.get_adjacency_matrix()
            # KinematicTree returns a jnp.ndarray; if some future backend
            # returns a numpy/torch-like object jnp.array will convert it.
            adj = jnp.asarray(adj)
            mask = (adj + jnp.eye(num_bodies_all)) != 0
        else:
            mask = jnp.ones((num_bodies_all, num_bodies_all), dtype=jnp.bool_)
        self.adjacency_mask = mask

        has_temporal = num_time_tokens > 1
        layer_keys = jax.random.split(key_layers, num_layers)
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

    def __call__(
        self,
        tokens: jax.Array,
        *,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Encode a single (unbatched) token grid.

        Args:
            tokens: ``(T + 1, B_all, D)`` from :class:`SpaceTimeTokenizer`.
            key: JAX PRNG key; used only if dropout > 0, but always required
                because ``eqx.nn.MultiheadAttention`` takes one unconditionally.
                When None, a fixed ``PRNGKey(0)`` is used — safe for dropout=0.

        Returns:
            ``(T + 1, B_all, D)`` encoded tokens.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        tokens = tokens + self.body_pe[None, :, :] + self.time_pe[:, None, :]

        layer_keys = jax.random.split(key, len(self.layers))
        call = _layer_call_ckpt if self.use_checkpoint else _layer_call
        for layer, k in zip(self.layers, layer_keys):
            tokens = call(layer, tokens, self.adjacency_mask, k)
        return tokens
