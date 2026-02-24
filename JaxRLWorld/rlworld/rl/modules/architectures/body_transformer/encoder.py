from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import equinox as eqx
import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = ["BodyTransformerLayer", "BodyTransformerEncoder"]


class BodyTransformerLayer(eqx.Module):
    embed_dim: int = eqx.field(static=True)

    attention: eqx.nn.MultiheadAttention
    ffn_linear1: eqx.nn.Linear
    ffn_linear2: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        *,
        key: jax.Array,
    ):
        self.embed_dim = embed_dim
        keys = jax.random.split(key, 4)

        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embed_dim,
            key_size=embed_dim,
            value_size=embed_dim,
            output_size=embed_dim,
            dropout_p=dropout,
            key=keys[0],
        )
        self.ffn_linear1 = eqx.nn.Linear(embed_dim, dim_feedforward, key=keys[1])
        self.ffn_linear2 = eqx.nn.Linear(dim_feedforward, embed_dim, key=keys[2])
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)

    def __call__(
        self,
        x: jax.Array,
        attn_mask: jax.Array | None = None,
        *,
        key: jax.Array,
    ) -> jax.Array:
        """
        Args:
            x: (seq_len, embed_dim) - unbatched
            attn_mask: (seq_len, seq_len), False = cannot attend (equinox convention)
        """
        key_attn, key_ffn = jax.random.split(key)

        # Self-attention with residual
        attn_out = self.attention(
            query=x,
            key_=x,
            value=x,
            mask=attn_mask,
            inference=False,
            key=key_attn,
        )
        x = jax.vmap(self.norm1)(x + attn_out)

        # FFN with residual
        ffn_out = jax.vmap(self.ffn_linear1)(x)
        ffn_out = jax.nn.elu(ffn_out)
        ffn_out = jax.vmap(self.ffn_linear2)(ffn_out)
        x = jax.vmap(self.norm2)(x + ffn_out)

        return x


class BodyTransformerEncoder(eqx.Module):
    """
    Body Transformer encoder.
    Processes unbatched input. Use jax.vmap for batched input.
    """
    num_bodies: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    use_mixed_attention: bool = eqx.field(static=True)
    first_masked_layer: int = eqx.field(static=True)

    pos_embedding: jax.Array  # (num_bodies, embed_dim)
    layers: tuple
    adjacency: jax.Array

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        embed_dim: int,
        num_heads: int,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        use_mixed_attention: bool = True,
        first_masked_layer: int = 1,
        *,
        key: jax.Array,
    ):
        self.num_bodies = kinematic_tree.num_bodies
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_mixed_attention = use_mixed_attention
        self.first_masked_layer = first_masked_layer

        key_pos, key_layers = jax.random.split(key)

        # Positional embedding as plain array
        self.pos_embedding = jax.random.normal(key_pos, (self.num_bodies, embed_dim)) * 0.02

        # Transformer layers
        layer_keys = jax.random.split(key_layers, num_layers)
        self.layers = tuple([
            BodyTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                key=layer_keys[i],
            )
            for i in range(num_layers)
        ])

        # Adjacency matrix
        adjacency = kinematic_tree.get_adjacency_matrix()
        if isinstance(adjacency, torch.Tensor):
            adjacency = jnp.array(adjacency.detach().cpu().numpy())
        adjacency = adjacency + jnp.eye(self.num_bodies)
        self.adjacency = adjacency.astype(jnp.float32)

    def __call__(
        self,
        tokens: jax.Array,
        *,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """
        Args:
            tokens: (num_bodies, embed_dim) - unbatched

        Returns:
            (num_bodies, embed_dim)
        """
        # True = can attend, False = cannot attend (equinox convention)
        attn_mask = (self.adjacency != 0)

        x = tokens + self.pos_embedding

        if key is not None:
            layer_keys = jax.random.split(key, self.num_layers)
        else:
            layer_keys = [None] * self.num_layers

        for layer_idx, (layer, layer_key) in enumerate(zip(self.layers, layer_keys)):
            if self.use_mixed_attention:
                use_mask = (layer_idx % 2) == (self.first_masked_layer % 2)
            else:
                use_mask = True

            if use_mask:
                x = layer(x, attn_mask=attn_mask, key=layer_key)
            else:
                x = layer(x, attn_mask=None, key=layer_key)

        return x