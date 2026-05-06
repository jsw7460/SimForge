from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = ["GNNEncoder"]


class GNNLayer(eqx.Module):
    """Single GNN layer with mean aggregation + linear transform."""

    linear_self: eqx.nn.Linear
    linear_neighbor: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, hidden_dim: int, dropout: float = 0.0, *, key: jax.Array):
        key1, key2, key3, key_init = jax.random.split(key, 4)

        self.linear_self = eqx.nn.Linear(hidden_dim, hidden_dim, key=key1)
        self.linear_neighbor = eqx.nn.Linear(hidden_dim, hidden_dim, key=key2)
        self.linear_out = eqx.nn.Linear(hidden_dim, hidden_dim, key=key3)
        self.norm = eqx.nn.LayerNorm(hidden_dim)
        self.dropout_rate = dropout

        # Orthogonal initialization
        self.linear_self = self._orthogonal_init(self.linear_self, jnp.sqrt(2.0), key_init)
        key_init, k2, k3 = jax.random.split(key_init, 3)
        self.linear_neighbor = self._orthogonal_init(self.linear_neighbor, jnp.sqrt(2.0), k2)
        self.linear_out = self._orthogonal_init(self.linear_out, jnp.sqrt(2.0), k3)

    @staticmethod
    def _orthogonal_init(linear: eqx.nn.Linear, gain: float, key: jax.Array) -> eqx.nn.Linear:
        weight = linear.weight
        max_dim = max(weight.shape)
        q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(max_dim, max_dim)))
        new_weight = gain * q[: weight.shape[0], : weight.shape[1]]
        new_bias = jnp.zeros_like(linear.bias)
        linear = eqx.tree_at(lambda l: l.weight, linear, new_weight)
        linear = eqx.tree_at(lambda l: l.bias, linear, new_bias)
        return linear

    def __call__(
        self,
        x: jax.Array,
        adjacency: jax.Array,
        *,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """
        Args:
            x: (N, D) node features (unbatched)
            adjacency: (N, N) adjacency matrix

        Returns:
            x_out: (N, D) updated node features
        """
        # Mean aggregation
        degree = adjacency.sum(axis=-1, keepdims=True).clip(min=1)  # (N, 1)
        neighbor_sum = adjacency @ x  # (N, D)
        neighbor_mean = neighbor_sum / degree  # (N, D)

        # Transform and combine
        h_self = jax.vmap(self.linear_self)(x)  # (N, D)
        h_neighbor = jax.vmap(self.linear_neighbor)(neighbor_mean)  # (N, D)
        h = h_self + h_neighbor  # (N, D)

        # Output projection with residual
        h = jax.nn.relu(h)
        h = jax.vmap(self.linear_out)(h)

        if self.dropout_rate > 0 and key is not None:
            h = eqx.nn.Dropout(self.dropout_rate)(h, key=key, inference=False)

        x_out = jax.vmap(self.norm)(x + h)  # Residual connection
        return x_out


class ObsProjection(eqx.Module):
    """Observation projection MLP."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, obs_dim: int, hidden_dim: int, *, key: jax.Array):
        k1, k2, k_init = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(obs_dim, hidden_dim * 2, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_dim * 2, hidden_dim, key=k2)

        # Orthogonal initialization
        k_init1, k_init2 = jax.random.split(k_init)
        self.linear1 = self._orthogonal_init(self.linear1, jnp.sqrt(2.0), k_init1)
        self.linear2 = self._orthogonal_init(self.linear2, 0.01, k_init2)  # gain=0.01

    @staticmethod
    def _orthogonal_init(linear: eqx.nn.Linear, gain: float, key: jax.Array) -> eqx.nn.Linear:
        weight = linear.weight
        max_dim = max(weight.shape)
        q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(max_dim, max_dim)))
        new_weight = gain * q[: weight.shape[0], : weight.shape[1]]
        new_bias = jnp.zeros_like(linear.bias)
        linear = eqx.tree_at(lambda l: l.weight, linear, new_weight)
        linear = eqx.tree_at(lambda l: l.bias, linear, new_bias)
        return linear

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


class GNNEncoder(eqx.Module):
    """
    Bidirectional GNN encoder for kinematic tree.
    Processes unbatched input. Use jax.vmap for batched input.
    """

    obs_projections: dict[int, ObsProjection]
    base_features: jax.Array
    layers: list[GNNLayer]

    adjacency: jax.Array = eqx.field(static=True)
    num_bodies: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)
    leaf_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        *,
        key: jax.Array,
        **kwargs,
    ):
        self.num_bodies = kinematic_tree.num_bodies
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

        # Identify leaf nodes
        self.leaf_indices = tuple(i for i in range(self.num_bodies) if len(kinematic_tree.get_children(i)) == 0)

        key, *proj_keys = jax.random.split(key, len(self.leaf_indices) + 1)

        # Per-body observation projection (leaf only)
        self.obs_projections = {
            idx: ObsProjection(obs_dim, hidden_dim, key=k) for idx, k in zip(self.leaf_indices, proj_keys)
        }

        # Learnable features for all bodies
        key, base_key = jax.random.split(key)
        self.base_features = jax.random.normal(base_key, (self.num_bodies, hidden_dim)) * 0.1

        # GNN layers
        key, *layer_keys = jax.random.split(key, num_layers + 1)
        self.layers = [GNNLayer(hidden_dim, dropout, key=k) for k in layer_keys]

        # Bidirectional adjacency matrix
        self.adjacency = kinematic_tree.get_adjacency_matrix()

    def __call__(self, observation: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        """
        Args:
            observation: (obs_dim,) unbatched

        Returns:
            features: (num_bodies, hidden_dim)
        """
        x = self.base_features

        for idx in self.leaf_indices:
            obs_feat = self.obs_projections[idx](observation)
            x = x.at[idx].add(obs_feat)

        for i, layer in enumerate(self.layers):
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            x = layer(x, self.adjacency, key=subkey)

        return x
