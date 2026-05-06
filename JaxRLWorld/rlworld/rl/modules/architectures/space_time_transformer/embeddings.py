"""Positional and relational embeddings for the SpaceTimeTransformer.

Two flavors of positional embedding for the body axis:

* :class:`LearnedPositionalEmbedding` — a single learnable ``(N, D)`` table,
  one row per body. No structural prior; matches the previous
  ``body_pe`` array semantics.
* :class:`TraversalPositionalEmbedding` — SWAT-style. Builds three
  learnable tables indexed by pre-order, in-order, and post-order DFS
  traversals of the kinematic tree, then concatenates the three lookups.
  Bodies that are nearby in the tree end up at nearby traversal indices
  in at least one of the orderings, giving the model an inductive bias
  for tree locality (vs. arbitrary index → arbitrary embedding).

And a relational embedding for the body × body attention bias:

* :class:`GraphRelationalEmbedding` — SWAT-style. Computes Laplacian,
  shortest-path-distance, and personalized-PageRank matrices once at
  init, then projects them to ``(num_heads, N, N)`` via a learnable
  ``Linear``. The result is added to spatial-attention scores so the
  attention has a continuous, head-specific tree-locality prior (vs.
  the old hard 0/1 adjacency mask which only allowed direct neighbors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


__all__ = [
    "LearnedPositionalEmbedding",
    "TraversalPositionalEmbedding",
    "GraphRelationalEmbedding",
]


# ============================================================
# Positional embeddings
# ============================================================


class LearnedPositionalEmbedding(eqx.Module):
    """Single learnable ``(num, embed_dim)`` table.

    Drop-in replacement for the previous raw ``body_pe`` field. Has no
    structural prior — each row is independent.
    """

    embedding: jax.Array
    num: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    def __init__(self, num: int, embed_dim: int, *, key: jax.Array):
        self.num = num
        self.embed_dim = embed_dim
        # Match the previous body_pe init scale (0.02 std).
        self.embedding = jax.random.normal(key, (num, embed_dim)) * 0.02

    def __call__(self) -> jax.Array:
        """Return ``(num, embed_dim)``."""
        return self.embedding


class TraversalPositionalEmbedding(eqx.Module):
    """SWAT-style traversal-based positional embedding.

    For each body, looks up three learnable embeddings indexed by its
    pre-order / in-order / post-order DFS position in the kinematic
    tree, then concatenates them. If ``embed_dim`` isn't divisible by
    three, an extra ``Linear`` re-projects from ``3 * (embed_dim // 3)``
    back to ``embed_dim`` so the output shape always matches the
    requested ``embed_dim``.
    """

    pre_embedding: jax.Array
    in_embedding: jax.Array
    post_embedding: jax.Array
    pre_order: jax.Array
    in_order: jax.Array
    post_order: jax.Array
    extra_proj: eqx.nn.Linear | None

    num_bodies: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    embed_dim_per_traversal: int = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: KinematicTree,
        embed_dim: int,
        *,
        key: jax.Array,
    ):
        self.num_bodies = kinematic_tree.num_bodies
        self.embed_dim = embed_dim
        self.embed_dim_per_traversal = embed_dim // 3

        pre_idx, in_idx, post_idx = self._compute_traversals(kinematic_tree)
        self.pre_order = jnp.asarray(pre_idx, dtype=jnp.int32)
        self.in_order = jnp.asarray(in_idx, dtype=jnp.int32)
        self.post_order = jnp.asarray(post_idx, dtype=jnp.int32)

        k_pre, k_in, k_post, k_proj = jax.random.split(key, 4)
        d = self.embed_dim_per_traversal
        self.pre_embedding = jax.random.normal(k_pre, (self.num_bodies, d)) * 0.02
        self.in_embedding = jax.random.normal(k_in, (self.num_bodies, d)) * 0.02
        self.post_embedding = jax.random.normal(k_post, (self.num_bodies, d)) * 0.02

        remaining = embed_dim - 3 * d
        if remaining > 0:
            self.extra_proj = eqx.nn.Linear(3 * d, embed_dim, key=k_proj)
        else:
            self.extra_proj = None

    @staticmethod
    def _compute_traversals(
        tree: KinematicTree,
    ) -> tuple[list[int], list[int], list[int]]:
        """Compute pre/in/post-order DFS indices for each node.

        Returns three length-``num_bodies`` lists; ``pre_indices[i]`` is
        body ``i``'s position in pre-order traversal.
        """
        num_bodies = tree.num_bodies
        root_idx = tree.root_idx

        pre_indices = [0] * num_bodies
        in_indices = [0] * num_bodies
        post_indices = [0] * num_bodies
        pre_counter = [0]
        in_counter = [0]
        post_counter = [0]

        def traverse(node: int) -> None:
            children = tree.get_children(node)

            # Pre-order: visit node first
            pre_indices[node] = pre_counter[0]
            pre_counter[0] += 1

            if len(children) == 0:
                in_indices[node] = in_counter[0]
                in_counter[0] += 1
            elif len(children) == 1:
                traverse(children[0])
                in_indices[node] = in_counter[0]
                in_counter[0] += 1
            else:
                mid = len(children) // 2
                for child in children[:mid]:
                    traverse(child)
                in_indices[node] = in_counter[0]
                in_counter[0] += 1
                for child in children[mid:]:
                    traverse(child)

            # Post-order: visit node last
            post_indices[node] = post_counter[0]
            post_counter[0] += 1

        traverse(root_idx)
        return pre_indices, in_indices, post_indices

    def __call__(self) -> jax.Array:
        """Return ``(num_bodies, embed_dim)``."""
        pre = self.pre_embedding[self.pre_order]  # (N, D//3)
        ino = self.in_embedding[self.in_order]  # (N, D//3)
        post = self.post_embedding[self.post_order]  # (N, D//3)
        combined = jnp.concatenate([pre, ino, post], axis=-1)  # (N, 3*(D//3))
        if self.extra_proj is not None:
            # Apply per-row Linear via vmap.
            combined = jax.vmap(self.extra_proj)(combined)  # (N, embed_dim)
        return combined


# ============================================================
# Relational embedding (attention bias)
# ============================================================


class GraphRelationalEmbedding(eqx.Module):
    """SWAT-style graph relational embedding for attention bias.

    Computes static graph features (Laplacian, shortest-path distance,
    personalized PageRank) once at init, then projects them per-head
    with a learnable ``Linear`` whose output is added to spatial
    attention scores. The features themselves are non-trainable buffers
    (registered as inexact arrays but they hold integer-derived values
    that should not be optimized).
    """

    # Stacked features of shape (N, N, num_features). Frozen at init.
    features: jax.Array
    projection: eqx.nn.Linear
    num_bodies: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    feature_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: KinematicTree,
        num_heads: int = 1,
        use_laplacian: bool = True,
        use_spd: bool = True,
        use_ppr: bool = True,
        ppr_alpha: float = 0.15,
        *,
        key: jax.Array,
    ):
        self.num_bodies = kinematic_tree.num_bodies
        self.num_heads = num_heads

        adj_np = np.asarray(kinematic_tree.get_adjacency_matrix(), dtype=np.float32)

        feats: list[np.ndarray] = []
        names: list[str] = []
        if use_laplacian:
            feats.append(self._compute_normalized_laplacian(adj_np))
            names.append("laplacian")
        if use_spd:
            feats.append(self._compute_shortest_path_distance(adj_np))
            names.append("spd")
        if use_ppr:
            feats.append(self._compute_ppr(adj_np, ppr_alpha))
            names.append("ppr")

        if not feats:
            raise ValueError(
                "GraphRelationalEmbedding requires at least one of use_laplacian / use_spd / use_ppr to be True."
            )

        self.feature_names = tuple(names)
        # (N, N, num_features), frozen.
        self.features = jnp.asarray(np.stack(feats, axis=-1))
        self.projection = eqx.nn.Linear(len(feats), num_heads, key=key)

    @staticmethod
    def _compute_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
        """``L = I - D^{-1/2} A D^{-1/2}``. Returns ``(N, N)``."""
        n = adj.shape[0]
        degree = adj.sum(axis=1)
        d_inv_sqrt = np.where(degree > 0, np.power(degree, -0.5), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return np.eye(n, dtype=np.float32) - D_inv_sqrt @ adj @ D_inv_sqrt

    @staticmethod
    def _compute_shortest_path_distance(adj: np.ndarray) -> np.ndarray:
        """BFS from every node. Disconnected pairs get ``inf``; we
        normalize by ``N`` to keep the magnitudes bounded.
        """
        n = adj.shape[0]
        spd = np.zeros((n, n), dtype=np.float32)
        for src in range(n):
            dist = np.full(n, np.inf, dtype=np.float32)
            dist[src] = 0.0
            queue = [src]
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                for nb in range(n):
                    if adj[cur, nb] > 0 and not np.isfinite(dist[nb]):
                        dist[nb] = dist[cur] + 1.0
                        queue.append(nb)
            spd[src] = dist
        # Replace inf (disconnected) with a large finite value so the
        # learnable projection can still distinguish "very far" from
        # "moderately far". n is a safe ceiling.
        spd = np.where(np.isfinite(spd), spd, float(n))
        return spd / n  # normalize to [0, 1]

    @staticmethod
    def _compute_ppr(adj: np.ndarray, alpha: float) -> np.ndarray:
        """Personalized PageRank: ``alpha * (I - (1-alpha) * P^T)^{-1}``.

        ``P`` is the row-stochastic transition matrix.
        """
        n = adj.shape[0]
        degree = adj.sum(axis=1, keepdims=True)
        degree = np.where(degree > 0, degree, 1.0)
        P = adj / degree
        I = np.eye(n, dtype=np.float32)
        return alpha * np.linalg.inv(I - (1.0 - alpha) * P.T)

    def __call__(self) -> jax.Array:
        """Return ``(num_heads, N, N)`` attention bias."""
        # features: (N, N, F) → projection over last dim → (N, N, H)
        bias = jax.vmap(jax.vmap(self.projection))(self.features)
        # Move heads to leading axis: (H, N, N)
        return jnp.transpose(bias, (2, 0, 1))
