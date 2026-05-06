# layers.py
"""
ABA (Articulated Body Algorithm) inspired layers for kinematic-tree-aware
feature extraction.

Performance notes:
  - All per-body weights are stacked tensors (batched matmul, not tuple-of-modules).
  - Tree traversal is depth-parallel: bodies at the same depth are processed
    simultaneously via batched gather/scatter, reducing sequential steps from
    num_bodies to max_depth (~4-5 for typical quadrupeds/humanoids).
"""

import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────


def _batched_layer_norm(
    x: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """
    LayerNorm over last two dims, batched over leading dim.

    x, scale, bias: (K, C, d)
    """
    mean = x.mean(axis=(-2, -1), keepdims=True)
    var = x.var(axis=(-2, -1), keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def _precompute_depth_levels_bu(kinematic_tree: "KinematicTree"):
    """
    Precompute bottom-up depth-parallel traversal data.

    Returns:
        depth_body_indices: tuple of tuples — body indices per depth (leaves first)
        depth_padded_children: tuple of tuples of tuples — padded children per body per depth
        depth_children_mask: tuple of tuples of tuples — mask for valid children
    """
    depth_groups = kinematic_tree.get_depth_groups()
    max_depth = kinematic_tree.get_max_depth()

    body_indices = []
    padded_children = []
    children_mask = []

    for d in range(max_depth, -1, -1):
        bodies = depth_groups.get(d, [])
        if not bodies:
            continue

        all_ch = [kinematic_tree.get_children(b) for b in bodies]
        max_ch = max((len(c) for c in all_ch), default=0)
        max_ch = max(max_ch, 1)  # at least 1 for array shape

        padded = []
        mask = []
        for ch in all_ch:
            p = list(ch) + [0] * (max_ch - len(ch))
            m = [1.0] * len(ch) + [0.0] * (max_ch - len(ch))
            padded.append(tuple(p))
            mask.append(tuple(m))

        body_indices.append(tuple(bodies))
        padded_children.append(tuple(padded))
        children_mask.append(tuple(mask))

    return tuple(body_indices), tuple(padded_children), tuple(children_mask)


def _precompute_depth_levels_td(kinematic_tree: "KinematicTree"):
    """
    Precompute top-down depth-parallel traversal data.

    Returns:
        depth_body_indices: tuple of tuples — body indices per depth (root first)
        depth_parent_indices: tuple of tuples — parent index per body per depth
    """
    depth_groups = kinematic_tree.get_depth_groups()
    max_depth = kinematic_tree.get_max_depth()

    body_indices = []
    parent_indices = []

    for d in range(0, max_depth + 1):
        bodies = depth_groups.get(d, [])
        if not bodies:
            continue
        parents = tuple(kinematic_tree.parent_indices[b] for b in bodies)
        body_indices.append(tuple(bodies))
        parent_indices.append(parents)

    return tuple(body_indices), tuple(parent_indices)


# ─────────────────────────────────────────────────────────────────
#  Bottom-Up Layer
# ─────────────────────────────────────────────────────────────────


class PerBodyABABottomUpLayer(eqx.Module):
    """
    ABA Bottom-Up Pass with depth-parallel tree traversal.

    Instead of iterating over bodies one-by-one (14 sequential steps),
    processes all bodies at the same tree depth simultaneously
    (~4-5 sequential steps for typical robots).
    """

    # Batched obs projection
    obs_W1: jax.Array  # (N, hidden_dim, obs_dim)
    obs_b1: jax.Array  # (N, hidden_dim)
    obs_W2: jax.Array  # (N, feature_dim, hidden_dim)
    obs_b2: jax.Array  # (N, feature_dim)

    # Learnable base features
    link_base: jax.Array  # (N, C, d)

    # Batched LayerNorm params
    ln_scale: jax.Array  # (N, C, d)
    ln_bias: jax.Array  # (N, C, d)

    # Motion basis
    motion_basis: jax.Array  # (N, C, d, d)

    # Optional learnable contribution weight (per body)
    contribution_weight: jax.Array | None  # (N,)

    # Optional global norm
    global_norm: eqx.nn.LayerNorm | None

    # Identity matrix for orthogonality loss
    _identity: jax.Array

    # Static configuration
    num_bodies: int = eqx.field(static=True)
    link_channels: int = eqx.field(static=True)
    spatial_dim: int = eqx.field(static=True)
    use_positive_constraint: bool = eqx.field(static=True)
    learnable_contribution_weight: bool = eqx.field(static=True)
    use_global_layer_norm: bool = eqx.field(static=True)

    # Depth-parallel traversal data (static)
    depth_body_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    depth_padded_children: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    depth_children_mask: tuple[tuple[tuple[float, ...], ...], ...] = eqx.field(static=True)

    # Keep for backward compat (used by encoder.py etc.)
    traversal_order: tuple[int, ...] = eqx.field(static=True)
    children_map: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        link_channels: int = 8,
        spatial_dim: int = 6,
        learnable_contribution_weight: bool = False,
        use_global_layer_norm: bool = False,
        use_positive_constraint: bool = True,
        *,
        key: jax.Array,
    ):
        self.num_bodies = kinematic_tree.num_bodies
        self.link_channels = link_channels
        self.spatial_dim = spatial_dim
        self.use_positive_constraint = use_positive_constraint
        self.learnable_contribution_weight = learnable_contribution_weight
        self.use_global_layer_norm = use_global_layer_norm

        # Legacy traversal data (kept for compat)
        self.traversal_order = tuple(kinematic_tree.traverse_bottom_up())
        self.children_map = tuple(tuple(kinematic_tree.get_children(i)) for i in range(self.num_bodies))

        # Depth-parallel traversal data
        (
            self.depth_body_indices,
            self.depth_padded_children,
            self.depth_children_mask,
        ) = _precompute_depth_levels_bu(kinematic_tree)

        N = self.num_bodies
        feature_dim = link_channels * spatial_dim
        hidden_dim = feature_dim * 2

        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Batched obs projection
        scale1 = 1.0 / jnp.sqrt(obs_dim)
        scale2 = 1.0 / jnp.sqrt(hidden_dim)
        self.obs_W1 = jax.random.normal(k1, (N, hidden_dim, obs_dim)) * scale1
        self.obs_b1 = jnp.zeros((N, hidden_dim))
        self.obs_W2 = jax.random.normal(k2, (N, feature_dim, hidden_dim)) * scale2
        self.obs_b2 = jnp.zeros((N, feature_dim))

        self.link_base = jax.random.normal(k3, (N, link_channels, spatial_dim)) * 0.1

        self.ln_scale = jnp.ones((N, link_channels, spatial_dim))
        self.ln_bias = jnp.zeros((N, link_channels, spatial_dim))

        self.motion_basis = self._init_orthogonal_basis(k4)

        if learnable_contribution_weight:
            init_logit = math.log(0.5 / 0.5)
            self.contribution_weight = jnp.full((N,), init_logit)
        else:
            self.contribution_weight = None

        if use_global_layer_norm:
            self.global_norm = eqx.nn.LayerNorm((feature_dim,))
        else:
            self.global_norm = None

        self._identity = jnp.eye(spatial_dim)

    def _init_orthogonal_basis(self, key: jax.Array) -> jax.Array:
        shape = (self.num_bodies, self.link_channels, self.spatial_dim, self.spatial_dim)
        W = jax.random.normal(key, shape)

        def orthogonalize(w):
            q, _ = jnp.linalg.qr(w)
            return q

        return jax.vmap(jax.vmap(orthogonalize))(W)

    def __call__(self, observations: jax.Array) -> jax.Array:
        """
        Args:
            observations: (obs_dim,) unbatched
        Returns:
            link_features: (num_bodies, link_channels, spatial_dim)
        """
        C, d = self.link_channels, self.spatial_dim

        # 1. Batched obs projection — all bodies in one einsum
        h = jnp.einsum("nhi,i->nh", self.obs_W1, observations) + self.obs_b1
        h = jax.nn.relu(h)
        obs_feat = jnp.einsum("nfh,nh->nf", self.obs_W2, h) + self.obs_b2
        obs_feat = obs_feat.reshape(self.num_bodies, C, d)

        # 2. Base + obs features
        if self.use_positive_constraint:
            base = jax.nn.softplus(self.link_base) + 1e-6
        else:
            base = self.link_base
        init_features = base + obs_feat  # (N, C, d)

        # 3. Depth-parallel bottom-up traversal
        body_features = jnp.zeros((self.num_bodies, C, d))

        for level in range(len(self.depth_body_indices)):
            bidxs = self.depth_body_indices[level]  # tuple of ints
            K = len(bidxs)
            bidxs_arr = jnp.array(bidxs)  # (K,) — constant in XLA

            # Init features for this depth
            feats = init_features[bidxs_arr]  # (K, C, d)

            # Children indices and mask
            ch_idx = jnp.array(self.depth_padded_children[level])  # (K, max_ch)
            ch_mask = jnp.array(self.depth_children_mask[level])  # (K, max_ch)

            # Gather children features and motion basis
            child_F = body_features[ch_idx]  # (K, max_ch, C, d)
            child_W = self.motion_basis[ch_idx]  # (K, max_ch, C, d, d)

            # Batched contribution: F - diag(F W W^T F)
            FW = child_F[..., :, None] * child_W  # (K, mc, C, d, d)
            FWWt = FW @ child_W.transpose(0, 1, 2, 4, 3)  # (K, mc, C, d, d)
            proj_diag = jnp.diagonal(FWWt * child_F[..., None, :], axis1=-2, axis2=-1)  # (K, mc, C, d)
            contribs = child_F - proj_diag  # (K, mc, C, d)

            # Mask invalid children and sum
            contribs = contribs * ch_mask[:, :, None, None]
            child_sum = contribs.sum(axis=1)  # (K, C, d)

            # Contribution weight
            if self.learnable_contribution_weight:
                weight = jax.nn.sigmoid(self.contribution_weight[bidxs_arr])
                feats = feats + weight[:, None, None] * child_sum
            else:
                feats = feats + 0.1 * child_sum

            # Batched LayerNorm
            feats = _batched_layer_norm(
                feats,
                self.ln_scale[bidxs_arr],
                self.ln_bias[bidxs_arr],
            )

            # Scatter — one per depth level instead of one per body
            body_features = body_features.at[bidxs_arr].set(feats)

        # 4. Optional global normalization
        if self.global_norm is not None:
            shape = body_features.shape
            body_features = self.global_norm(body_features.reshape(self.num_bodies, -1))
            body_features = body_features.reshape(shape)

        return body_features

    def compute_orthogonality_loss(
        self,
        observations: jax.Array,
        link_features: jax.Array | None = None,
    ) -> jax.Array:
        """
        Compute orthogonality regularization (vectorized).

        Args:
            observations: (obs_dim,) — used only if link_features not provided
            link_features: (N, C, d) — pass precomputed to avoid double forward
        """
        if link_features is None:
            link_features = self(observations)

        F = link_features  # (N, C, d)
        W = self.motion_basis  # (N, C, d, d)
        FW = F[:, :, :, None] * W  # (N, C, d, d)
        gram = W.transpose(0, 1, 3, 2) @ FW  # (N, C, d, d)

        target = jax.lax.stop_gradient(self._identity)
        return ((gram - target) ** 2).mean()

    @property
    def output_dim(self) -> tuple[int, int]:
        return (self.num_bodies, self.link_channels * self.spatial_dim)


# ─────────────────────────────────────────────────────────────────
#  Top-Down Layer
# ─────────────────────────────────────────────────────────────────


class PerBodyABATopDownLayer(eqx.Module):
    """
    ABA Top-Down Pass with depth-parallel traversal.

    Propagates global context root → leaves. Bodies at the same depth
    are processed simultaneously.
    """

    # Batched parent→child projection
    proj_W: jax.Array  # (N, feature_dim, feature_dim)
    proj_b: jax.Array  # (N, feature_dim)

    # Batched LayerNorm params
    ln_scale: jax.Array  # (N, C, d)
    ln_bias: jax.Array  # (N, C, d)

    # Per-body learnable gate
    gate_bias: jax.Array  # (N,)

    # Static
    num_bodies: int = eqx.field(static=True)
    link_channels: int = eqx.field(static=True)
    spatial_dim: int = eqx.field(static=True)

    # Depth-parallel traversal data (static)
    td_body_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    td_parent_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        link_channels: int = 8,
        spatial_dim: int = 6,
        *,
        key: jax.Array,
    ):
        self.num_bodies = kinematic_tree.num_bodies
        self.link_channels = link_channels
        self.spatial_dim = spatial_dim

        self.td_body_indices, self.td_parent_indices = _precompute_depth_levels_td(kinematic_tree)

        N = self.num_bodies
        feature_dim = link_channels * spatial_dim

        scale = 0.1 / jnp.sqrt(feature_dim)
        self.proj_W = jax.random.normal(key, (N, feature_dim, feature_dim)) * scale
        self.proj_b = jnp.zeros((N, feature_dim))

        self.ln_scale = jnp.ones((N, link_channels, spatial_dim))
        self.ln_bias = jnp.zeros((N, link_channels, spatial_dim))

        self.gate_bias = jnp.full((N,), -2.0)

    def __call__(self, bottom_up_features: jax.Array) -> jax.Array:
        """
        Args:
            bottom_up_features: (num_bodies, link_channels, spatial_dim)
        Returns:
            top_down_features: (num_bodies, link_channels, spatial_dim)
        """
        C, d = self.link_channels, self.spatial_dim
        td = jnp.zeros_like(bottom_up_features)

        for level in range(len(self.td_body_indices)):
            bidxs = self.td_body_indices[level]
            pidxs = self.td_parent_indices[level]
            K = len(bidxs)
            bidxs_arr = jnp.array(bidxs)

            bu = bottom_up_features[bidxs_arr]  # (K, C, d)

            if pidxs[0] == -1:
                # Root level: pass through bottom-up
                feats = bu
            else:
                # Gather parent top-down features
                pidxs_arr = jnp.array(pidxs)
                parent_flat = td[pidxs_arr].reshape(K, -1)  # (K, C*d)

                # Batched projection
                W = self.proj_W[bidxs_arr]  # (K, F, F)
                b = self.proj_b[bidxs_arr]  # (K, F)
                message = jnp.einsum("kij,kj->ki", W, parent_flat) + b
                message = message.reshape(K, C, d)

                gate = jax.nn.sigmoid(self.gate_bias[bidxs_arr])  # (K,)
                feats = bu + gate[:, None, None] * message

            feats = _batched_layer_norm(
                feats,
                self.ln_scale[bidxs_arr],
                self.ln_bias[bidxs_arr],
            )
            td = td.at[bidxs_arr].set(feats)

        return td
