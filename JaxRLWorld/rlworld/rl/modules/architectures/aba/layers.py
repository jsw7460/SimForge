# layers.py
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import math

from rlworld.rl.modules.utils import MLP

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


class PerBodyABABottomUpLayer(eqx.Module):
    """
    ABA Bottom-Up Pass with per-body observation projections.
    """
    # Network parameters
    obs_projections: tuple[MLP, ...]
    link_base: jax.Array
    link_norms: tuple[eqx.nn.LayerNorm, ...]
    motion_basis: jax.Array

    # Optional learnable contribution weight
    contribution_weight: jax.Array | None

    # Static configuration
    num_bodies: int = eqx.field(static=True)
    link_channels: int = eqx.field(static=True)
    spatial_dim: int = eqx.field(static=True)
    use_positive_constraint: bool = eqx.field(static=True)
    learnable_contribution_weight: bool = eqx.field(static=True)
    use_global_layer_norm: bool = eqx.field(static=True)

    # Tree structure (static)
    traversal_order: tuple[int, ...] = eqx.field(static=True)
    children_map: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    # Optional global norm
    global_norm: eqx.nn.LayerNorm | None

    # Identity matrix for orthogonality loss
    _identity: jax.Array

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

        # Store tree structure as static tuples
        self.traversal_order = tuple(kinematic_tree.traverse_bottom_up())
        self.children_map = tuple(
            tuple(kinematic_tree.get_children(i))
            for i in range(self.num_bodies)
        )

        feature_dim = link_channels * spatial_dim
        hidden_dim = feature_dim * 2

        # Split keys
        keys = jax.random.split(key, self.num_bodies + 2)
        proj_keys = keys[:self.num_bodies]
        base_key = keys[self.num_bodies]
        motion_key = keys[self.num_bodies + 1]

        # Per-body observation projections using MLP
        self.obs_projections = tuple(
            MLP(
                input_dim=obs_dim,
                hidden_dims=[hidden_dim],
                output_dim=feature_dim,
                activation="relu",
                output_activation=None,
                use_layer_norm=False,
                key=proj_keys[i],
            )
            for i in range(self.num_bodies)
        )

        # Learnable base features
        self.link_base = jax.random.normal(base_key, (self.num_bodies, link_channels, spatial_dim)) * 0.1

        # Per-body LayerNorm
        self.link_norms = tuple(
            eqx.nn.LayerNorm((link_channels, spatial_dim))
            for _ in range(self.num_bodies)
        )

        # Motion basis W per body
        self.motion_basis = self._init_orthogonal_basis(motion_key)

        # Contribution weight
        if learnable_contribution_weight:
            init_scale = 0.5
            init_logit = math.log(init_scale / (1 - init_scale))
            self.contribution_weight = jnp.full((len(self.traversal_order),), init_logit)
        else:
            self.contribution_weight = None

        # Global LayerNorm
        if use_global_layer_norm:
            self.global_norm = eqx.nn.LayerNorm((feature_dim,))
        else:
            self.global_norm = None

        # Identity for orthogonality loss
        self._identity = jnp.eye(spatial_dim)

    def _init_orthogonal_basis(self, key: jax.Array) -> jax.Array:
        """Initialize motion basis as orthonormal matrices."""
        shape = (self.num_bodies, self.link_channels, self.spatial_dim, self.spatial_dim)
        W = jax.random.normal(key, shape)

        def orthogonalize(w):
            q, _ = jnp.linalg.qr(w)
            return q

        W = jax.vmap(jax.vmap(orthogonalize))(W)
        return W

    def __call__(self, observations: jax.Array) -> jax.Array:
        """
        Args:
            observations: (obs_dim,) unbatched

        Returns:
            link_features: (num_bodies, link_channels, spatial_dim)
        """
        # Per-body observation encoding
        obs_features_list = []
        for body_idx in range(self.num_bodies):
            obs_feat = self.obs_projections[body_idx](observations)  # MLP handles unbatched
            obs_feat = obs_feat.reshape(self.link_channels, self.spatial_dim)
            obs_features_list.append(obs_feat)

        obs_features_per_body = jnp.stack(obs_features_list, axis=0)

        # Bottom-up pass
        body_features = {}

        for idx, body_idx in enumerate(self.traversal_order):
            children = self.children_map[body_idx]

            # Base feature
            if self.use_positive_constraint:
                base_feature = jax.nn.softplus(self.link_base[body_idx]) + 1e-6
            else:
                base_feature = self.link_base[body_idx]

            # Add observation features
            base_feature = base_feature + obs_features_per_body[body_idx]

            # Aggregate children
            if len(children) > 0:
                child_contributions = []
                for child_idx in children:
                    F_child = body_features[child_idx]
                    contribution = self._compute_contribution(F_child, child_idx)
                    child_contributions.append(contribution)

                child_sum = jnp.stack(child_contributions, axis=0).sum(axis=0)

                if self.learnable_contribution_weight:
                    weight = jax.nn.sigmoid(self.contribution_weight[idx])
                else:
                    weight = 0.1

                body_features[body_idx] = base_feature + weight * child_sum
            else:
                body_features[body_idx] = base_feature

            # Apply LayerNorm
            body_features[body_idx] = self.link_norms[body_idx](body_features[body_idx])

        # Stack into tensor
        link_features = jnp.stack([body_features[i] for i in range(self.num_bodies)], axis=0)

        # Optional global normalization
        if self.global_norm is not None:
            shape = link_features.shape
            link_features = self.global_norm(link_features.reshape(self.num_bodies, -1))
            link_features = link_features.reshape(shape)

        return link_features

    def _compute_contribution(self, F: jax.Array, body_idx: int) -> jax.Array:
        """
        Compute: F - diag(F W W^T F)
        """
        W = self.motion_basis[body_idx]  # (C, d, d)

        FW = F[:, :, None] * W
        FWWt = FW @ W.transpose(0, 2, 1)
        proj_full = FWWt * F[:, None, :]
        proj_diag = jnp.diagonal(proj_full, axis1=-2, axis2=-1)

        return F - proj_diag

    def compute_orthogonality_loss(self, observations: jax.Array) -> jax.Array:
        """Compute orthogonality regularization."""
        link_features = self(observations)

        loss = 0.0
        for body_idx in range(self.num_bodies):
            F = link_features[body_idx]
            W = self.motion_basis[body_idx]

            FW = F[:, :, None] * W
            gram = W.transpose(0, 2, 1) @ FW

            loss = loss + ((gram - jax.lax.stop_gradient(self._identity)) ** 2).mean()

        return loss / self.num_bodies

    @property
    def output_dim(self) -> tuple[int, int]:
        return (self.num_bodies, self.link_channels * self.spatial_dim)


class PerBodyABATopDownLayer(eqx.Module):
    """
    ABA Top-Down Pass: propagates global context from root → leaves.

    Takes bottom-up features and refines them by passing parent context
    down the kinematic tree. Each child receives its parent's top-down
    feature (projected) added to its own bottom-up feature.

    Output: (num_bodies, link_channels, spatial_dim) — same shape as input.
    """

    # Per-body linear projections for parent → child message
    parent_to_child: tuple[eqx.nn.Linear, ...]
    link_norms: tuple[eqx.nn.LayerNorm, ...]
    gate_bias: jax.Array  # per-body learnable gate bias

    # Static
    num_bodies: int = eqx.field(static=True)
    link_channels: int = eqx.field(static=True)
    spatial_dim: int = eqx.field(static=True)
    traversal_order: tuple[int, ...] = eqx.field(static=True)  # root-first
    parent_map: tuple[int, ...] = eqx.field(static=True)

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

        # Top-down order = reverse of bottom-up
        self.traversal_order = tuple(reversed(kinematic_tree.traverse_bottom_up()))
        self.parent_map = tuple(kinematic_tree.parent_indices)

        feature_dim = link_channels * spatial_dim
        keys = jax.random.split(key, self.num_bodies)

        # Per-body projection: parent feature → child message
        # Small init so top-down starts as mild correction
        projections = []
        for i in range(self.num_bodies):
            lin = eqx.nn.Linear(feature_dim, feature_dim, key=keys[i])
            # Scale down initial weights for stability
            lin = eqx.tree_at(
                lambda l: l.weight, lin, lin.weight * 0.1
            )
            lin = eqx.tree_at(
                lambda l: l.bias, lin, jnp.zeros_like(lin.bias)
            )
            projections.append(lin)

        self.parent_to_child = tuple(projections)

        self.link_norms = tuple(
            eqx.nn.LayerNorm((link_channels, spatial_dim))
            for _ in range(self.num_bodies)
        )

        # Learnable gate: sigmoid(bias) controls how much top-down info to mix in
        # Init at -2.0 → sigmoid ≈ 0.12, so starts as small correction
        self.gate_bias = jnp.full((self.num_bodies,), -2.0)

    def __call__(self, bottom_up_features: jax.Array) -> jax.Array:
        """
        Args:
            bottom_up_features: (num_bodies, link_channels, spatial_dim)

        Returns:
            top_down_features: (num_bodies, link_channels, spatial_dim)
        """
        C, d = self.link_channels, self.spatial_dim
        td = {}

        for body_idx in self.traversal_order:
            parent_idx = self.parent_map[body_idx]
            bu = bottom_up_features[body_idx]  # (C, d)

            if parent_idx == -1:
                # Root: no parent context, just pass through
                td[body_idx] = bu
            else:
                # Project parent's top-down feature → message for this child
                parent_flat = td[parent_idx].reshape(-1)  # (C*d,)
                message = self.parent_to_child[body_idx](parent_flat)
                message = message.reshape(C, d)

                # Gated residual: td = bu + gate * message
                gate = jax.nn.sigmoid(self.gate_bias[body_idx])
                td[body_idx] = bu + gate * message

            td[body_idx] = self.link_norms[body_idx](td[body_idx])

        return jnp.stack([td[i] for i in range(self.num_bodies)], axis=0)