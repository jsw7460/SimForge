"""
ABD-Net Dynamics Model.

Adapts the ABA bottom-up architecture from the policy encoder to serve
as a dynamics model: (state, action) -> delta_state.

Key differences from the policy ABAEncoder:
- Input: (state, action) concatenated, not just observations
- Output: delta_state vector (readout from per-body features)
- Used as: s_{t+1} = s_t + scale * ABDDynamicsLayer(s_t, a_t)
"""

import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.utils import MLP

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


class ABDDynamicsLayer(eqx.Module):
    """
    ABA Bottom-Up pass for dynamics prediction.

    Takes concatenated (state, action) as input, computes per-body features
    via ABA-structured message passing, then reads out delta_state.

    Mirrors PerBodyABABottomUpLayer from layers.py with:
    - input_dim = state_dim + action_dim (instead of obs_dim)
    - readout MLP appended (per-body features -> delta_state)
    """

    # Per-body input projections
    obs_projections: tuple[MLP, ...]
    link_base: jax.Array
    link_norms: tuple[eqx.nn.LayerNorm, ...]
    motion_basis: jax.Array

    # Optional learnable contribution weight
    contribution_weight: jax.Array | None

    # Readout: ABA features -> delta_state
    readout: MLP

    # Global normalization across all bodies
    global_norm: eqx.nn.LayerNorm

    # Static configuration
    num_bodies: int = eqx.field(static=True)
    link_channels: int = eqx.field(static=True)
    spatial_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    use_positive_constraint: bool = eqx.field(static=True)
    learnable_contribution_weight: bool = eqx.field(static=True)

    # Tree structure (static)
    traversal_order: tuple[int, ...] = eqx.field(static=True)
    children_map: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        state_dim: int,
        action_dim: int,
        link_channels: int = 8,
        spatial_dim: int = 6,
        learnable_contribution_weight: bool = False,
        use_positive_constraint: bool = True,
        *,
        key: jax.Array,
    ):
        self.num_bodies = kinematic_tree.num_bodies
        self.link_channels = link_channels
        self.spatial_dim = spatial_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_positive_constraint = use_positive_constraint
        self.learnable_contribution_weight = learnable_contribution_weight

        # Tree structure
        self.traversal_order = tuple(kinematic_tree.traverse_bottom_up())
        self.children_map = tuple(tuple(kinematic_tree.get_children(i)) for i in range(self.num_bodies))

        input_dim = state_dim + action_dim
        feature_dim = link_channels * spatial_dim
        hidden_dim = feature_dim * 2

        # Split keys
        keys = jax.random.split(key, self.num_bodies + 3)
        proj_keys = keys[: self.num_bodies]
        base_key = keys[self.num_bodies]
        motion_key = keys[self.num_bodies + 1]
        readout_key = keys[self.num_bodies + 2]

        # Per-body (state, action) projections (same structure as policy encoder)
        self.obs_projections = tuple(
            MLP(
                input_dim=input_dim,
                hidden_dims=[hidden_dim],
                output_dim=feature_dim,
                activation="relu",
                output_activation=None,
                use_layer_norm=False,
                key=proj_keys[i],
            )
            for i in range(self.num_bodies)
        )

        # Learnable base features (analogous to rigid-body inertia I_i)
        self.link_base = jax.random.normal(base_key, (self.num_bodies, link_channels, spatial_dim)) * 0.1

        # Per-body LayerNorm
        self.link_norms = tuple(eqx.nn.LayerNorm((link_channels, spatial_dim)) for _ in range(self.num_bodies))

        # Motion basis W per body (orthogonally initialized)
        self.motion_basis = self._init_orthogonal_basis(motion_key)

        # Contribution weight
        if learnable_contribution_weight:
            init_scale = 0.5
            init_logit = math.log(init_scale / (1 - init_scale))
            self.contribution_weight = jnp.full((len(self.traversal_order),), init_logit)
        else:
            self.contribution_weight = None

        # Readout: flatten ABA features -> delta_state
        total_feature_dim = self.num_bodies * feature_dim
        self.readout = MLP(
            input_dim=total_feature_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=state_dim,
            activation="relu",
            output_activation=None,
            use_layer_norm=False,
            key=readout_key,
        )

        # Global LayerNorm across all body features (stabilizes dynamics output)
        self.global_norm = eqx.nn.LayerNorm((total_feature_dim,))

    def _init_orthogonal_basis(self, key: jax.Array) -> jax.Array:
        """Initialize motion basis as orthonormal matrices."""
        shape = (
            self.num_bodies,
            self.link_channels,
            self.spatial_dim,
            self.spatial_dim,
        )
        W = jax.random.normal(key, shape)

        def orthogonalize(w):
            q, _ = jnp.linalg.qr(w)
            return q

        W = jax.vmap(jax.vmap(orthogonalize))(W)
        return W

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Predict delta_state via ABA bottom-up pass.

        Args:
            state: (state_dim,) unbatched physical state
            action: (action_dim,) unbatched action

        Returns:
            delta_state: (state_dim,) predicted state change
        """
        sa = jnp.concatenate([state, action], axis=-1)

        # Per-body (state, action) encoding
        obs_features_list = []
        for body_idx in range(self.num_bodies):
            obs_feat = self.obs_projections[body_idx](sa)
            obs_feat = obs_feat.reshape(self.link_channels, self.spatial_dim)
            obs_features_list.append(obs_feat)

        obs_features_per_body = jnp.stack(obs_features_list, axis=0)

        # Bottom-up ABA pass (leaf -> root, same as PerBodyABABottomUpLayer)
        body_features = {}

        for idx, body_idx in enumerate(self.traversal_order):
            children = self.children_map[body_idx]

            # Base feature (analogous to rigid-body inertia)
            if self.use_positive_constraint:
                base_feature = jax.nn.softplus(self.link_base[body_idx]) + 1e-6
            else:
                base_feature = self.link_base[body_idx]

            # Add (state, action) features
            base_feature = base_feature + obs_features_per_body[body_idx]

            # Aggregate children contributions (ABA leaf-to-root)
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

            # Per-body LayerNorm
            body_features[body_idx] = self.link_norms[body_idx](body_features[body_idx])

        # Stack and flatten for readout
        link_features = jnp.stack([body_features[i] for i in range(self.num_bodies)], axis=0)  # (num_bodies, C, d)
        flat_features = link_features.reshape(-1)  # (num_bodies * C * d,)

        # Global normalization across all bodies
        flat_features = self.global_norm(flat_features)

        # Readout: predict delta_state
        delta_state = self.readout(flat_features)
        return delta_state

    def _compute_contribution(self, F: jax.Array, body_idx: int) -> jax.Array:
        """
        Compute child contribution: F - diag(F W W^T F)

        Learnable approximation of ABA constraint elimination (Eq. aba_contribution).
        Identical to PerBodyABABottomUpLayer._compute_contribution.
        """
        W = self.motion_basis[body_idx]  # (C, d, d)

        FW = F[:, :, None] * W
        FWWt = FW @ W.transpose(0, 2, 1)
        proj_full = FWWt * F[:, None, :]
        proj_diag = jnp.diagonal(proj_full, axis1=-2, axis2=-1)

        return F - proj_diag

    def compute_orthogonality_loss(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Compute orthogonality regularization loss.

        Encourages W_j W_j^T to approximate an orthogonal projection.
        Identical structure to PerBodyABABottomUpLayer.compute_orthogonality_loss.
        """
        sa = jnp.concatenate([state, action], axis=-1)

        # Forward pass to get body features
        obs_features_list = []
        for body_idx in range(self.num_bodies):
            obs_feat = self.obs_projections[body_idx](sa)
            obs_feat = obs_feat.reshape(self.link_channels, self.spatial_dim)
            obs_features_list.append(obs_feat)

        obs_features_per_body = jnp.stack(obs_features_list, axis=0)

        body_features = {}
        for idx, body_idx in enumerate(self.traversal_order):
            children = self.children_map[body_idx]

            if self.use_positive_constraint:
                base_feature = jax.nn.softplus(self.link_base[body_idx]) + 1e-6
            else:
                base_feature = self.link_base[body_idx]

            base_feature = base_feature + obs_features_per_body[body_idx]

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

            body_features[body_idx] = self.link_norms[body_idx](body_features[body_idx])

        loss = 0.0
        for body_idx in range(self.num_bodies):
            F = body_features[body_idx]
            W = self.motion_basis[body_idx]

            FW = F[:, :, None] * W
            gram = W.transpose(0, 2, 1) @ FW
            loss = loss + ((gram - jnp.eye(self.spatial_dim)) ** 2).mean()

        return loss / self.num_bodies


class ABDDynamics(eqx.Module):
    """
    ABD-Net dynamics module with residual connection.

    s_{t+1} = s_t + scale * ABDDynamicsLayer(s_t, a_t)
    """

    dynamics_layer: ABDDynamicsLayer
    residual_scale: jax.Array

    state_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        state_dim: int,
        action_dim: int,
        link_channels: int = 8,
        spatial_dim: int = 6,
        learnable_contribution_weight: bool = False,
        use_positive_constraint: bool = True,
        residual_scale_init: float = 0.1,
        *,
        key: jax.Array,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dynamics_layer = ABDDynamicsLayer(
            kinematic_tree=kinematic_tree,
            state_dim=state_dim,
            action_dim=action_dim,
            link_channels=link_channels,
            spatial_dim=spatial_dim,
            learnable_contribution_weight=learnable_contribution_weight,
            use_positive_constraint=use_positive_constraint,
            key=key,
        )

        # Learnable residual scale (initialized small for stable early training)
        self.residual_scale = jnp.array(residual_scale_init)

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """
        Predict next state with residual connection.

        Args:
            state: (state_dim,) or (B, state_dim)
            action: (action_dim,) or (B, action_dim)

        Returns:
            next_state: same shape as state
        """
        delta = self.dynamics_layer(state, action)
        return state + self.residual_scale * delta

    def predict_delta(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Predict only the delta (for analysis/logging)."""
        return self.dynamics_layer(state, action)

    def compute_orthogonality_loss(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Compute ABA orthogonality regularization."""
        return self.dynamics_layer.compute_orthogonality_loss(state, action)
