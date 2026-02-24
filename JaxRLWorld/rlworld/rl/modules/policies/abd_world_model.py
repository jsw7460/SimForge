"""
ABD-Net World Model for TD-MPC2.

Key changes from TDMPC2WorldModel:
- Encoder: REMOVED (physical state IS the latent state)
- Dynamics: ABD-Net (ABA bottom-up pass with residual delta prediction)
- Reward, Q-ensemble, Policy: Standard MLPs on physical state (unchanged logic)
- Consistency loss: direct MSE in physical state space

This preserves all TD-MPC2 interfaces (encode, next_latent, predict_reward,
predict_q, q_value, pi) so it is a drop-in replacement.
"""

from typing import Optional, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.algorithms.tdmpc2.math import (
    TwoHotConfig,
    two_hot_inv,
    log_std_transform,
    gaussian_logprob,
    squash,
)
from rlworld.rl.modules.policies.tdmpc2_world_model import (
    TDMPC2MLP,
    QEnsemble,
)

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree

from rlworld.rl.algorithms.scaffolded_tdmpc2.abd_dynamics import ABDDynamics


def simnorm(x: jax.Array, simplex_dim: int = 8) -> jax.Array:
    """
    SimNorm: split into groups, apply softmax per group.

    Handles non-divisible dimensions by padding the last group.

    Args:
        x: (..., dim)
        simplex_dim: size of each softmax group

    Returns:
        Normalized array of same shape, values in [0, 1]
    """
    shape = x.shape
    dim = shape[-1]
    remainder = dim % simplex_dim

    if remainder == 0:
        x = x.reshape(*shape[:-1], -1, simplex_dim)
        x = jax.nn.softmax(x, axis=-1)
        return x.reshape(shape)

    # Split into full groups + remainder group
    full_part = x[..., :dim - remainder]
    last_part = x[..., dim - remainder:]

    full_part = full_part.reshape(*shape[:-1], -1, simplex_dim)
    full_part = jax.nn.softmax(full_part, axis=-1)
    full_part = full_part.reshape(*shape[:-1], -1)

    last_part = jax.nn.softmax(last_part, axis=-1)

    return jnp.concatenate([full_part, last_part], axis=-1)


class ABDNetWorldModel(eqx.Module):
    """
    TD-MPC2 World Model with ABD-Net dynamics and no encoder.

    Physical state is used directly as the "latent" state.
    All method signatures match TDMPC2WorldModel exactly.
    """

    # Networks
    dynamics: ABDDynamics
    reward_head: TDMPC2MLP
    q_ensemble: QEnsemble
    policy: TDMPC2MLP
    pi_encoder_weight: jax.Array  # (obs_dim, obs_dim)
    pi_encoder_bias: jax.Array    # (obs_dim,)

    # Configuration (static) — matches TDMPC2WorldModel fields
    latent_dim: int = eqx.field(static=True)   # = obs_dim
    action_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)
    num_q: int = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    simnorm_dim: int = eqx.field(static=True)

    # Squash control: True = tanh squashing (original), False = raw Gaussian output
    squash_action: bool = eqx.field(static=True)

    # Log-std bounds
    log_std_min: float = eqx.field(static=True)
    log_std_dif: float = eqx.field(static=True)

    # Action bounds (static tuples for JIT compatibility)
    # Used by MPPI planning when squash_action=False.
    # When squash_action=True, these are (-1.0, ...) and (1.0, ...) (original behavior).
    action_low_tuple: tuple = eqx.field(static=True)
    action_high_tuple: tuple = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        action_dim: int,
        mlp_dim: int = 512,
        num_q: int = 5,
        num_bins: int = 101,
        dropout: float = 0.01,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        # ABD-Net specific
        link_channels: int = 8,
        spatial_dim: int = 6,
        learnable_contribution_weight: bool = False,
        use_positive_constraint: bool = True,
        residual_scale_init: float = 0.1,
        simnorm_dim: int = 8,
        action_low: tuple = None,
        action_high: tuple = None,
        *,
        key: jax.Array,
    ):
        self.obs_dim = obs_dim
        self.latent_dim = obs_dim  # no encoder compression
        self.action_dim = action_dim
        self.num_q = num_q
        self.num_bins = num_bins
        self.log_std_min = log_std_min
        self.log_std_dif = log_std_max - log_std_min
        self.simnorm_dim = simnorm_dim
        self.squash_action = True

        if self.squash_action:
            # Original TD-MPC2: tanh outputs in [-1, 1]
            self.action_low_tuple = tuple([-1.0] * action_dim)
            self.action_high_tuple = tuple([1.0] * action_dim)
        else:
            if action_low is None or action_high is None:
                raise ValueError(
                    "action_low and action_high must be provided when squash_action=False"
                )
            self.action_low_tuple = tuple(float(x) for x in action_low)
            self.action_high_tuple = tuple(float(x) for x in action_high)

        out_bins = max(num_bins, 1)

        k_dyn, k_rew, k_q, k_pi, k_pi_enc = jax.random.split(key, 5)

        # Dynamics: ABD-Net (replaces MLP dynamics + encoder)
        self.dynamics = ABDDynamics(
            kinematic_tree=kinematic_tree,
            state_dim=obs_dim,
            action_dim=action_dim,
            link_channels=link_channels,
            spatial_dim=spatial_dim,
            learnable_contribution_weight=learnable_contribution_weight,
            use_positive_constraint=use_positive_constraint,
            residual_scale_init=residual_scale_init,
            key=k_dyn,
        )

        # Reward: (state, action) -> reward logits
        # Same structure as TDMPC2WorldModel.reward_head
        self.reward_head = TDMPC2MLP(
            in_dim=obs_dim + action_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=out_bins,
            act=None,
            key=k_rew,
        )

        # Q-ensemble: (state, action) -> Q logits
        # Same structure as TDMPC2WorldModel.q_ensemble
        self.q_ensemble = QEnsemble(
            num_q=num_q,
            in_dim=obs_dim + action_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=out_bins,
            dropout=dropout,
            key=k_q,
        )

        # Policy: state -> (mean, log_std)
        # Same structure as TDMPC2WorldModel.policy
        self.policy = TDMPC2MLP(
            in_dim=obs_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=2 * action_dim,
            act=None,
            key=k_pi,
        )

        # Lightweight linear projection for policy input (before SimNorm)
        self.pi_encoder_weight = jax.random.normal(
            k_pi_enc, (obs_dim, obs_dim)
        ) * (1.0 / obs_dim ** 0.5)
        self.pi_encoder_bias = jnp.zeros(obs_dim)

        # Zero-init reward head output (matches TDMPC2WorldModel.__init__)
        self.reward_head = eqx.tree_at(
            lambda m: m.layers[-1].weight,
            self.reward_head,
            jnp.zeros_like(self.reward_head.layers[-1].weight),
        )

        # Zero-init Q-function outputs (matches TDMPC2WorldModel.__init__)
        q_fns = list(self.q_ensemble.q_functions)
        for i in range(num_q):
            q_fns[i] = eqx.tree_at(
                lambda q: q.net.layers[-1].weight,
                q_fns[i],
                jnp.zeros_like(q_fns[i].net.layers[-1].weight),
            )
        self.q_ensemble = eqx.tree_at(
            lambda m: m.q_functions,
            self.q_ensemble,
            tuple(q_fns),
        )

    # ==================== Forward Methods ====================
    # Identical signatures to TDMPC2WorldModel.

    def encode(self, obs: jax.Array) -> jax.Array:
        """Identity encoder: physical state IS the latent."""
        return obs

    def next_latent(self, z: jax.Array, a: jax.Array) -> jax.Array:
        """ABD-Net dynamics: s' = s + scale * delta(s, a). Batched."""
        return jax.vmap(self.dynamics)(z, a)

    def predict_reward(self, z: jax.Array, a: jax.Array) -> jax.Array:
        """Predict reward logits."""
        za = jnp.concatenate([z, a], axis=-1)
        return self.reward_head(za)

    def predict_q(
        self,
        z: jax.Array,
        a: jax.Array,
        *,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> jax.Array:
        """Predict Q-values from ensemble."""
        za = jnp.concatenate([z, a], axis=-1)
        return self.q_ensemble(za, key=key, inference=inference)

    def q_value(
        self,
        z: jax.Array,
        a: jax.Array,
        two_hot_cfg: TwoHotConfig,
        return_type: str = "min",
        *,
        key: jax.Array,
        inference: bool = True,
    ) -> jax.Array:
        """Get scalar Q-value from ensemble."""
        key1, key2 = jax.random.split(key)
        q_logits = self.predict_q(z, a, key=key1, inference=inference)

        if return_type == "all":
            return q_logits

        q_idx = jax.random.permutation(key2, self.num_q)[:2]
        q_selected = q_logits[q_idx]
        q_values = two_hot_inv(q_selected, two_hot_cfg)

        if return_type == "min":
            return q_values.min(axis=0)
        else:
            return q_values.mean(axis=0)

    def pi(
        self,
        z: jax.Array,
        *,
        key: jax.Array,
    ) -> tuple[jax.Array, dict]:
        """Sample action from Gaussian policy prior."""
        raw = self._pi_forward(z)
        mean, log_std_raw = jnp.split(raw, 2, axis=-1)

        log_std = log_std_transform(log_std_raw, self.log_std_min, self.log_std_dif)
        eps = jax.random.normal(key, mean.shape)

        log_prob = gaussian_logprob(eps, log_std)

        action_dims = float(self.action_dim)
        scaled_log_prob = log_prob * action_dims

        action = mean + eps * jnp.exp(log_std)
        mean, action, log_prob = squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)

        info = {
            "mean": mean,
            "log_std": log_std,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def _pi_forward(self, z: jax.Array) -> jax.Array:
        """Raw policy forward pass with lightweight SimNorm encoding."""
        z_enc = z @ self.pi_encoder_weight.T + self.pi_encoder_bias
        z_enc = simnorm(z_enc, self.simnorm_dim)
        return self.policy(z_enc)

    # ==================== ABD-Net Specific ====================

    def compute_dynamics_orthogonality_loss(
        self, state: jax.Array, action: jax.Array
    ) -> jax.Array:
        """Compute ABD-Net orthogonality regularization for dynamics."""
        return self.dynamics.compute_orthogonality_loss(state, action)

    def act_inference(self, obs: jax.Array, *, key: jax.Array) -> tuple[jax.Array, dict]:
        raise NotImplementedError()
        """Deterministic action from policy mean."""
        raw = self._pi_forward(obs)
        mean, _ = jnp.split(raw, 2, axis=-1)
        action = jnp.tanh(mean)
        return action, {"mean": action}