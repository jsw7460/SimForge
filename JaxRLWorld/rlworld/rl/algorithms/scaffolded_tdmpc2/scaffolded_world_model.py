"""
Scaffolded World Model with ABD-Net dynamics on s+.

Training-time only. Operates on s+ = [s-, s_priv].
At deployment, only ABDNetWorldModel (on s-) is used.
"""

from typing import Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
import equinox as eqx

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
from rlworld.rl.modules.policies.abd_world_model import simnorm


class ScaffoldedWorldModel(eqx.Module):
    """
    Scaffolded world model with ABD-Net dynamics on s+.

    Same forward interface as ABDNetWorldModel, plus pi_explore
    for scaffolded exploration policy.
    """

    dynamics: ABDDynamics
    reward_head: TDMPC2MLP
    q_ensemble: QEnsemble
    exploration_policy: TDMPC2MLP
    pi_encoder_weight: jax.Array  # (scaffolded_dim, scaffolded_dim)
    pi_encoder_bias: jax.Array    # (scaffolded_dim,)

    scaffolded_dim: int = eqx.field(static=True)
    target_obs_dim: int = eqx.field(static=True)
    privileged_obs_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)   # = scaffolded_dim
    num_q: int = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    simnorm_dim: int = eqx.field(static=True)

    log_std_min: float = eqx.field(static=True)
    log_std_dif: float = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        target_obs_dim: int,
        privileged_obs_dim: int,
        action_dim: int,
        mlp_dim: int = 512,
        num_q: int = 5,
        num_bins: int = 101,
        dropout: float = 0.01,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        link_channels: int = 8,
        spatial_dim: int = 6,
        learnable_contribution_weight: bool = False,
        use_positive_constraint: bool = True,
        residual_scale_init: float = 0.1,
        simnorm_dim: int = 8,
        *,
        key: jax.Array,
    ):
        self.target_obs_dim = target_obs_dim
        self.privileged_obs_dim = privileged_obs_dim
        self.scaffolded_dim = target_obs_dim + privileged_obs_dim
        self.latent_dim = self.scaffolded_dim
        self.action_dim = action_dim
        self.num_q = num_q
        self.num_bins = num_bins
        self.log_std_min = log_std_min
        self.log_std_dif = log_std_max - log_std_min
        self.simnorm_dim = simnorm_dim

        out_bins = max(num_bins, 1)
        k_dyn, k_rew, k_q, k_pi, k_pi_enc = jax.random.split(key, 5)

        self.dynamics = ABDDynamics(
            kinematic_tree=kinematic_tree,
            state_dim=self.scaffolded_dim,
            action_dim=action_dim,
            link_channels=link_channels,
            spatial_dim=spatial_dim,
            learnable_contribution_weight=learnable_contribution_weight,
            use_positive_constraint=use_positive_constraint,
            residual_scale_init=residual_scale_init,
            key=k_dyn,
        )

        self.reward_head = TDMPC2MLP(
            in_dim=self.scaffolded_dim + action_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=out_bins,
            act=None,
            key=k_rew,
        )

        self.q_ensemble = QEnsemble(
            num_q=num_q,
            in_dim=self.scaffolded_dim + action_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=out_bins,
            dropout=dropout,
            key=k_q,
        )

        self.exploration_policy = TDMPC2MLP(
            in_dim=self.scaffolded_dim,
            hidden_dims=[mlp_dim, mlp_dim],
            out_dim=2 * action_dim,
            act=None,
            key=k_pi,
        )

        # Lightweight linear projection for exploration policy input (before SimNorm)
        self.pi_encoder_weight = jax.random.normal(
            k_pi_enc, (self.scaffolded_dim, self.scaffolded_dim)
        ) * (1.0 / self.scaffolded_dim ** 0.5)
        self.pi_encoder_bias = jnp.zeros(self.scaffolded_dim)

        # Zero-init outputs
        self.reward_head = eqx.tree_at(
            lambda m: m.layers[-1].weight,
            self.reward_head,
            jnp.zeros_like(self.reward_head.layers[-1].weight),
        )

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

    def encode(self, scaffolded_obs: jax.Array) -> jax.Array:
        """Identity encoder."""
        return scaffolded_obs

    def next_latent(self, z_plus: jax.Array, a: jax.Array) -> jax.Array:
        """ABD-Net dynamics on s+. Batched."""
        return jax.vmap(self.dynamics)(z_plus, a)

    def predict_reward(self, z_plus: jax.Array, a: jax.Array) -> jax.Array:
        za = jnp.concatenate([z_plus, a], axis=-1)
        return self.reward_head(za)

    def predict_q(
        self, z_plus: jax.Array, a: jax.Array,
        *, key: Optional[jax.Array] = None, inference: bool = False,
    ) -> jax.Array:
        za = jnp.concatenate([z_plus, a], axis=-1)
        return self.q_ensemble(za, key=key, inference=inference)

    def q_value(
        self, z_plus: jax.Array, a: jax.Array,
        two_hot_cfg: TwoHotConfig, return_type: str = "min",
        *, key: jax.Array, inference: bool = True,
    ) -> jax.Array:
        key1, key2 = jax.random.split(key)
        q_logits = self.predict_q(z_plus, a, key=key1, inference=inference)
        if return_type == "all":
            return q_logits
        q_idx = jax.random.permutation(key2, self.num_q)[:2]
        q_selected = q_logits[q_idx]
        q_values = two_hot_inv(q_selected, two_hot_cfg)
        if return_type == "min":
            return q_values.min(axis=0)
        return q_values.mean(axis=0)

    def pi_explore(
        self, z_plus: jax.Array, *, key: jax.Array,
    ) -> tuple[jax.Array, dict]:
        """Scaffolded exploration policy. Training-time only."""
        z_enc = z_plus @ self.pi_encoder_weight.T + self.pi_encoder_bias
        z_enc = simnorm(z_enc, self.simnorm_dim)
        raw = self.exploration_policy(z_enc)
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
            "mean": mean, "log_std": log_std,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

    def compute_dynamics_orthogonality_loss(
        self, state: jax.Array, action: jax.Array,
    ) -> jax.Array:
        return self.dynamics.compute_orthogonality_loss(state, action)
