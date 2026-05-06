"""
Scaffolded TD-MPC2 loss functions.

Follows rlworld.rl.algorithms.tdmpc2.losses exactly:
- Same NamedTuple return types (extended with ortho field)
- Same function signatures and internal structure
- Same gradient flow invariants

Two world model losses:
- compute_abd_world_model_loss: target ABDNetWorldModel on s-
- compute_scaffolded_world_model_loss: ScaffoldedWorldModel on s+

Both use identity encoder -> consistency target is raw next state.
Both add orthogonality regularization for ABD-Net motion basis.

NOTE: Horizon loops replaced with jax.lax.scan to reduce XLA compile time.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rlworld.rl.algorithms.scaffolded_tdmpc2.scaffolded_world_model import (
    ScaffoldedWorldModel,
)
from rlworld.rl.algorithms.tdmpc2.math import (
    TwoHotConfig,
    soft_ce,
    two_hot_inv,
)
from rlworld.rl.modules.policies.abd_world_model import ABDNetWorldModel
from rlworld.rl.modules.policies.tdmpc2_world_model import QEnsemble
from rlworld.rl.storages.sequence_replay_buffer import SequenceBatch

# ==================== Loss Info Types ====================
# Extends tdmpc2 WorldModelLossInfo with orthogonality_loss field.


class ABDWorldModelLossInfo(NamedTuple):
    consistency_loss: jax.Array
    reward_loss: jax.Array
    value_loss: jax.Array
    orthogonality_loss: jax.Array
    total_loss: jax.Array


# ==================== Target World Model Loss (s-) ====================


def compute_abd_world_model_loss(
    model: ABDNetWorldModel,
    target_q_ensemble: QEnsemble,
    batch: SequenceBatch,
    two_hot_cfg: TwoHotConfig,
    discount: float,
    rho: float,
    consistency_coef: float,
    reward_coef: float,
    value_coef: float,
    ortho_coef: float = 0.01,
    *,
    key: jax.Array,
) -> tuple[jax.Array, ABDWorldModelLossInfo]:
    """
    Compute target ABD-Net world model loss on s-.

    Mirrors compute_world_model_loss from tdmpc2/losses.py:
    1. Consistency targets: stop_gradient(obs[1:])  (encoder = identity)
    2. TD targets (stop_gradient)
    3. Sequential dynamics rollout from obs[0] via lax.scan
    4. Consistency + reward + value losses with rho weighting
    5. + orthogonality regularization
    """
    obs = batch.observations  # [H+1, B, obs_dim]
    actions = batch.actions  # [H, B, action_dim]
    rewards = batch.rewards  # [H, B, 1]
    terminated = batch.terminated  # [H, B, 1]
    horizon = actions.shape[0]

    key, td_key, q_key = jax.random.split(key, 3)

    # Consistency targets (encoder = identity, so raw next states)
    next_z_targets = jax.lax.stop_gradient(obs[1:])  # [H, B, obs_dim]

    # TD targets
    td_targets = _compute_td_targets(
        model,
        target_q_ensemble,
        next_z_targets,
        rewards,
        terminated,
        discount,
        two_hot_cfg,
        td_key,
    )

    # Sequential dynamics rollout via lax.scan
    z0 = model.encode(obs[0])  # identity

    def _dynamics_step(z, a):
        z_next = model.next_latent(z, a)
        return z_next, (z, z_next)

    _, (zs_pre, zs_post) = jax.lax.scan(_dynamics_step, z0, actions)
    # zs_pre: [H, B, obs_dim], zs_post: [H, B, obs_dim]

    rho_weights = rho ** jnp.arange(horizon)

    # Consistency loss
    consistency_per_t = jnp.mean((zs_post - next_z_targets) ** 2, axis=(-1, -2))
    consistency_loss = jnp.sum(consistency_per_t * rho_weights) / horizon

    # Reward loss
    def _reward_loss_single(z, a, r):
        pred = model.predict_reward(z, a)
        return soft_ce(pred, r, two_hot_cfg).mean()

    reward_losses = jax.vmap(_reward_loss_single)(zs_pre, actions, rewards)
    reward_loss = jnp.sum(reward_losses * rho_weights) / horizon

    # Value loss
    q_keys = jax.random.split(q_key, horizon)

    def _value_loss_single(z, a, td_target, k):
        q_preds = model.predict_q(z, a, key=k)
        td_broadcast = jnp.broadcast_to(td_target[None], (q_preds.shape[0],) + td_target.shape)
        q_ce = jax.vmap(soft_ce, in_axes=(0, 0, None))(q_preds, td_broadcast, two_hot_cfg)
        return q_ce.mean()

    value_losses = jax.vmap(_value_loss_single)(zs_pre, actions, td_targets, q_keys)
    value_loss = jnp.sum(value_losses * rho_weights) / horizon

    # Orthogonality regularization
    ortho_loss = model.compute_dynamics_orthogonality_loss(obs[0][0], actions[0][0])

    total_loss = (
        consistency_coef * consistency_loss
        + reward_coef * reward_loss
        + value_coef * value_loss
        + ortho_coef * ortho_loss
    )

    return total_loss, ABDWorldModelLossInfo(
        consistency_loss=consistency_loss,
        reward_loss=reward_loss,
        value_loss=value_loss,
        orthogonality_loss=ortho_loss,
        total_loss=total_loss,
    )


# ==================== Scaffolded World Model Loss (s+) ====================


def compute_scaffolded_world_model_loss(
    scaffolded_model: ScaffoldedWorldModel,
    scaff_target_q_ensemble: QEnsemble,
    scaffolded_obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    terminated: jax.Array,
    two_hot_cfg: TwoHotConfig,
    discount: float,
    rho: float,
    consistency_coef: float,
    reward_coef: float,
    value_coef: float,
    ortho_coef: float = 0.01,
    *,
    key: jax.Array,
) -> tuple[jax.Array, ABDWorldModelLossInfo]:
    """
    Compute scaffolded ABD-Net world model loss on s+.

    Same structure as compute_abd_world_model_loss, operating on
    scaffolded observations s+ = [s-, s_priv].
    """
    horizon = actions.shape[0]
    key, td_key, q_key = jax.random.split(key, 3)

    next_z_targets = jax.lax.stop_gradient(scaffolded_obs[1:])

    td_targets = _compute_scaffolded_td_targets(
        scaffolded_model,
        scaff_target_q_ensemble,
        next_z_targets,
        rewards,
        terminated,
        discount,
        two_hot_cfg,
        td_key,
    )

    # Sequential dynamics rollout via lax.scan
    z0 = scaffolded_model.encode(scaffolded_obs[0])

    def _dynamics_step(z, a):
        z_next = scaffolded_model.next_latent(z, a)
        return z_next, (z, z_next)

    _, (zs_pre, zs_post) = jax.lax.scan(_dynamics_step, z0, actions)

    rho_weights = rho ** jnp.arange(horizon)

    consistency_per_t = jnp.mean((zs_post - next_z_targets) ** 2, axis=(-1, -2))
    consistency_loss = jnp.sum(consistency_per_t * rho_weights) / horizon

    def _reward_loss_single(z, a, r):
        pred = scaffolded_model.predict_reward(z, a)
        return soft_ce(pred, r, two_hot_cfg).mean()

    reward_losses = jax.vmap(_reward_loss_single)(zs_pre, actions, rewards)
    reward_loss = jnp.sum(reward_losses * rho_weights) / horizon

    q_keys = jax.random.split(q_key, horizon)

    def _value_loss_single(z, a, td_target, k):
        q_preds = scaffolded_model.predict_q(z, a, key=k)
        td_broadcast = jnp.broadcast_to(td_target[None], (q_preds.shape[0],) + td_target.shape)
        q_ce = jax.vmap(soft_ce, in_axes=(0, 0, None))(q_preds, td_broadcast, two_hot_cfg)
        return q_ce.mean()

    value_losses = jax.vmap(_value_loss_single)(zs_pre, actions, td_targets, q_keys)
    value_loss = jnp.sum(value_losses * rho_weights) / horizon

    ortho_loss = scaffolded_model.compute_dynamics_orthogonality_loss(scaffolded_obs[0][0], actions[0][0])

    total_loss = (
        consistency_coef * consistency_loss
        + reward_coef * reward_loss
        + value_coef * value_loss
        + ortho_coef * ortho_loss
    )

    return total_loss, ABDWorldModelLossInfo(
        consistency_loss=consistency_loss,
        reward_loss=reward_loss,
        value_loss=value_loss,
        orthogonality_loss=ortho_loss,
        total_loss=total_loss,
    )


# ==================== TD Target Helpers ====================


def _compute_td_targets(
    model,
    target_q_ensemble,
    next_z,
    rewards,
    terminated,
    discount,
    two_hot_cfg,
    key,
) -> jax.Array:
    """
    TD targets for target model.
    Identical to tdmpc2/losses.py _compute_td_targets,
    except next_z is raw state (encoder = identity).
    """
    horizon = next_z.shape[0]
    key, pi_key, q_key = jax.random.split(key, 3)
    pi_keys = jax.random.split(pi_key, horizon)
    q_keys = jax.random.split(q_key, horizon)

    def _single_td(z, r, done, pi_k, q_k):
        action_next, _ = model.pi(z, key=pi_k)
        za = jnp.concatenate([z, action_next], axis=-1)
        q_logits = target_q_ensemble(za, inference=True)
        q_idx = jax.random.permutation(q_k, model.num_q)[:2]
        q_selected = q_logits[q_idx]
        q_values = two_hot_inv(q_selected, two_hot_cfg)
        target_q = q_values.min(axis=0)
        return r + discount * (1.0 - done) * target_q

    td_targets = jax.vmap(_single_td)(next_z, rewards, terminated, pi_keys, q_keys)
    return jax.lax.stop_gradient(td_targets)


def _compute_scaffolded_td_targets(
    scaffolded_model,
    scaff_target_q_ensemble,
    next_z_plus,
    rewards,
    terminated,
    discount,
    two_hot_cfg,
    key,
) -> jax.Array:
    """TD targets for scaffolded model using exploration policy."""
    horizon = next_z_plus.shape[0]
    key, pi_key, q_key = jax.random.split(key, 3)
    pi_keys = jax.random.split(pi_key, horizon)
    q_keys = jax.random.split(q_key, horizon)

    def _single_td(z_plus, r, done, pi_k, q_k):
        action_next, _ = scaffolded_model.pi_explore(z_plus, key=pi_k)
        za = jnp.concatenate([z_plus, action_next], axis=-1)
        q_logits = scaff_target_q_ensemble(za, inference=True)
        q_idx = jax.random.permutation(q_k, scaffolded_model.num_q)[:2]
        q_selected = q_logits[q_idx]
        q_values = two_hot_inv(q_selected, two_hot_cfg)
        target_q = q_values.min(axis=0)
        return r + discount * (1.0 - done) * target_q

    td_targets = jax.vmap(_single_td)(next_z_plus, rewards, terminated, pi_keys, q_keys)
    return jax.lax.stop_gradient(td_targets)
