from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from rlworld.rl.algorithms.tdmpc2.math import (
    TwoHotConfig,
    soft_ce,
    two_hot_inv,
)
from rlworld.rl.modules.policies.tdmpc2_world_model import (
    QEnsemble,
    TDMPC2WorldModel,
)
from rlworld.rl.storages.sequence_replay_buffer import SequenceBatch


class WorldModelLossInfo(NamedTuple):
    consistency_loss: jax.Array
    reward_loss: jax.Array
    value_loss: jax.Array
    termination_loss: jax.Array
    total_loss: jax.Array


class PolicyLossInfo(NamedTuple):
    pi_loss: jax.Array
    entropy: jax.Array
    scaled_entropy: jax.Array


def compute_world_model_loss(
    model: TDMPC2WorldModel,
    target_q_ensemble: QEnsemble,
    batch: SequenceBatch,
    two_hot_cfg: TwoHotConfig,
    discount: float,
    rho: float,
    consistency_coef: float,
    reward_coef: float,
    value_coef: float,
    episodic: bool = False,
    termination_coef: float = 5.0,
    *,
    key: jax.Array,
) -> tuple[jax.Array, WorldModelLossInfo]:
    """Compute total world model loss."""
    obs = batch.observations  # [H+1, B, obs_dim]
    actions = batch.actions  # [H, B, action_dim]
    rewards = batch.rewards  # [H, B, 1]
    terminated = batch.terminated  # [H, B, 1]
    horizon = actions.shape[0]

    key, td_key, q_key = jax.random.split(key, 3)

    # Encode all next observations (consistency targets, fixed)
    next_z_targets = jax.lax.stop_gradient(jax.vmap(model.encode)(obs[1:]))  # [H, B, latent_dim]

    # TD-targets (vectorized)
    td_targets = _compute_td_targets(
        model,
        target_q_ensemble,
        next_z_targets,
        rewards,
        terminated,
        discount,
        two_hot_cfg,
        td_key,
    )  # [H, B, 1]

    # Sequential dynamics rollout (cannot vectorize due to sequential dependency)
    zs_pre = []  # z before dynamics: for reward/Q prediction
    zs_post = []  # z after dynamics: for consistency
    z = model.encode(obs[0])
    for t in range(horizon):
        zs_pre.append(z)
        z = model.next_latent(z, actions[t])
        zs_post.append(z)
    zs_pre = jnp.stack(zs_pre, axis=0)  # [H, B, latent_dim]
    zs_post = jnp.stack(zs_post, axis=0)  # [H, B, latent_dim]

    rho_weights = rho ** jnp.arange(horizon)  # [H]

    # Vectorized consistency loss
    consistency_per_t = jnp.mean((zs_post - next_z_targets) ** 2, axis=(-1, -2))  # [H]
    consistency_loss = jnp.sum(consistency_per_t * rho_weights) / horizon

    # Vectorized reward loss
    def _reward_loss_single(z, a, r):
        pred = model.predict_reward(z, a)
        return soft_ce(pred, r, two_hot_cfg).mean()

    reward_losses = jax.vmap(_reward_loss_single)(zs_pre, actions, rewards)  # [H]
    reward_loss = jnp.sum(reward_losses * rho_weights) / horizon

    # Vectorized value loss
    q_keys = jax.random.split(q_key, horizon)

    def _value_loss_single(z, a, td_target, k):
        q_preds = model.predict_q(z, a, key=k)  # [num_q, B, bins]
        td_broadcast = jnp.broadcast_to(td_target[None], (q_preds.shape[0],) + td_target.shape)  # [num_q, B, 1]
        q_ce = jax.vmap(soft_ce, in_axes=(0, 0, None))(q_preds, td_broadcast, two_hot_cfg)  # [num_q, B, 1]
        return q_ce.mean()

    value_losses = jax.vmap(_value_loss_single)(zs_pre, actions, td_targets, q_keys)  # [H]
    value_loss = jnp.sum(value_losses * rho_weights) / horizon

    # Termination loss (episodic mode only)
    if episodic:
        term_logits = jax.vmap(model.predict_termination)(zs_post)  # [H, B, 1]
        term_labels = terminated  # [H, B, 1]
        termination_loss = optax.sigmoid_binary_cross_entropy(
            term_logits,
            term_labels,
        ).mean()
    else:
        termination_loss = jnp.float32(0.0)

    total_loss = (
        consistency_coef * consistency_loss
        + reward_coef * reward_loss
        + value_coef * value_loss
        + termination_coef * termination_loss
    )

    return total_loss, WorldModelLossInfo(
        consistency_loss=consistency_loss,
        reward_loss=reward_loss,
        value_loss=value_loss,
        termination_loss=termination_loss,
        total_loss=total_loss,
    )


def _compute_td_targets(
    model: TDMPC2WorldModel,
    target_q_ensemble: QEnsemble,
    next_z: jax.Array,
    rewards: jax.Array,
    terminated: jax.Array,
    discount: float,
    two_hot_cfg: TwoHotConfig,
    key: jax.Array,
) -> jax.Array:
    """Vectorized TD-target: r + gamma * (1 - done) * min Q_target(z', pi(z'))"""
    horizon = next_z.shape[0]
    key, pi_key, q_key = jax.random.split(key, 3)
    pi_keys = jax.random.split(pi_key, horizon)
    q_keys = jax.random.split(q_key, horizon)

    def _single_td(z, r, done, pi_k, q_k):
        action_next, _ = model.pi(z, key=pi_k)
        za = jnp.concatenate([z, action_next], axis=-1)
        q_logits = target_q_ensemble(za, inference=True)  # [num_q, B, bins]
        q_idx = jax.random.permutation(q_k, model.num_q)[:2]
        q_selected = q_logits[q_idx]  # [2, B, bins]
        q_values = two_hot_inv(q_selected, two_hot_cfg)  # [2, B, 1]
        target_q = q_values.min(axis=0)
        return r + discount * (1.0 - done) * target_q

    td_targets = jax.vmap(_single_td)(next_z, rewards, terminated, pi_keys, q_keys)  # [H, B, 1]

    return jax.lax.stop_gradient(td_targets)


def compute_policy_loss(
    model: TDMPC2WorldModel,
    zs: jax.Array,
    two_hot_cfg: TwoHotConfig,
    rho: float,
    entropy_coef: float,
    scale_value: float,
    *,
    key: jax.Array,
) -> tuple[jax.Array, PolicyLossInfo]:
    """Vectorized policy loss: maximize entropy-regularized Q-value."""
    horizon_plus_one = zs.shape[0]

    key, pi_key, q_key = jax.random.split(key, 3)
    pi_keys = jax.random.split(pi_key, horizon_plus_one)
    q_keys = jax.random.split(q_key, horizon_plus_one)

    rho_weights = rho ** jnp.arange(horizon_plus_one)  # [H+1]

    def _single_step(z, pi_k, q_k):
        action, info = model.pi(z, key=pi_k)
        q_val = model.q_value(
            z,
            action,
            two_hot_cfg,
            return_type="avg",
            key=q_k,
        )
        q_normalized = q_val / jnp.maximum(scale_value, 1.0)
        step_loss = -(entropy_coef * info["scaled_entropy"] + q_normalized).mean()
        return step_loss, info["entropy"].mean(), info["scaled_entropy"].mean()

    step_losses, entropies, scaled_entropies = jax.vmap(_single_step)(zs, pi_keys, q_keys)  # each [H+1]

    pi_loss = jnp.sum(step_losses * rho_weights) / horizon_plus_one

    return pi_loss, PolicyLossInfo(
        pi_loss=pi_loss,
        entropy=jnp.sum(entropies) / horizon_plus_one,
        scaled_entropy=jnp.sum(scaled_entropies) / horizon_plus_one,
    )
