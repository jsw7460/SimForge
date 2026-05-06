from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.policies.sac_ac import SACActorCritic
from rlworld.rl.storages.replay_buffer import ReplayBatch


def compute_critic_loss(
    model: SACActorCritic,
    target_critic1_params: Any,
    target_critic2_params: Any,
    critic1_static: Any,
    critic2_static: Any,
    batch: ReplayBatch,
    gamma: float,
    ent_coef: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Compute critic loss for SAC.

    Args:
        model: Current SAC model
        target_critic1_params: Target critic1 parameters
        target_critic2_params: Target critic2 parameters
        critic1_static: Static parts of critic1
        critic2_static: Static parts of critic2
        batch: Replay batch
        gamma: Discount factor
        ent_coef: Entropy coefficient
        key: JAX random key

    Returns:
        Tuple of (total_critic_loss, info_dict)
    """
    # Sample next actions from current policy
    next_actions, next_log_prob, _ = model.act_with_log_prob(batch.next_actor_observations, key=key)

    # Compute target Q-values using target critics
    target_critic1 = eqx.combine(target_critic1_params, critic1_static)
    target_critic2 = eqx.combine(target_critic2_params, critic2_static)

    # Normalize observations for target critics
    normalized_next_critic_obs = model._normalize_critic_obs(batch.next_critic_observations)

    target_q1 = target_critic1(normalized_next_critic_obs, next_actions)
    target_q2 = target_critic2(normalized_next_critic_obs, next_actions)

    # Take minimum to reduce overestimation bias
    target_q = jnp.minimum(target_q1, target_q2)

    # Add entropy term
    target_q = target_q - ent_coef * next_log_prob[..., None]

    # Compute target value (Bellman backup)
    target_q_values = batch.rewards + (1 - batch.terminated) * batch.gamma_power * target_q
    target_q_values = jax.lax.stop_gradient(target_q_values)

    # Compute current Q-values
    current_q1 = model.critic1_forward(batch.critic_observations, batch.actions)
    current_q2 = model.critic2_forward(batch.critic_observations, batch.actions)

    # Critic loss (MSE)
    critic1_loss = jnp.mean((current_q1 - target_q_values) ** 2)
    critic2_loss = jnp.mean((current_q2 - target_q_values) ** 2)
    critic_loss = 0.5 * (critic1_loss + critic2_loss)

    info = {
        "critic1_loss": critic1_loss,
        "critic2_loss": critic2_loss,
        "critic_loss": critic_loss,
        "q1_value": jnp.mean(current_q1),
        "q2_value": jnp.mean(current_q2),
        "target_q_value": jnp.mean(target_q_values),
        "current_q1_std": jnp.std(current_q1),
        "current_q2_std": jnp.std(current_q2),
    }

    return critic_loss, info


def compute_actor_loss(
    model: SACActorCritic,
    batch: ReplayBatch,
    ent_coef: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, Dict[str, jax.Array]]:
    """
    Compute actor loss for SAC.

    Actor loss: minimize alpha * log_prob - Q

    Args:
        model: Current SAC model
        batch: Replay batch
        ent_coef: Entropy coefficient
        key: JAX random key

    Returns:
        Tuple of (actor_loss, log_prob, info_dict)
    """
    # Sample actions and compute log probabilities
    actions, log_prob, _ = model.act_with_log_prob(batch.actor_observations, key=key)

    # Compute Q-values for sampled actions
    q1_pi = model.critic1_forward(batch.critic_observations, actions)
    q2_pi = model.critic2_forward(batch.critic_observations, actions)
    min_q_pi = jnp.minimum(q1_pi, q2_pi)

    # Actor loss: maximize Q - alpha * log_prob
    # Equivalent to: minimize alpha * log_prob - Q
    actor_loss = jnp.mean(ent_coef * log_prob[..., None] - min_q_pi)

    info = {
        "actor_loss": actor_loss,
        "entropy": -jnp.mean(log_prob),
    }

    return actor_loss, log_prob, info


def compute_alpha_loss(
    log_ent_coef: jax.Array,
    log_prob: jax.Array,
    target_entropy: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute entropy coefficient loss for automatic tuning.

    Args:
        log_ent_coef: Log entropy coefficient
        log_prob: Log probabilities of actions
        target_entropy: Target entropy value

    Returns:
        Tuple of (alpha_loss, current_alpha)
    """
    ent_coef = jnp.exp(log_ent_coef)
    alpha_loss = -jnp.mean(log_ent_coef * jax.lax.stop_gradient(log_prob + target_entropy))
    return alpha_loss, ent_coef
