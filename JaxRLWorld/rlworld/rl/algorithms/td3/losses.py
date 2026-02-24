from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.policies.td3_ac import TD3ActorCritic
from rlworld.rl.storages.replay_buffer import ReplayBatch


def compute_critic_loss(
    model: TD3ActorCritic,
    target_actor_params: Any,
    target_actor_static: Any,
    target_critic1_params: Any,
    target_critic1_static: Any,
    target_critic2_params: Any,
    target_critic2_static: Any,
    batch: ReplayBatch,
    gamma: float,
    target_policy_noise: float,
    target_noise_clip: float,
    key: jax.Array,
) -> tuple[jax.Array, Dict[str, jax.Array]]:
    key, noise_key = jax.random.split(key)

    target_actor = eqx.combine(target_actor_params, target_actor_static)

    normalized_next_obs = model._normalize_actor_obs(batch.next_actor_observations)

    if normalized_next_obs.ndim == 2:
        keys = jax.random.split(key, normalized_next_obs.shape[0])
        next_actions_raw, _ = jax.vmap(target_actor)(normalized_next_obs, key=keys)
    else:
        next_actions_raw, _ = target_actor(normalized_next_obs, key=key)

    next_actions = jnp.tanh(next_actions_raw)

    noise = jax.random.normal(noise_key, next_actions.shape) * target_policy_noise
    noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)

    target_critic1 = eqx.combine(target_critic1_params, target_critic1_static)
    target_critic2 = eqx.combine(target_critic2_params, target_critic2_static)

    normalized_next_critic_obs = model._normalize_critic_obs(batch.next_critic_observations)
    target_q1 = target_critic1(normalized_next_critic_obs, next_actions)
    target_q2 = target_critic2(normalized_next_critic_obs, next_actions)

    target_q = jnp.minimum(target_q1, target_q2)

    target_q_values = batch.rewards + (1 - batch.terminated) * batch.gamma_power * target_q
    target_q_values = jax.lax.stop_gradient(target_q_values)

    current_q1 = model.critic1_forward(batch.critic_observations, batch.actions)
    current_q2 = model.critic2_forward(batch.critic_observations, batch.actions)

    critic1_loss = jnp.mean((current_q1 - target_q_values) ** 2)
    critic2_loss = jnp.mean((current_q2 - target_q_values) ** 2)
    critic_loss = critic1_loss + critic2_loss

    info = {
        "critic1_loss": critic1_loss,
        "critic2_loss": critic2_loss,
        "critic_loss": critic_loss,
        "q1_value": jnp.mean(current_q1),
        "q2_value": jnp.mean(current_q2),
        "target_q_value": jnp.mean(target_q_values),
        "current_q1_std": jnp.std(current_q1),
        "current_q2_std": jnp.std(current_q2),
        # # DEBUG
        # "debug_next_actions_sum": jnp.sum(next_actions),
        # "debug_target_q1_sum": jnp.sum(target_q1),
        # "debug_target_q2_sum": jnp.sum(target_q2),
        # "debug_current_q1_sum": jnp.sum(current_q1),
        # "debug_current_q2_sum": jnp.sum(current_q2),
        # "debug_target_q_values_sum": jnp.sum(target_q_values),
        # "debug_batch_rewards_sum": jnp.sum(batch.rewards),
        # "debug_batch_terminated_sum": jnp.sum(batch.terminated.astype(jnp.float32)),
        # "debug_batch_gamma_power_sum": jnp.sum(batch.gamma_power),
        # "debug_target_q_sum": jnp.sum(target_q),
    }

    return critic_loss, info


def compute_actor_loss(
    model: TD3ActorCritic,
    batch: ReplayBatch,
    key: jax.Array,
) -> tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Compute actor loss for TD3.

    Actor loss: maximize Q(s, actor(s)) = minimize -Q(s, actor(s))

    Args:
        model: Current TD3 model
        batch: Replay batch
        key: JAX random key

    Returns:
        Tuple of (actor_loss, info_dict)
    """
    # Get actions from current actor (deterministic)
    actions, _ = model.act(batch.actor_observations, key=key)

    # Compute Q-value using only critic1 (as in original TD3)
    q1_pi = model.critic1_forward(batch.critic_observations, actions)

    # Actor loss: maximize Q = minimize -Q
    actor_loss = -jnp.mean(q1_pi)

    info = {
        "actor_loss": actor_loss,
        "action_mean": jnp.mean(actions),
        "action_std": jnp.std(actions),
    }

    return actor_loss, info
