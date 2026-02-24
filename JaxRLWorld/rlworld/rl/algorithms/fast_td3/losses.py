from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.policies.fast_td3_ac import (
    FastTD3ActorCritic,
    project_distribution_batched,
)
from rlworld.rl.storages.replay_buffer import ReplayBatch


def compute_critic_loss(
    model: FastTD3ActorCritic,
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
    use_cdq: bool,
    key: jax.Array,
) -> tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Compute distributional critic loss for FastTD3 using C51.

    Uses cross-entropy loss between projected target distribution and current distribution.

    NOTE: Observation normalization is handled via model._normalize_actor_obs()
    and model._normalize_critic_obs() methods.

    Args:
        model: Current FastTD3 model
        target_actor_params: Target actor trainable parameters
        target_actor_static: Target actor static parts
        target_critic1_params: Target critic1 trainable parameters
        target_critic1_static: Target critic1 static parts
        target_critic2_params: Target critic2 trainable parameters
        target_critic2_static: Target critic2 static parts
        batch: Replay batch
        gamma: Discount factor
        target_policy_noise: Stddev of noise added to target actions
        target_noise_clip: Clipping range for target noise
        use_cdq: Whether to use Clipped Double Q-learning (min) or average
        key: JAX random key

    Returns:
        Tuple of (total_critic_loss, info_dict)
    """
    key, noise_key = jax.random.split(key)

    # Reconstruct target actor
    target_actor = eqx.combine(target_actor_params, target_actor_static)

    # Normalize next actor observations and get next actions
    next_actor_obs = model._normalize_actor_obs(batch.next_actor_observations)
    if next_actor_obs.ndim == 2:
        keys = jax.random.split(key, next_actor_obs.shape[0])
        next_actions_raw, _ = jax.vmap(target_actor)(next_actor_obs, key=keys)
    else:
        next_actions_raw, _ = target_actor(next_actor_obs, key=key)

    # Conditionally apply tanh based on model.is_squashed
    if model.is_squashed:
        next_actions = jnp.tanh(next_actions_raw)
        # Add clipped noise (target policy smoothing)
        noise = jax.random.normal(noise_key, next_actions.shape) * target_policy_noise
        noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
    else:
        # No tanh: add noise and clip to action bounds (assuming [-1, 1] for simplicity)
        noise = jax.random.normal(noise_key, next_actions_raw.shape) * target_policy_noise
        noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
        next_actions = next_actions_raw + noise

    # Reconstruct target critics
    target_critic1 = eqx.combine(target_critic1_params, target_critic1_static)
    target_critic2 = eqx.combine(target_critic2_params, target_critic2_static)

    # Normalize next critic observations
    next_critic_obs = model._normalize_critic_obs(batch.next_critic_observations)
    target_logits1 = target_critic1(next_critic_obs, next_actions)
    target_logits2 = target_critic2(next_critic_obs, next_actions)

    target_probs1 = jax.nn.softmax(target_logits1, axis=-1)
    target_probs2 = jax.nn.softmax(target_logits2, axis=-1)

    # Compute target Q-values for CDQ selection
    support = model.support
    target_q1 = jnp.sum(target_probs1 * support, axis=-1)
    target_q2 = jnp.sum(target_probs2 * support, axis=-1)

    # Select target distribution based on CDQ
    if use_cdq:
        # Use distribution corresponding to minimum Q-value
        use_q1 = target_q1 < target_q2
        target_probs = jnp.where(use_q1[:, None], target_probs1, target_probs2)
    else:
        # Average the distributions
        target_probs = (target_probs1 + target_probs2) / 2.0

    # Compute bootstrap mask
    bootstrap = (1.0 - batch.terminated.astype(jnp.float32))

    # Compute effective discount (supports n-step returns via gamma_power)
    discount = batch.gamma_power

    # Project target distribution
    projected_dist = project_distribution_batched(
        next_probs=target_probs,
        rewards=batch.rewards,
        bootstrap=bootstrap,
        discount=discount,
        num_atoms=model.num_atoms,
        v_min=model.v_min,
        v_max=model.v_max,
    )
    projected_dist = jax.lax.stop_gradient(projected_dist)

    # Compute current critic logits (normalization handled inside critic*_forward)
    current_logits1 = model.critic1_forward(batch.critic_observations, batch.actions)
    current_logits2 = model.critic2_forward(batch.critic_observations, batch.actions)

    # Cross-entropy loss: -sum(target * log_softmax(current))
    log_probs1 = jax.nn.log_softmax(current_logits1, axis=-1)
    log_probs2 = jax.nn.log_softmax(current_logits2, axis=-1)

    critic1_loss = -jnp.sum(projected_dist * log_probs1, axis=-1).mean()
    critic2_loss = -jnp.sum(projected_dist * log_probs2, axis=-1).mean()
    critic_loss = critic1_loss + critic2_loss

    # Compute Q-values for logging
    current_probs1 = jax.nn.softmax(current_logits1, axis=-1)
    current_probs2 = jax.nn.softmax(current_logits2, axis=-1)
    current_q1 = jnp.sum(current_probs1 * support, axis=-1)
    current_q2 = jnp.sum(current_probs2 * support, axis=-1)
    target_q = jnp.minimum(target_q1, target_q2)

    info = {
        "critic1_loss": critic1_loss,
        "critic2_loss": critic2_loss,
        "critic_loss": critic_loss,
        "q1_value": jnp.mean(current_q1),
        "q2_value": jnp.mean(current_q2),
        "target_q_value": jnp.mean(target_q),
        "q1_std": jnp.std(current_q1),
        "q2_std": jnp.std(current_q2),
        "q1_max": jnp.max(current_q1),
        "q1_min": jnp.min(current_q1),
    }

    return critic_loss, info


def compute_actor_loss(
    model: FastTD3ActorCritic,
    batch: ReplayBatch,
    use_cdq: bool,
    key: jax.Array,
) -> tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Compute actor loss for FastTD3.

    NOTE: Observation normalization is handled via model.act() and
    model.critic*_q_value() methods which internally call _normalize_*_obs().
    """
    # model.act() handles normalization internally via _normalize_actor_obs()
    actions, _ = model.act(batch.actor_observations, key=key)

    # model.critic*_q_value() handles normalization internally via _normalize_critic_obs()
    q1 = model.critic1_q_value(batch.critic_observations, actions)
    q2 = model.critic2_q_value(batch.critic_observations, actions)

    if use_cdq:
        q_value = jnp.minimum(q1, q2)
    else:
        q_value = (q1 + q2) / 2.0

    actor_loss = -jnp.mean(q_value)

    info = {
        "actor_loss": actor_loss,
        "action_mean": jnp.mean(actions),
        "action_std": jnp.std(actions),
        "actor_q_value": jnp.mean(q_value),
    }

    return actor_loss, info