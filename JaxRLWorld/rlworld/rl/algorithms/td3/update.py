from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from rlworld.rl.algorithms.base import polyak_update
from rlworld.rl.algorithms.td3.losses import (
    compute_actor_loss,
    compute_critic_loss,
)
from rlworld.rl.modules.policies.td3_ac import TD3ActorCritic
from rlworld.rl.storages.replay_buffer import ReplayBatch

# ==================== Forward Functions ====================


@eqx.filter_jit
def act_deterministic(
    model: TD3ActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compiled deterministic action selection."""
    key, subkey = jax.random.split(key)
    actions, _ = model.act(actor_obs, key=subkey)
    values = model.evaluate(actor_obs, critic_obs, key=key)
    return actions, values, key


@eqx.filter_jit
def act_with_noise(
    model: TD3ActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    exploration_noise: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compiled action selection with exploration noise."""
    key, action_key, noise_key = jax.random.split(key, 3)

    # Get deterministic action
    actions, _ = model.act(actor_obs, key=action_key)

    # Add exploration noise
    noise = jax.random.normal(noise_key, actions.shape) * exploration_noise
    actions = jnp.clip(actions + noise, -1.0, 1.0)

    values = model.evaluate(actor_obs, critic_obs, key=key)
    return actions, values, key


@eqx.filter_jit
def get_value(
    model: TD3ActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """JIT-compiled value estimation."""
    return model.evaluate(actor_obs, critic_obs, key=key)


# ==================== Update Functions ====================


@eqx.filter_jit
def update_critics(
    model: TD3ActorCritic,
    target_actor_params: Any,
    target_actor_static: Any,
    target_critic1_params: Any,
    target_critic1_static: Any,
    target_critic2_params: Any,
    target_critic2_static: Any,
    critic_opt_state: optax.OptState,
    batch: ReplayBatch,
    critic_optimizer: optax.GradientTransformation,
    gamma: float,
    target_policy_noise: float,
    target_noise_clip: float,
    key: jax.Array,
) -> tuple[Any, optax.OptState, Dict[str, jax.Array]]:
    """JIT-compiled critic update."""
    # Get current critic params and static
    critic1_params, critic1_static = eqx.partition(model.critic1, eqx.is_inexact_array)
    critic2_params, critic2_static = eqx.partition(model.critic2, eqx.is_inexact_array)

    def critic_loss_fn(critic_params_tuple):
        c1_params, c2_params = critic_params_tuple
        # Reconstruct model with new critic params
        new_critic1 = eqx.combine(c1_params, critic1_static)
        new_critic2 = eqx.combine(c2_params, critic2_static)
        new_model = eqx.tree_at(
            lambda m: (m.critic1, m.critic2),
            model,
            (new_critic1, new_critic2),
        )
        loss, info = compute_critic_loss(
            new_model,
            target_actor_params,
            target_actor_static,
            target_critic1_params,
            target_critic1_static,
            target_critic2_params,
            target_critic2_static,
            batch,
            gamma,
            target_policy_noise,
            target_noise_clip,
            key,
        )
        return loss, info

    (loss, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)((critic1_params, critic2_params))

    updates, new_critic_opt_state = critic_optimizer.update(grads, critic_opt_state, (critic1_params, critic2_params))
    new_critic1_params, new_critic2_params = optax.apply_updates((critic1_params, critic2_params), updates)

    # Reconstruct model with updated critics
    new_critic1 = eqx.combine(new_critic1_params, critic1_static)
    new_critic2 = eqx.combine(new_critic2_params, critic2_static)
    new_model = eqx.tree_at(
        lambda m: (m.critic1, m.critic2),
        model,
        (new_critic1, new_critic2),
    )

    return new_model, new_critic_opt_state, info


@eqx.filter_jit
def update_actor(
    model: TD3ActorCritic,
    actor_opt_state: optax.OptState,
    actor_optimizer: optax.GradientTransformation,
    batch: ReplayBatch,
    key: jax.Array,
) -> tuple[Any, optax.OptState, Dict[str, jax.Array]]:
    """JIT-compiled actor update."""
    # Get actor params
    actor_params, actor_static = eqx.partition(model.actor, eqx.is_inexact_array)

    def actor_loss_fn(a_params):
        # Reconstruct model with new actor params
        new_actor = eqx.combine(a_params, actor_static)
        new_model = eqx.tree_at(
            lambda m: m.actor,
            model,
            new_actor,
        )
        loss, info = compute_actor_loss(new_model, batch, key)
        return loss, info

    (loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params)

    updates, new_actor_opt_state = actor_optimizer.update(grads, actor_opt_state, actor_params)
    new_actor_params = optax.apply_updates(actor_params, updates)

    # Reconstruct model with updated actor
    new_actor = eqx.combine(new_actor_params, actor_static)
    new_model = eqx.tree_at(
        lambda m: m.actor,
        model,
        new_actor,
    )

    return new_model, new_actor_opt_state, info


@eqx.filter_jit
def update_targets(
    model: TD3ActorCritic,
    target_actor_params: Any,
    target_critic1_params: Any,
    target_critic2_params: Any,
    tau: float,
) -> tuple[Any, Any, Any]:
    """JIT-compiled target network update."""
    actor_params, _ = eqx.partition(model.actor, eqx.is_inexact_array)
    critic1_params, _ = eqx.partition(model.critic1, eqx.is_inexact_array)
    critic2_params, _ = eqx.partition(model.critic2, eqx.is_inexact_array)

    new_target_actor = polyak_update(actor_params, target_actor_params, tau)
    new_target_critic1 = polyak_update(critic1_params, target_critic1_params, tau)
    new_target_critic2 = polyak_update(critic2_params, target_critic2_params, tau)

    return new_target_actor, new_target_critic1, new_target_critic2
