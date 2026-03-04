from typing import Any, Dict

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from rlworld.rl.modules.policies.fast_td3_ac import FastTD3ActorCritic
from rlworld.rl.storages.replay_buffer import ReplayBatch
from rlworld.rl.algorithms.base import polyak_update
from rlworld.rl.algorithms.fast_td3.losses import (
    compute_critic_loss,
    compute_actor_loss,
)


# ==================== Forward Functions ====================


@eqx.filter_jit
def act_deterministic(
    model: FastTD3ActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    JIT-compiled deterministic action selection.

    NOTE: Observation normalization is handled inside model.act() and
    model.evaluate() via _normalize_*_obs() methods.
    """
    key, subkey = jax.random.split(key)
    actions, _ = model.act(actor_obs, key=subkey)
    values = model.evaluate(actor_obs, critic_obs, key=key)
    return actions, values, key


@eqx.filter_jit
def act_with_noise(
    model: FastTD3ActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    noise_scales: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    JIT-compiled action selection with per-environment exploration noise.

    NOTE: Observation normalization is handled inside model.act() and
    model.evaluate() via _normalize_*_obs() methods.

    Args:
        model: FastTD3 model
        actor_obs: Actor observations [num_envs, obs_dim]
        critic_obs: Critic observations [num_envs, critic_obs_dim]
        noise_scales: Per-environment noise scales [num_envs, 1]
        key: JAX random key

    Returns:
        Tuple of (noisy_actions, values, new_key)
    """
    key, action_key, noise_key = jax.random.split(key, 3)

    # Get deterministic action (normalization handled inside model.act())
    actions, _ = model.act(actor_obs, key=action_key)

    # Add per-environment exploration noise
    noise = jax.random.normal(noise_key, actions.shape) * noise_scales
    actions = jnp.clip(actions + noise, -1.0, 1.0)

    values = model.evaluate(actor_obs, critic_obs, key=key)
    return actions, values, key


@eqx.filter_jit
def get_value(
    model: FastTD3ActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """JIT-compiled value estimation."""
    return model.evaluate(actor_obs, critic_obs, key=key)


# ==================== Update Functions ====================


@eqx.filter_jit
def update_critics(
    model: FastTD3ActorCritic,
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
    use_cdq: bool,
    use_target_actor: bool,
    key: jax.Array,
) -> tuple[Any, optax.OptState, Dict[str, jax.Array]]:
    """
    JIT-compiled distributional critic update.

    NOTE: Observation normalization is handled inside compute_critic_loss
    via model._normalize_*_obs() methods.
    """

    critic1_params, critic1_static = eqx.partition(model.critic1, eqx.is_inexact_array)
    critic2_params, critic2_static = eqx.partition(model.critic2, eqx.is_inexact_array)

    def critic_loss_fn(critic_params_tuple):
        c1_params, c2_params = critic_params_tuple
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
            use_cdq,
            use_target_actor,
            key,
        )
        return loss, info

    (loss, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        (critic1_params, critic2_params)
    )

    updates, new_critic_opt_state = critic_optimizer.update(
        grads, critic_opt_state, (critic1_params, critic2_params)
    )
    new_critic1_params, new_critic2_params = optax.apply_updates(
        (critic1_params, critic2_params), updates
    )

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
    model: FastTD3ActorCritic,
    actor_opt_state: optax.OptState,
    actor_optimizer: optax.GradientTransformation,
    batch: ReplayBatch,
    use_cdq: bool,
    key: jax.Array,
) -> tuple[Any, optax.OptState, Dict[str, jax.Array]]:
    """
    JIT-compiled actor update.

    NOTE: Observation normalization is handled inside compute_actor_loss
    via model.act() and model.critic*_q_value() methods.
    """

    actor_params, actor_static = eqx.partition(model.actor, eqx.is_inexact_array)

    def actor_loss_fn(a_params):
        new_actor = eqx.combine(a_params, actor_static)
        new_model = eqx.tree_at(lambda m: m.actor, model, new_actor)
        loss, info = compute_actor_loss(
            new_model,
            batch,
            use_cdq,
            key,
        )
        return loss, info

    (loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params)

    updates, new_actor_opt_state = actor_optimizer.update(
        grads, actor_opt_state, actor_params
    )
    new_actor_params = optax.apply_updates(actor_params, updates)

    new_actor = eqx.combine(new_actor_params, actor_static)
    new_model = eqx.tree_at(lambda m: m.actor, model, new_actor)

    return new_model, new_actor_opt_state, info


@eqx.filter_jit
def update_targets(
    model: FastTD3ActorCritic,
    target_actor_params: Any,
    target_critic1_params: Any,
    target_critic2_params: Any,
    tau: float,
    use_target_actor: bool = False,
) -> tuple[Any, Any, Any]:
    """JIT-compiled target network update with Polyak averaging."""
    critic1_params, _ = eqx.partition(model.critic1, eqx.is_inexact_array)
    critic2_params, _ = eqx.partition(model.critic2, eqx.is_inexact_array)

    if use_target_actor:
        actor_params, _ = eqx.partition(model.actor, eqx.is_inexact_array)
        new_target_actor = polyak_update(actor_params, target_actor_params, tau)
    else:
        new_target_actor = target_actor_params

    new_target_critic1 = polyak_update(critic1_params, target_critic1_params, tau)
    new_target_critic2 = polyak_update(critic2_params, target_critic2_params, tau)

    return new_target_actor, new_target_critic1, new_target_critic2


# ==================== Noise Management ====================


def init_noise_scales(
    num_envs: int,
    noise_min: float,
    noise_max: float,
    key: jax.Array,
) -> jax.Array:
    """
    Initialize per-environment noise scales.

    Args:
        num_envs: Number of parallel environments
        noise_min: Minimum noise scale
        noise_max: Maximum noise scale
        key: JAX random key

    Returns:
        Noise scales [num_envs, 1]
    """
    return jax.random.uniform(key, (num_envs, 1), minval=noise_min, maxval=noise_max)


@jax.jit
def resample_noise_on_done(
    noise_scales: jax.Array,
    dones: jax.Array,
    noise_min: float,
    noise_max: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Resample noise scales for environments that are done.

    Args:
        noise_scales: Current noise scales [num_envs, 1]
        dones: Done flags [num_envs,]
        noise_min: Minimum noise scale
        noise_max: Maximum noise scale
        key: JAX random key

    Returns:
        Tuple of (new_noise_scales, new_key)
    """
    key, subkey = jax.random.split(key)
    new_scales = jax.random.uniform(
        subkey, noise_scales.shape, minval=noise_min, maxval=noise_max
    )
    # Only update noise for done environments
    dones_expanded = dones[:, None] if dones.ndim == 1 else dones
    updated_scales = jnp.where(dones_expanded > 0, new_scales, noise_scales)
    return updated_scales, key