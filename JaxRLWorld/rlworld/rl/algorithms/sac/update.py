from typing import Any, Dict

import equinox as eqx
import jax
import optax

from rlworld.rl.algorithms.sac.losses import (
    compute_actor_loss,
    compute_alpha_loss,
    compute_critic_loss,
)
from rlworld.rl.modules.policies.sac_ac import SACActorCritic
from rlworld.rl.storages.replay_buffer import ReplayBatch

# ==================== Forward Functions ====================


@eqx.filter_jit
def act_stochastic(
    model: SACActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compiled stochastic action selection."""
    key, subkey = jax.random.split(key)
    actions, _ = model.act(actor_obs, key=subkey, deterministic=False)
    values = model.evaluate(actor_obs, critic_obs, key=key)
    return actions, values, key


@eqx.filter_jit
def act_deterministic(
    model: SACActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JIT-compiled deterministic action selection."""
    key, subkey = jax.random.split(key)
    actions, _ = model.act(actor_obs, key=subkey, deterministic=True)
    values = model.evaluate(actor_obs, critic_obs, key=key)
    return actions, values, key


@eqx.filter_jit
def get_value(
    model: SACActorCritic,
    critic_obs: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """JIT-compiled value estimation."""
    return model.evaluate(critic_obs, key=key)


# ==================== Update Functions ====================


@eqx.filter_jit
def update_critics(
    model: SACActorCritic,
    target_critic1_params: Any,
    target_critic2_params: Any,
    critic_opt_state: optax.OptState,
    batch: ReplayBatch,
    critic_optimizer: optax.GradientTransformation,
    gamma: float,
    ent_coef: jax.Array,
    key: jax.Array,
) -> tuple[Any, optax.OptState, Dict[str, jax.Array]]:
    """JIT-compiled critic update."""
    # Get static parts of critics
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
            target_critic1_params,
            target_critic2_params,
            critic1_static,
            critic2_static,
            batch,
            gamma,
            ent_coef,
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
    model: SACActorCritic,
    actor_opt_state: optax.OptState,
    actor_optimizer: optax.GradientTransformation,
    batch: ReplayBatch,
    ent_coef: jax.Array,
    key: jax.Array,
) -> tuple[Any, optax.OptState, jax.Array, Dict[str, jax.Array]]:
    """JIT-compiled actor update."""
    # Get actor and log_std_net params
    actor_params, actor_static = eqx.partition(model.actor, eqx.is_inexact_array)
    log_std_params, log_std_static = eqx.partition(model.log_std_net, eqx.is_inexact_array)

    def actor_loss_fn(params_tuple):
        a_params, ls_params = params_tuple
        # Reconstruct model with new actor params
        new_actor = eqx.combine(a_params, actor_static)
        new_log_std = eqx.combine(ls_params, log_std_static)
        new_model = eqx.tree_at(
            lambda m: (m.actor, m.log_std_net),
            model,
            (new_actor, new_log_std),
        )
        loss, log_prob, info = compute_actor_loss(new_model, batch, ent_coef, key)
        return loss, (log_prob, info)

    (loss, (log_prob, info)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)((actor_params, log_std_params))

    updates, new_actor_opt_state = actor_optimizer.update(grads, actor_opt_state, (actor_params, log_std_params))
    new_actor_params, new_log_std_params = optax.apply_updates((actor_params, log_std_params), updates)

    # Reconstruct model with updated actor
    new_actor = eqx.combine(new_actor_params, actor_static)
    new_log_std = eqx.combine(new_log_std_params, log_std_static)
    new_model = eqx.tree_at(
        lambda m: (m.actor, m.log_std_net),
        model,
        (new_actor, new_log_std),
    )

    return new_model, new_actor_opt_state, log_prob, info


@eqx.filter_jit
def update_alpha(
    log_ent_coef: jax.Array,
    alpha_opt_state: optax.OptState,
    alpha_optimizer: optax.GradientTransformation,
    log_prob: jax.Array,
    target_entropy: float,
) -> tuple[jax.Array, optax.OptState, jax.Array, jax.Array]:
    """JIT-compiled alpha update."""

    def alpha_loss_fn(log_alpha):
        loss, alpha = compute_alpha_loss(log_alpha, log_prob, target_entropy)
        return loss, alpha

    (loss, alpha), grad = jax.value_and_grad(alpha_loss_fn, has_aux=True)(log_ent_coef)

    updates, new_alpha_opt_state = alpha_optimizer.update(grad, alpha_opt_state, log_ent_coef)
    new_log_ent_coef = optax.apply_updates(log_ent_coef, updates)

    return new_log_ent_coef, new_alpha_opt_state, loss, alpha
