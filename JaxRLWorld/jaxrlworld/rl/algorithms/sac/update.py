from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from jaxrlworld.rl.algorithms.base import polyak_update
from jaxrlworld.rl.algorithms.sac.losses import (
    compute_actor_loss,
    compute_alpha_loss,
    compute_critic_loss,
)
from jaxrlworld.rl.modules.policies.sac_ac import SACActorCritic
from jaxrlworld.rl.storages.replay_buffer import ReplayBatch

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


def metric_scalars(
    critic_info: Dict[str, jax.Array],
    actor_loss: jax.Array,
    entropy: jax.Array,
    alpha_loss: jax.Array,
    alpha_value: jax.Array,
    batch: ReplayBatch,
) -> Dict[str, jax.Array]:
    """Every number a metric set needs, still on device.

    Shared by the single update and the scanned loop so the two report
    the same things. The scan carries these nineteen scalars per step
    rather than the batches they came from — stacking a batch per update
    would cost ``num_gradient_steps`` copies of the whole thing for
    numbers only the last step contributes.
    """
    return {
        "critic_loss": critic_info["critic_loss"],
        "critic1_loss": critic_info["critic1_loss"],
        "critic2_loss": critic_info["critic2_loss"],
        "q1_value": critic_info["q1_value"],
        "q2_value": critic_info["q2_value"],
        "current_q1_std": critic_info["current_q1_std"],
        "current_q2_std": critic_info["current_q2_std"],
        "target_q_value": critic_info["target_q_value"],
        "actor_loss": actor_loss,
        "entropy": entropy,
        "alpha_loss": alpha_loss,
        "alpha_value": alpha_value,
        "reward_mean": batch.rewards.mean(),
        "reward_std": batch.rewards.std(),
        "reward_min": batch.rewards.min(),
        "reward_max": batch.rewards.max(),
        "action_mean": batch.actions.mean(),
        "action_std": batch.actions.std(),
        "terminated_ratio": batch.terminated.mean(),
    }


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


@eqx.filter_jit
def update_all(
    model: SACActorCritic,
    target_critic1_params: Any,
    target_critic2_params: Any,
    critic_opt_state: optax.OptState,
    actor_opt_state: optax.OptState,
    alpha_opt_state: optax.OptState,
    log_ent_coef: jax.Array,
    batch: ReplayBatch,
    key: jax.Array,
    critic_optimizer: optax.GradientTransformation,
    actor_optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
    gamma: float,
    tau: float,
    target_entropy: float,
    fixed_ent_coef: float,
    update_actor_now: bool,
    auto_entropy: bool,
) -> tuple[Any, ...]:
    """One SAC step — critics, actor, alpha, targets — in one program.

    Run as four separate compiled functions with Python between them,
    a step costs four launches plus a target-network tree walk, and an
    off-policy algorithm pays that per environment step. At this batch
    size the launches, not the arithmetic, are what the step costs.

    ``update_actor_now`` and ``auto_entropy`` are static, so the delayed
    and undelayed variants compile separately rather than carrying a
    branch; with the default ``policy_delay`` of 1 only one of them is
    ever traced.

    Returns the new state plus the metric scalars, left on device — see
    ``host_scalars`` for why they are not converted here.
    """
    key, critic_key, actor_key = jax.random.split(key, 3)

    ent_coef = jnp.exp(log_ent_coef) if auto_entropy else jnp.asarray(fixed_ent_coef)

    model, critic_opt_state, critic_info = update_critics(
        model,
        target_critic1_params,
        target_critic2_params,
        critic_opt_state,
        batch,
        critic_optimizer,
        gamma,
        ent_coef,
        critic_key,
    )

    if update_actor_now:
        model, actor_opt_state, log_prob, actor_info = update_actor(
            model,
            actor_opt_state,
            actor_optimizer,
            batch,
            ent_coef,
            actor_key,
        )
        actor_loss = actor_info["actor_loss"]
        entropy = actor_info["entropy"]

        if auto_entropy:
            log_ent_coef, alpha_opt_state, alpha_loss, alpha_value = update_alpha(
                log_ent_coef,
                alpha_opt_state,
                alpha_optimizer,
                log_prob,
                target_entropy,
            )
        else:
            alpha_loss = jnp.zeros(())
            alpha_value = ent_coef

        critic1_params, _ = eqx.partition(model.critic1, eqx.is_inexact_array)
        critic2_params, _ = eqx.partition(model.critic2, eqx.is_inexact_array)
        target_critic1_params, target_critic2_params = polyak_update(
            (critic1_params, critic2_params),
            (target_critic1_params, target_critic2_params),
            tau,
        )
    else:
        actor_loss = jnp.zeros(())
        entropy = jnp.zeros(())
        alpha_loss = jnp.zeros(())
        alpha_value = ent_coef

    return (
        model,
        target_critic1_params,
        target_critic2_params,
        critic_opt_state,
        actor_opt_state,
        alpha_opt_state,
        log_ent_coef,
        key,
        critic_info,
        actor_loss,
        entropy,
        alpha_loss,
        alpha_value,
    )


@eqx.filter_jit
def scan_updates(
    state: tuple,
    sampler: Any,
    sample_state: Any,
    num_updates: int,
    critic_optimizer: optax.GradientTransformation,
    actor_optimizer: optax.GradientTransformation,
    alpha_optimizer: optax.GradientTransformation,
    gamma: float,
    tau: float,
    target_entropy: float,
    fixed_ent_coef: float,
    auto_entropy: bool,
) -> tuple:
    """``num_updates`` gradient steps as one program, sampling included.

    ``filter_jit`` keeps every non-array argument static, so
    ``num_updates``, the optimizers, the scalars and the sampler are all
    baked in and only ``state`` is traced.

    ``sampler`` is the buffer's traceable sampler and ``sample_state``
    the arrays it reads (see ``DeviceReplayBuffer.batch_sampler``). They
    are separate because the sampler has to stay static and hashable for
    the compilation cache to hit, while the rings must stay traced.

    The per-step outputs are the metric scalars, stacked; the caller
    takes the last row. Returning the batches instead would stack
    ``num_updates`` copies of them for numbers only one step supplies.
    """

    def body(carry, _):
        model, tc1, tc2, critic_opt, actor_opt, alpha_opt, log_ent_coef, key, sample_key = carry
        sample_key, sub = jax.random.split(sample_key)
        batch = sampler(sample_state, sub)
        (
            model,
            tc1,
            tc2,
            critic_opt,
            actor_opt,
            alpha_opt,
            log_ent_coef,
            key,
            critic_info,
            actor_loss,
            entropy,
            alpha_loss,
            alpha_value,
        ) = update_all(
            model,
            tc1,
            tc2,
            critic_opt,
            actor_opt,
            alpha_opt,
            log_ent_coef,
            batch,
            key,
            critic_optimizer,
            actor_optimizer,
            alpha_optimizer,
            gamma,
            tau,
            target_entropy,
            fixed_ent_coef,
            True,
            auto_entropy,
        )
        carry = (model, tc1, tc2, critic_opt, actor_opt, alpha_opt, log_ent_coef, key, sample_key)
        return carry, metric_scalars(critic_info, actor_loss, entropy, alpha_loss, alpha_value, batch)

    return jax.lax.scan(body, state, None, length=num_updates)
