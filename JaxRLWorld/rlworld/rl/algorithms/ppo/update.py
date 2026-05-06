from functools import partial
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from rlworld.rl.algorithms.ppo.losses import (
    compute_analytical_kl,
    compute_policy_loss,
    compute_value_loss,
)
from rlworld.rl.modules.policies.ppo_ac import PPOActorCritic
from rlworld.rl.storages.rollout_storage import RolloutBatch

# ==================== Data Structures ====================


class PPOLossInfo(NamedTuple):
    """Loss components for logging."""

    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    analytical_kl: jax.Array
    clip_fraction: jax.Array
    aux: dict


class ScanCarry(NamedTuple):
    """Carry state for scan loop."""

    params: Any
    key: jax.Array
    opt_state: optax.OptState
    early_stopped: jax.Array


class ScanOutput(NamedTuple):
    """Output from each scan iteration."""

    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    analytical_kl: jax.Array
    clip_fraction: jax.Array
    did_update: jax.Array
    aux: dict


# ==================== Forward Functions ====================


@eqx.filter_jit
def forward_policy_and_value(
    model: PPOActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict]:
    """Combined forward pass for actor and critic (stochastic).

    Returns:
        env_actions: Actions to send to environment (squashed if applicable)
        raw_actions: Actions to store for PPO update (pre-tanh if squashed)
        mean, std, log_prob, values, aux
    """
    key, subkey = jax.random.split(key)
    dist, actor_aux = model.get_distribution(actor_obs, key=subkey)

    if dist.is_squashed:
        # Brax-style: store raw (pre-tanh) actions, use for log_prob
        raw_actions = dist.sample_raw(key)
        env_actions = jnp.tanh(raw_actions)
        log_prob = dist.log_prob_raw(raw_actions)
    else:
        raw_actions = dist.sample(key)
        env_actions = raw_actions
        log_prob = dist.log_prob(raw_actions)

    values, critic_aux = model.evaluate_value(critic_obs)
    values = values.squeeze(-1)

    aux = {**actor_aux, **critic_aux}
    return env_actions, raw_actions, dist.mean, dist.std, log_prob, values, aux


@eqx.filter_jit
def forward_policy_and_value_deterministic(
    model: PPOActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict]:
    """Combined forward pass for actor and critic (deterministic)."""
    dist, actor_aux = model.get_distribution(actor_obs, key=key)

    if dist.is_squashed:
        raw_actions = dist.mean  # pre-tanh mean
        env_actions = jnp.tanh(raw_actions)
        log_prob = dist.log_prob_raw(raw_actions)
    else:
        raw_actions = dist.mean
        env_actions = raw_actions
        log_prob = dist.log_prob(raw_actions)

    values, critic_aux = model.evaluate_value(critic_obs)
    values = values.squeeze(-1)

    aux = {**actor_aux, **critic_aux}
    return env_actions, raw_actions, dist.mean, dist.std, log_prob, values, aux


@eqx.filter_jit
def get_value(model: PPOActorCritic, critic_obs: jax.Array) -> jax.Array:
    """JIT-compiled value estimation."""
    value, _ = model.evaluate_value(critic_obs)
    return value


# ==================== Batch Loss Computation ====================


def compute_batch_loss(
    params: Any,
    static: Any,
    batch: RolloutBatch,
    clip_param: float,
    value_loss_coef: float,
    entropy_coef: float,
    use_clipped_value_loss: bool,
    normalize_advantages: bool,
    key: jax.Array,
) -> tuple[jax.Array, PPOLossInfo]:
    """Compute loss for a single batch."""
    model = eqx.combine(params, static)

    advantages = batch.advantages
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_probs, entropy, mu_new, sigma_new, actor_aux = model.evaluate_actions(
        batch.actor_observations, batch.actions, key=key
    )
    values, critic_aux = model.evaluate_value(batch.critic_observations)
    values = values.squeeze(-1)

    policy_loss, approx_kl, clip_fraction = compute_policy_loss(
        log_probs=log_probs,
        old_log_probs=batch.old_log_probs,
        advantages=advantages,
        clip_param=clip_param,
    )

    # Closed-form KL on the base Gaussian — used by the adaptive-LR schedule.
    # Lower-variance signal than approx_kl; matches rsl_rl PPO.
    analytical_kl = compute_analytical_kl(
        mu_new=mu_new,
        sigma_new=sigma_new,
        mu_old=batch.old_mu,
        sigma_old=batch.old_sigma,
    )

    value_loss = compute_value_loss(
        values=values,
        old_values=batch.values,
        returns=batch.returns,
        clip_param=clip_param,
        use_clipped=use_clipped_value_loss,
    )

    entropy_mean = entropy.mean()
    total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_mean

    aux = {**actor_aux, **critic_aux}

    loss_info = PPOLossInfo(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        analytical_kl=analytical_kl,
        clip_fraction=clip_fraction,
        aux=aux,
    )

    return total_loss, loss_info


# ==================== Main Update Function ====================


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10))
def update_all_batches(
    params: Any,
    static: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    clip_param: float,
    value_loss_coef: float,
    entropy_coef: float,
    use_clipped_value_loss: bool,
    normalize_advantages: bool,
    use_early_stop: bool,
    desired_kl: float,
    batches: RolloutBatch,
    key: jax.Array,
) -> tuple[Any, optax.OptState, ScanOutput, jax.Array]:
    """
    Update over all batches using jax.lax.scan with early stopping support.

    Args:
        params: Model parameters (pytree of arrays)
        static: Model static parts
        opt_state: Optimizer state
        optimizer: Optax optimizer (static)
        clip_param: PPO clip parameter (static)
        value_loss_coef: Value loss coefficient (static)
        entropy_coef: Entropy coefficient (static)
        use_clipped_value_loss: Whether to clip value loss (static)
        normalize_advantages: Whether to normalize advantages (static)
        use_early_stop: Whether to use KL-based early stopping (static)
        desired_kl: Target KL for early stopping (static, used as threshold)
        batches: Stacked batches (num_batches, batch_size, ...)
        key: JAX random key

    Returns:
        Updated params, opt_state, and aggregated outputs
    """

    def scan_fn(carry: ScanCarry, batch: RolloutBatch) -> tuple[ScanCarry, ScanOutput]:
        params, opt_state, key, early_stopped = (
            carry.params,
            carry.opt_state,
            carry.key,
            carry.early_stopped,
        )
        key, subkey = jax.random.split(key)

        def loss_fn(p):
            return compute_batch_loss(
                p,
                static,
                batch,
                clip_param,
                value_loss_coef,
                entropy_coef,
                use_clipped_value_loss,
                normalize_advantages,
                subkey,
            )

        # Compute loss and gradients
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Check early stop condition.
        # Threshold follows the standard PPO recipe (Spinning Up / OpenAI):
        # stop once the sample-based KL drifts past 1.5 x the target.
        should_stop = use_early_stop & (loss_info.approx_kl > 1.5 * desired_kl)
        do_update = ~early_stopped & ~should_stop

        # Conditionally apply update
        def apply_update(_):
            updates, new_opt = optimizer.update(grads, opt_state, params)
            new_p = optax.apply_updates(params, updates)
            return new_p, new_opt

        def skip_update(_):
            return params, opt_state

        new_params, new_opt_state = jax.lax.cond(
            do_update,
            apply_update,
            skip_update,
            operand=None,
        )

        # Update early_stopped flag
        new_early_stopped = early_stopped | should_stop

        new_carry = ScanCarry(
            params=new_params,
            opt_state=new_opt_state,
            key=key,
            early_stopped=new_early_stopped,
        )
        output = ScanOutput(
            policy_loss=loss_info.policy_loss,
            value_loss=loss_info.value_loss,
            entropy=loss_info.entropy,
            approx_kl=loss_info.approx_kl,
            analytical_kl=loss_info.analytical_kl,
            clip_fraction=loss_info.clip_fraction,
            did_update=do_update,
            aux=loss_info.aux,
        )

        return new_carry, output

    init_carry = ScanCarry(
        params=params,
        opt_state=opt_state,
        key=key,
        early_stopped=jnp.array(False),
    )
    final_carry, outputs = jax.lax.scan(scan_fn, init_carry, batches)

    return final_carry.params, final_carry.opt_state, outputs, final_carry.key
