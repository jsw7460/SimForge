from functools import partial
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from jaxrlworld.rl.algorithms.ppo.losses import (
    compute_analytical_kl,
    compute_policy_loss,
    compute_value_loss,
)
from jaxrlworld.rl.algorithms.ppo.symmetry import symmetry_mirror_loss
from jaxrlworld.rl.modules.policies.ppo_ac import PPOActorCritic
from jaxrlworld.rl.storages.rollout_storage import RolloutBatch, compute_gae

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
    symmetry_spec=None,
    symmetry_coef: float = 0.0,
    bound_loss_coef: float = 0.0,
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

    # Action-bound penalty on the (unsquashed) policy mean outside [-1, 1]
    # (booster_gym's bound loss). Static coefficient — no cost when 0.
    bound_loss = jnp.zeros(())
    if bound_loss_coef > 0.0:
        bound_loss = (
            jnp.square(jnp.maximum(mu_new - 1.0, 0.0)).mean() + jnp.square(jnp.minimum(mu_new + 1.0, 0.0)).mean()
        )
        total_loss = total_loss + bound_loss_coef * bound_loss

    # Left/right symmetry (mirror) auxiliary loss. symmetry_coef is a Python
    # static scalar, so this branch is resolved at trace time (no cost when off).
    mirror_loss = jnp.zeros(())
    if symmetry_coef > 0.0:
        key, mkey = jax.random.split(key)
        mirror_loss = symmetry_mirror_loss(model, batch.actor_observations, symmetry_spec, mkey)
        total_loss = total_loss + symmetry_coef * mirror_loss

    aux = {**actor_aux, **critic_aux, "mirror_loss": mirror_loss, "bound_loss": bound_loss}

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


# ``opt_state`` is donated: it is consumed and immediately replaced by the
# caller, so XLA can write the new Adam moments into the old buffers
# instead of allocating 2x parameter bytes fresh every iteration.
# ``params`` is NOT donated — on the first update its leaves are the
# initial model's arrays, which the runner's ``actor_critic`` reference
# still shares; donating them would invalidate that object's buffers.
@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10, 12, 13), donate_argnums=(2,))
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
    symmetry_spec: Any,
    symmetry_coef: float,
    bound_loss_coef: float,
    flat_batch: RolloutBatch,
    batch_indices: jax.Array,
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
        flat_batch: The whole rollout, flat: ``[T*N, ...]`` per field
        batch_indices: Shuffled minibatch rows into ``flat_batch``,
            ``(num_batches, minibatch_size)`` — each scan step gathers
            just its own minibatch, so the shuffled epochs are never
            materialized together (the old pre-gathered layout held
            ``num_epochs`` full rollout copies on device)
        key: JAX random key

    Returns:
        Updated params, opt_state, and aggregated outputs
    """

    def scan_fn(carry: ScanCarry, idx: jax.Array) -> tuple[ScanCarry, ScanOutput]:
        params, opt_state, key, early_stopped = (
            carry.params,
            carry.opt_state,
            carry.key,
            carry.early_stopped,
        )
        key, subkey = jax.random.split(key)

        # Gather this step's minibatch from the flat rollout (dict obs
        # groups included via the tree map).
        batch = jax.tree.map(lambda x: x[idx], flat_batch)

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
                symmetry_spec,
                symmetry_coef,
                bound_loss_coef,
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
    final_carry, outputs = jax.lax.scan(scan_fn, init_carry, batch_indices)

    return final_carry.params, final_carry.opt_state, outputs, final_carry.key


# ==================== Full-batch update with per-epoch GAE ====================


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 10, 11, 12, 13), donate_argnums=(2,))
def update_recompute_gae_epochs(
    params: Any,
    static: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    clip_param: float,
    value_loss_coef: float,
    entropy_coef: float,
    use_clipped_value_loss: bool,
    bound_loss_coef: float,
    symmetry_spec: Any,
    symmetry_coef: float,
    num_epochs: int,
    gamma: float,
    gae_lambda: float,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    old_mu: jax.Array,
    old_sigma: jax.Array,
    rewards: jax.Array,
    episode_starts: jax.Array,
    trunc_mask: jax.Array,
    last_critic_obs: jax.Array,
    last_dones: jax.Array,
    key: jax.Array,
) -> tuple[Any, optax.OptState, ScanOutput, jax.Array, jax.Array]:
    """booster_gym-style update: ``num_epochs`` FULL-batch gradient steps,
    each recomputing values, GAE and advantage normalization with the
    CURRENT critic before taking one step.

    Per epoch (matching booster_gym's ``Runner.train`` inner loop):

      1. ``values = V(critic_obs)`` fresh (also ``V(last_critic_obs)``)
      2. ``rewards[trunc_mask] = values[trunc_mask]`` — truncation steps
         (time-outs and command resamples without reset) get their reward
         replaced by the value estimate, which makes their TD error zero
      3. GAE with the episode cut extended by ``trunc_mask``
      4. ``returns = values + advantages`` (constant target for the MSE)
      5. one full-batch gradient step (advantages normalized over the
         whole batch inside :func:`compute_batch_loss`)

    Shapes: rollout tensors are ``(T, N, ...)``; obs must be plain arrays
    (no image groups — the PPO wrapper enforces that).

    Returns ``(params, opt_state, outputs, key, last_epoch_returns_flat)``;
    the extra returns tensor feeds the batch-statistics metrics.
    """
    T, N = rewards.shape
    flat_actor = actor_obs.reshape((T * N,) + actor_obs.shape[2:])
    flat_critic = critic_obs.reshape((T * N,) + critic_obs.shape[2:])
    flat_actions = actions.reshape((T * N,) + actions.shape[2:])
    flat_log_probs = old_log_probs.reshape(T * N)
    flat_mu = old_mu.reshape((T * N,) + old_mu.shape[2:])
    flat_sigma = old_sigma.reshape((T * N,) + old_sigma.shape[2:])

    # The truncation cut, folded into the episode-start convention
    # compute_gae consumes: a truncation at t starts a "new episode" at t+1.
    episode_starts_gae = episode_starts.at[1:].set(episode_starts[1:] | trunc_mask[:-1])
    last_dones_gae = last_dones | trunc_mask[-1]

    def epoch_fn(carry: ScanCarry, _x) -> tuple[ScanCarry, tuple[ScanOutput, jax.Array]]:
        params, opt_state, key = carry.params, carry.opt_state, carry.key
        key, subkey = jax.random.split(key)

        model = eqx.combine(params, static)
        values_flat, _ = model.evaluate_value(flat_critic)
        values = values_flat.squeeze(-1).reshape(T, N)
        last_values, _ = model.evaluate_value(last_critic_obs)
        last_values = last_values.squeeze(-1)

        rewards_eff = jnp.where(trunc_mask, values, rewards)
        advantages, returns = compute_gae(
            rewards=rewards_eff,
            values=values,
            episode_starts=episode_starts_gae,
            last_values=last_values,
            last_dones=last_dones_gae,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        batch = RolloutBatch(
            actor_observations=flat_actor,
            critic_observations=flat_critic,
            actions=flat_actions,
            values=values.reshape(T * N),
            advantages=advantages.reshape(T * N),
            returns=returns.reshape(T * N),
            old_log_probs=flat_log_probs,
            old_mu=flat_mu,
            old_sigma=flat_sigma,
        )

        def loss_fn(p):
            return compute_batch_loss(
                p,
                static,
                batch,
                clip_param,
                value_loss_coef,
                entropy_coef,
                use_clipped_value_loss,
                True,  # normalize_advantages: full-batch, per epoch
                subkey,
                symmetry_spec,
                symmetry_coef,
                bound_loss_coef,
            )

        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_carry = ScanCarry(
            params=new_params,
            opt_state=new_opt_state,
            key=key,
            early_stopped=carry.early_stopped,
        )
        output = ScanOutput(
            policy_loss=loss_info.policy_loss,
            value_loss=loss_info.value_loss,
            entropy=loss_info.entropy,
            approx_kl=loss_info.approx_kl,
            analytical_kl=loss_info.analytical_kl,
            clip_fraction=loss_info.clip_fraction,
            did_update=jnp.array(True),
            aux=loss_info.aux,
        )
        return new_carry, (output, returns.reshape(T * N))

    init_carry = ScanCarry(
        params=params,
        opt_state=opt_state,
        key=key,
        early_stopped=jnp.array(False),
    )
    final_carry, (outputs, returns_per_epoch) = jax.lax.scan(epoch_fn, init_carry, None, length=num_epochs)

    return final_carry.params, final_carry.opt_state, outputs, final_carry.key, returns_per_epoch[-1]
