from functools import partial
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from rlworld.rl.storages.rollout_storage import RolloutBatch

from .losses import (
    compute_dr3_regularizer,
    compute_feature_similarity_metrics,
    compute_policy_loss,
    compute_value_loss,
)

# ==================== Data Structures ====================


class PPODR3LossInfo(NamedTuple):
    """Loss components for logging."""

    policy_loss: jax.Array
    value_loss: jax.Array
    dr3_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array
    feature_dot_product: jax.Array
    feature_cosine_similarity: jax.Array
    feature_norm: jax.Array
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
    dr3_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array
    feature_dot_product: jax.Array
    feature_cosine_similarity: jax.Array
    feature_norm: jax.Array
    did_update: jax.Array
    aux: dict


# ==================== Batch Loss Computation ====================


def compute_batch_loss_dr3(
    params: Any,
    static: Any,
    batch: RolloutBatch,
    clip_param: float,
    value_loss_coef: float,
    entropy_coef: float,
    dr3_coef: float,
    use_clipped_value_loss: bool,
    normalize_advantages: bool,
    key: jax.Array,
) -> tuple[jax.Array, PPODR3LossInfo]:
    """
    Compute loss for a single batch with DR3 regularization.

    DR3 is applied to critic features between current and next states.
    Next states are obtained by shifting observations within the batch.
    """
    model = eqx.combine(params, static)

    advantages = batch.advantages
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Actor forward pass
    log_probs, entropy, _mu_new, _sigma_new, actor_aux = model.evaluate_actions(
        batch.actor_observations, batch.actions, key=key
    )

    # Critic forward pass with features
    values, features, critic_aux = model.critic.forward_with_features(batch.critic_observations)
    values = values.squeeze(-1)

    # Compute next state features for DR3
    # Shift observations: next_obs[i] = obs[i+1], last one is dropped
    next_critic_obs = batch.critic_observations[1:]  # [batch-1, obs_dim]
    current_features = features[:-1]  # [batch-1, feature_dim]

    _, next_features, _ = model.critic.forward_with_features(next_critic_obs)

    # Policy loss
    policy_loss, approx_kl, clip_fraction = compute_policy_loss(
        log_probs=log_probs,
        old_log_probs=batch.old_log_probs,
        advantages=advantages,
        clip_param=clip_param,
    )

    # Value loss
    value_loss = compute_value_loss(
        values=values,
        old_values=batch.values,
        returns=batch.returns,
        clip_param=clip_param,
        use_clipped=use_clipped_value_loss,
    )

    # DR3 regularizer
    dr3_loss = compute_dr3_regularizer(current_features, next_features)

    # Compute similarity metrics for logging
    similarity_metrics = compute_feature_similarity_metrics(current_features, next_features)

    # Entropy
    entropy_mean = entropy.mean()

    # Total loss
    total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_mean + dr3_coef * dr3_loss

    aux = {**actor_aux, **critic_aux}

    loss_info = PPODR3LossInfo(
        policy_loss=policy_loss,
        value_loss=value_loss,
        dr3_loss=dr3_loss,
        entropy=entropy_mean,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        feature_dot_product=similarity_metrics["dot_product"],
        feature_cosine_similarity=similarity_metrics["cosine_similarity"],
        feature_norm=similarity_metrics["feature_norm"],
        aux=aux,
    )

    return total_loss, loss_info


# ==================== Main Update Function ====================


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 10, 11))
def update_all_batches_dr3(
    params: Any,
    static: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    clip_param: float,
    value_loss_coef: float,
    entropy_coef: float,
    dr3_coef: float,
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
        dr3_coef: DR3 regularization coefficient (static)
        use_clipped_value_loss: Whether to clip value loss (static)
        normalize_advantages: Whether to normalize advantages (static)
        use_early_stop: Whether to use KL-based early stopping (static)
        desired_kl: Target KL for early stopping (static)
        batches: Stacked batches (num_batches, batch_size, ...)
        key: JAX random key

    Returns:
        Updated params, opt_state, and aggregated outputs
    """

    def scan_fn(carry: ScanCarry, batch: RolloutBatch) -> tuple[ScanCarry, ScanOutput]:
        params, opt_state, key, early_stopped = (carry.params, carry.opt_state, carry.key, carry.early_stopped)
        key, subkey = jax.random.split(key)

        def loss_fn(p):
            return compute_batch_loss_dr3(
                p,
                static,
                batch,
                clip_param,
                value_loss_coef,
                entropy_coef,
                dr3_coef,
                use_clipped_value_loss,
                normalize_advantages,
                subkey,
            )

        # Compute loss and gradients
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Check early stop condition
        should_stop = use_early_stop & (loss_info.approx_kl > desired_kl)
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
            dr3_loss=loss_info.dr3_loss,
            entropy=loss_info.entropy,
            approx_kl=loss_info.approx_kl,
            clip_fraction=loss_info.clip_fraction,
            feature_dot_product=loss_info.feature_dot_product,
            feature_cosine_similarity=loss_info.feature_cosine_similarity,
            feature_norm=loss_info.feature_norm,
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
