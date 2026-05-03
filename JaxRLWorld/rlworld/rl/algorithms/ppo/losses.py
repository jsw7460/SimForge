import jax
import jax.numpy as jnp


def compute_analytical_kl(
    mu_new: jax.Array,
    sigma_new: jax.Array,
    mu_old: jax.Array,
    sigma_old: jax.Array,
) -> jax.Array:
    """Closed-form KL(N(mu_old, sigma_old) || N(mu_new, sigma_new)) per sample, mean over batch.

    Matches rsl_rl PPO's adaptive-LR criterion. Lower variance than Schulman's
    approx-KL, so it is the right signal for the schedule heuristic.
    """
    var_new = jnp.square(sigma_new)
    var_old = jnp.square(sigma_old)
    kl = jnp.log(sigma_new / (sigma_old + 1e-5) + 1e-5) \
        + (var_old + jnp.square(mu_old - mu_new)) / (2.0 * var_new) \
        - 0.5
    return kl.sum(axis=-1).mean()


def compute_policy_loss(
    log_probs: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    clip_param: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute PPO clipped surrogate loss.

    Args:
        log_probs: Current policy log probabilities
        old_log_probs: Old policy log probabilities
        advantages: Advantage estimates
        clip_param: PPO clipping parameter

    Returns:
        policy_loss: Clipped surrogate loss
        approx_kl: Approximate KL divergence
        clip_fraction: Fraction of clipped ratios
    """
    log_ratio = log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)

    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * jnp.clip(ratio, 1.0 - clip_param, 1.0 + clip_param)
    policy_loss = jnp.maximum(surrogate, surrogate_clipped).mean()

    approx_kl = ((ratio - 1) - log_ratio).mean()
    clip_fraction = (jnp.abs(ratio - 1.0) > clip_param).astype(jnp.float32).mean()

    return policy_loss, approx_kl, clip_fraction


def compute_value_loss(
    values: jax.Array,
    old_values: jax.Array,
    returns: jax.Array,
    clip_param: float,
    use_clipped: bool = True,
) -> jax.Array:
    """
    Compute value function loss with optional clipping.

    Args:
        values: Current value estimates
        old_values: Old value estimates
        returns: Target returns
        clip_param: Value clipping parameter
        use_clipped: Whether to use clipped value loss

    Returns:
        Value loss
    """
    if use_clipped:
        values_clipped = old_values + jnp.clip(values - old_values, -clip_param, clip_param)
        value_losses = (values - returns) ** 2
        value_losses_clipped = (values_clipped - returns) ** 2
        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()
    else:
        value_loss = ((returns - values) ** 2).mean()
    return value_loss