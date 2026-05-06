import jax
import jax.numpy as jnp


def compute_dr3_regularizer(
    features: jax.Array,
    next_features: jax.Array,
) -> jax.Array:
    """
    Compute DR3 feature regularizer.

    DR3 regularizes the dot product between features at consecutive states
    to prevent feature co-adaptation in TD learning.

    R_DR3 = mean(phi(s)^T @ phi(s'))

    Args:
        features: Current state features [batch_size, feature_dim]
        next_features: Next state features [batch_size, feature_dim]

    Returns:
        Scalar DR3 regularization loss
    """
    # Dot product per sample: sum over feature dimension
    dot_products = jnp.sum(features * next_features, axis=-1)

    # Mean over batch
    return dot_products.mean()


def compute_feature_similarity_metrics(
    features: jax.Array,
    next_features: jax.Array,
) -> dict[str, jax.Array]:
    """
    Compute feature similarity metrics for logging.

    Args:
        features: Current state features [batch_size, feature_dim]
        next_features: Next state features [batch_size, feature_dim]

    Returns:
        Dictionary containing:
        - dot_product: Mean dot product (same as DR3 loss)
        - cosine_similarity: Mean cosine similarity
        - feature_norm: Mean feature norm
        - next_feature_norm: Mean next feature norm
    """
    # Dot product
    dot_products = jnp.sum(features * next_features, axis=-1)

    # Norms
    feature_norms = jnp.linalg.norm(features, axis=-1)
    next_feature_norms = jnp.linalg.norm(next_features, axis=-1)

    # Cosine similarity: dot / (norm1 * norm2)
    cosine_sim = dot_products / (feature_norms * next_feature_norms + 1e-8)

    return {
        "dot_product": dot_products.mean(),
        "cosine_similarity": cosine_sim.mean(),
        "feature_norm": feature_norms.mean(),
        "next_feature_norm": next_feature_norms.mean(),
    }


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
