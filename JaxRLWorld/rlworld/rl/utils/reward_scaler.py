import jax.numpy as jnp


class RewardScaler:
    """Discount-based reward scaling (Engstrom et al., "Implementation Matters
    in Deep Policy Gradients").

    Maintains the running variance of the discounted-return estimate
        R_t = gamma * R_{t-1} + r_t
    using the parallel-batch form of Welford's online algorithm, then divides
    each reward by ``sqrt(running_var)``. The mean is intentionally NOT
    subtracted so reward sign is preserved (Engstrom recipe).

    The parallel Welford update keeps a running ``mean`` and ``M2``
    (sum of squared deviations from the running mean), which makes the
    estimate exact regardless of batch size — unlike the previous
    EWMA-of-batch-variance scheme, which under-counted time-axis
    variation and grew stale once the policy moved off its initial
    return scale.
    """

    def __init__(self, num_envs: int, gamma: float, warmup_steps: int = 100):
        self.gamma = gamma
        self.num_envs = num_envs
        self.R = jnp.zeros(num_envs)

        # Welford accumulators. M2 / count = running variance once count >= 1.
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0.0

        self.warmup_steps = warmup_steps
        self.step_count = 0

    def scale(self, rewards: jnp.ndarray) -> jnp.ndarray:
        # Step the discounted-return tracker, then fold every per-env value
        # into the running stats as a fresh sample.
        self.R = self.gamma * self.R + rewards

        batch_count = float(self.R.size)
        batch_mean = float(self.R.mean())
        # Population variance is what Welford's M2 accumulates; ``var(ddof=0)``
        # is correct here. With num_envs == 1 it correctly returns 0.
        batch_var = float(self.R.var()) if self.R.size > 1 else 0.0
        batch_M2 = batch_var * batch_count

        # Parallel Welford merge of (count, mean, M2) with the new batch.
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        new_M2 = self.M2 + batch_M2 + (delta * delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.M2 = new_M2
        self.count = total_count
        self.step_count += 1

        # Warm-up: hand back raw rewards until we have a meaningful estimate.
        if self.step_count < self.warmup_steps:
            return rewards

        running_var = self.M2 / self.count if self.count > 0 else 1.0
        std = max(running_var**0.5, 1e-8)
        return rewards / std

    def reset_envs_vectorized(self, dones: jnp.ndarray):
        """Reset the discounted-return tracker for envs that just ended an
        episode. The Welford accumulators are NOT reset — they describe the
        global return distribution and should keep accruing across episodes.
        """
        self.R = jnp.where(dones, 0.0, self.R)

    @property
    def running_var(self) -> float:
        """Current running variance estimate (read-only convenience for logging)."""
        return float(self.M2 / self.count) if self.count > 0 else 1.0
