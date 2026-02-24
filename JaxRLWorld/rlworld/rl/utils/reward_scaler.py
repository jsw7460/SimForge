import jax.numpy as jnp


class RewardScaler:
    """
    Discount-based reward scaling.

    Scales rewards by running std of discounted returns,
    without subtracting mean (preserves reward sign).

    Reference: "Implementation Matters in Deep Policy Gradients" (Engstrom et al.)
    """

    def __init__(self, num_envs: int, gamma: float, warmup_steps: int = 100):
        self.gamma = gamma
        self.num_envs = num_envs
        self.R = jnp.zeros(num_envs)

        self.running_var = 1.0
        self.count = 1e-4

        self.warmup_steps = warmup_steps
        self.step_count = 0

    def scale(self, rewards: jnp.ndarray) -> jnp.ndarray:
        self.R = self.gamma * self.R + rewards

        # Update running variance
        batch_var = float(self.R.var()) if self.R.size > 1 else 0.0
        batch_count = self.R.size

        total_count = self.count + batch_count
        self.running_var = (self.running_var * self.count + batch_var * batch_count) / total_count
        self.count = total_count

        self.step_count += 1

        # Warmup: don't scale until we have enough samples
        if self.step_count < self.warmup_steps:
            return rewards

        std = max(self.running_var ** 0.5, 1e-8)
        return rewards / std

    def reset_envs_vectorized(self, dones: jnp.ndarray):
        """Vectorized reset - no dynamic indexing."""
        self.R = jnp.where(dones, 0.0, self.R)