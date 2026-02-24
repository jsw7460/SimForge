from typing import Dict, Any, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ReplayBatch(NamedTuple):
    """
    Batch of transitions sampled from replay buffer.
    Supports n-step returns with variable effective n per sample.
    """
    actor_observations: jax.Array
    critic_observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_actor_observations: jax.Array
    next_critic_observations: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    gamma_power: jax.Array


class ReplayBuffer:
    """
    Replay Buffer optimized for parallel environments.
    Uses NumPy for storage (mutable, fast writes) and converts to JAX on sampling.
    Supports n-step returns computation at sampling time.
    """

    def __init__(
        self,
        num_envs: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        size_per_env: int,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        """
        Initialize the parallel replay buffer.

        Args:
            num_envs: Number of parallel environments
            actor_obs_dim: Dimension of actor observations
            critic_obs_dim: Dimension of critic observations
            act_dim: Dimension of actions
            size_per_env: Maximum size of buffer per environment
            n_steps: Number of steps for n-step returns (default: 1)
            gamma: Discount factor for n-step returns (default: 0.99)
        """
        self.num_envs = num_envs
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
        self.size_per_env = size_per_env
        self.total_size = num_envs * size_per_env
        self.n_steps = n_steps
        self.gamma = gamma

        # NumPy buffers for fast in-place writes: [num_envs, size_per_env, dim]
        self.actor_obs_buf = np.zeros(
            (num_envs, size_per_env, actor_obs_dim), dtype=np.float32
        )
        self.critic_obs_buf = np.zeros(
            (num_envs, size_per_env, critic_obs_dim), dtype=np.float32
        )
        self.next_actor_obs_buf = np.zeros(
            (num_envs, size_per_env, actor_obs_dim), dtype=np.float32
        )
        self.next_critic_obs_buf = np.zeros(
            (num_envs, size_per_env, critic_obs_dim), dtype=np.float32
        )
        self.acts_buf = np.zeros(
            (num_envs, size_per_env, act_dim), dtype=np.float32
        )
        self.rews_buf = np.zeros(
            (num_envs, size_per_env, 1), dtype=np.float32
        )
        self.terminated_buf = np.zeros(
            (num_envs, size_per_env, 1), dtype=np.float32
        )
        self.truncated_buf = np.zeros(
            (num_envs, size_per_env, 1), dtype=np.float32
        )

        # Single synchronized pointer
        self.ptr = 0
        self.filled_size = 0

    @property
    def size(self) -> int:
        """Get the current total number of transitions stored."""
        return self.filled_size * self.num_envs

    @property
    def max_size(self) -> int:
        """Get the maximum capacity of the buffer."""
        return self.total_size

    def store_parallel(
        self,
        actor_obs: jax.Array,
        critic_obs: jax.Array,
        act: jax.Array,
        rew: jax.Array,
        next_actor_obs: jax.Array,
        next_critic_obs: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        **kwargs
    ) -> None:
        """
        Store transitions from multiple parallel environments.
        All environments write to the same buffer position.

        Args:
            actor_obs: [num_envs, actor_obs_dim]
            critic_obs: [num_envs, critic_obs_dim]
            act: [num_envs, act_dim]
            rew: [num_envs] or [num_envs, 1]
            next_actor_obs: [num_envs, actor_obs_dim]
            next_critic_obs: [num_envs, critic_obs_dim]
            terminated: [num_envs] or [num_envs, 1]
            truncated: [num_envs] or [num_envs, 1]
        """
        # Convert JAX arrays to NumPy
        actor_obs_np = np.asarray(actor_obs)
        critic_obs_np = np.asarray(critic_obs)
        act_np = np.asarray(act)
        rew_np = np.asarray(rew)
        next_actor_obs_np = np.asarray(next_actor_obs)
        next_critic_obs_np = np.asarray(next_critic_obs)
        terminated_np = np.asarray(terminated, dtype=np.float32)
        truncated_np = np.asarray(truncated, dtype=np.float32)

        # Ensure correct shapes [num_envs, 1]
        if rew_np.ndim == 1:
            rew_np = rew_np[:, None]
        if terminated_np.ndim == 1:
            terminated_np = terminated_np[:, None]
        if truncated_np.ndim == 1:
            truncated_np = truncated_np[:, None]

        # In-place update (fast)
        self.actor_obs_buf[:, self.ptr] = actor_obs_np
        self.critic_obs_buf[:, self.ptr] = critic_obs_np
        self.acts_buf[:, self.ptr] = act_np
        self.rews_buf[:, self.ptr] = rew_np
        self.next_actor_obs_buf[:, self.ptr] = next_actor_obs_np
        self.next_critic_obs_buf[:, self.ptr] = next_critic_obs_np
        self.terminated_buf[:, self.ptr] = terminated_np
        self.truncated_buf[:, self.ptr] = truncated_np

        # Update pointer (circular)
        self.ptr = (self.ptr + 1) % self.size_per_env

        # Update filled size
        self.filled_size = min(self.filled_size + 1, self.size_per_env)

    def _compute_nstep_data(
        self,
        env_indices: np.ndarray,
        start_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute n-step returns and get observations n steps later.

        Args:
            env_indices: Environment indices [batch_size]
            start_positions: Starting positions in buffer [batch_size]

        Returns:
            nstep_rewards, final_next_actor_obs, final_next_critic_obs,
            final_terminated, final_truncated, gamma_power
        """
        batch_size = env_indices.shape[0]

        # Create sequence offsets [0, 1, ..., n_steps-1]
        seq_offsets = np.arange(self.n_steps)

        # Calculate all positions for n-step sequence: [batch_size, n_steps]
        all_positions = (start_positions[:, None] + seq_offsets[None, :]) % self.size_per_env

        # Expand env_indices to match: [batch_size, n_steps]
        env_indices_expanded = np.broadcast_to(env_indices[:, None], (batch_size, self.n_steps))

        # Gather rewards, terminated, and truncated for all steps
        all_rewards = self.rews_buf[env_indices_expanded, all_positions, 0]
        all_terminated = self.terminated_buf[env_indices_expanded, all_positions, 0]
        all_truncated = self.truncated_buf[env_indices_expanded, all_positions, 0]

        # Episode boundary = terminated OR truncated
        all_boundaries = all_terminated + all_truncated

        # Mask: include reward at boundary step, exclude after
        cumsum_boundaries = np.cumsum(all_boundaries, axis=1)
        cumsum_boundaries_shifted = np.concatenate([
            np.zeros((batch_size, 1)),
            cumsum_boundaries[:, :-1]
        ], axis=1)
        done_masks = (cumsum_boundaries_shifted == 0).astype(np.float32)

        # Discount factors [γ^0, γ^1, ..., γ^(n-1)]
        discounts = np.power(self.gamma, np.arange(self.n_steps).astype(np.float32))

        # Apply mask and discounts
        masked_rewards = all_rewards * done_masks
        discounted_rewards = masked_rewards * discounts[None, :]

        # Sum to get n-step reward
        nstep_rewards = discounted_rewards.sum(axis=1, keepdims=True)

        # Find effective n (number of steps actually used)
        has_boundary = all_boundaries.sum(axis=1) > 0
        first_boundary_idx = np.argmax(all_boundaries > 0, axis=1)
        effective_n = np.where(has_boundary, first_boundary_idx + 1, self.n_steps)

        # Position of the last used transition
        last_used_position = (start_positions + effective_n - 1) % self.size_per_env

        # Get next_obs from the last used transition
        final_next_actor_obs = self.next_actor_obs_buf[env_indices, last_used_position]
        final_next_critic_obs = self.next_critic_obs_buf[env_indices, last_used_position]

        # Terminal flags from the last used transition
        final_terminated = self.terminated_buf[env_indices, last_used_position]
        final_truncated = self.truncated_buf[env_indices, last_used_position]

        # gamma^n for bootstrap
        gamma_power = np.power(self.gamma, effective_n.astype(np.float32))[:, None]

        return (
            nstep_rewards,
            final_next_actor_obs,
            final_next_critic_obs,
            final_terminated,
            final_truncated,
            gamma_power,
        )

    def sample_batch(self, batch_size: int, key: jax.Array) -> ReplayBatch:
        """
        Sample a batch of transitions from the buffer.
        Computes n-step returns if n_steps > 1.

        Args:
            batch_size: Size of the batch to sample
            key: JAX random key

        Returns:
            ReplayBatch object with n-step data (JAX arrays)
        """
        if self.filled_size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        key1, key2 = jax.random.split(key)

        # Sample indices using JAX, convert to NumPy for indexing
        env_indices = np.asarray(jax.random.randint(key1, (batch_size,), 0, self.num_envs))

        # Sample position indices
        if self.filled_size >= self.size_per_env:
            max_logical = self.size_per_env - (self.n_steps - 1)
            logical_start = np.asarray(jax.random.randint(key2, (batch_size,), 0, max_logical))
            pos_indices = (self.ptr + logical_start) % self.size_per_env
        else:
            max_start_idx = max(1, self.filled_size - self.n_steps + 1)
            pos_indices = np.asarray(jax.random.randint(key2, (batch_size,), 0, max_start_idx))

        # Get starting observations and actions (NumPy indexing)
        actor_obs = self.actor_obs_buf[env_indices, pos_indices]
        critic_obs = self.critic_obs_buf[env_indices, pos_indices]
        actions = self.acts_buf[env_indices, pos_indices]

        if self.n_steps > 1:
            (
                nstep_rewards,
                next_actor_obs,
                next_critic_obs,
                terminated,
                truncated,
                gamma_power,
            ) = self._compute_nstep_data(env_indices, pos_indices)
        else:
            next_actor_obs = self.next_actor_obs_buf[env_indices, pos_indices]
            next_critic_obs = self.next_critic_obs_buf[env_indices, pos_indices]
            nstep_rewards = self.rews_buf[env_indices, pos_indices]
            terminated = self.terminated_buf[env_indices, pos_indices]
            truncated = self.truncated_buf[env_indices, pos_indices]
            gamma_power = np.full((batch_size, 1), self.gamma, dtype=np.float32)

        # Convert to JAX arrays for training
        return ReplayBatch(
            actor_observations=jnp.asarray(actor_obs),
            critic_observations=jnp.asarray(critic_obs),
            actions=jnp.asarray(actions),
            rewards=jnp.asarray(nstep_rewards),
            next_actor_observations=jnp.asarray(next_actor_obs),
            next_critic_observations=jnp.asarray(next_critic_obs),
            terminated=jnp.asarray(terminated),
            truncated=jnp.asarray(truncated),
            gamma_power=jnp.asarray(gamma_power),
        )

    def get_recent_actions(self, n: int) -> jax.Array:
        """Get the most recent n actions from the buffer."""
        if self.filled_size == 0:
            return jnp.zeros((0, self.act_dim))

        n = min(n, self.filled_size * self.num_envs)

        if self.filled_size >= self.size_per_env:
            recent_pos = (self.ptr - 1 - np.arange(min(n // self.num_envs, self.size_per_env))) % self.size_per_env
        else:
            recent_pos = np.arange(self.filled_size - 1, -1, -1)[:n // self.num_envs]

        actions = self.acts_buf[:, recent_pos].reshape(-1, self.act_dim)
        return jnp.asarray(actions[:n])

    def clear(self) -> None:
        """Clear the buffer by resetting pointers and sizes."""
        self.ptr = 0
        self.filled_size = 0

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffer state."""
        return {
            "filled_size": self.filled_size,
            "ptr": self.ptr,
            "capacity": self.size_per_env,
            "total_transitions": self.filled_size * self.num_envs,
            "fill_ratio": self.filled_size / max(1, self.size_per_env),
            "n_steps": self.n_steps,
            "gamma": self.gamma,
        }

    def save(self, path: str) -> None:
        """Save the replay buffer to a file."""
        save_dict = {
            "actor_obs": self.actor_obs_buf,
            "critic_obs": self.critic_obs_buf,
            "acts": self.acts_buf,
            "rews": self.rews_buf,
            "next_actor_obs": self.next_actor_obs_buf,
            "next_critic_obs": self.next_critic_obs_buf,
            "terminated": self.terminated_buf,
            "truncated": self.truncated_buf,
            "ptr": self.ptr,
            "filled_size": self.filled_size,
            "num_envs": self.num_envs,
            "size_per_env": self.size_per_env,
            "actor_obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "act_dim": self.act_dim,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
        }
        np.savez(path, **save_dict)

    def load(self, path: str) -> None:
        """Load the replay buffer from a file."""
        data = np.load(path)

        if (
            data["num_envs"] != self.num_envs
            or data["actor_obs_dim"] != self.actor_obs_dim
            or data["critic_obs_dim"] != self.critic_obs_dim
            or data["act_dim"] != self.act_dim
        ):
            raise ValueError("Loaded buffer config doesn't match current buffer")

        self.actor_obs_buf = data["actor_obs"].copy()
        self.critic_obs_buf = data["critic_obs"].copy()
        self.acts_buf = data["acts"].copy()
        self.rews_buf = data["rews"].copy()
        self.next_actor_obs_buf = data["next_actor_obs"].copy()
        self.next_critic_obs_buf = data["next_critic_obs"].copy()
        self.terminated_buf = data["terminated"].copy()
        self.truncated_buf = data["truncated"].copy()
        self.ptr = int(data["ptr"])
        self.filled_size = int(data["filled_size"])