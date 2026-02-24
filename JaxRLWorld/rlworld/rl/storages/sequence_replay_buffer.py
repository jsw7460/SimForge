"""
Sequence Replay Buffer for model-based RL.

Samples consecutive trajectory chunks of fixed horizon length.
Uses NumPy for storage (fast mutable writes) and converts to JAX on sampling.
Handles episode boundaries correctly by avoiding sequences that cross them.
"""

from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class SequenceBatch(NamedTuple):
    """
    Batch of trajectory sequences for model-based RL.

    All arrays have shape [horizon+1, batch_size, dim] for observations
    and [horizon, batch_size, dim] for actions/rewards/terminated.
    """
    observations: jax.Array     # [horizon+1, batch_size, obs_dim]
    actions: jax.Array          # [horizon, batch_size, action_dim]
    rewards: jax.Array          # [horizon, batch_size, 1]
    terminated: jax.Array       # [horizon, batch_size, 1]


class SequenceReplayBuffer:
    """
    Replay buffer that samples consecutive trajectory sequences.

    Storage layout: [num_envs, size_per_env, feature_dim]
    Episode boundaries are tracked to avoid sampling across them.

    Used by TD-MPC2 and similar model-based RL algorithms that require
    short trajectory chunks for latent dynamics rollout training.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        size_per_env: int,
        horizon: int,
    ):
        """
        Args:
            num_envs: Number of parallel environments
            obs_dim: Observation dimension
            action_dim: Action dimension
            size_per_env: Maximum buffer size per environment
            horizon: Sequence length for sampling (number of transitions)
        """
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size_per_env = size_per_env
        self.horizon = horizon

        # Storage: obs, action, reward, terminated at each timestep
        # obs[t] is the observation before action[t], reward[t] is obtained after action[t]
        self.obs_buf = np.zeros((num_envs, size_per_env, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((num_envs, size_per_env, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((num_envs, size_per_env, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((num_envs, size_per_env, 1), dtype=np.float32)
        self.terminated_buf = np.zeros((num_envs, size_per_env, 1), dtype=np.float32)

        # Episode boundary tracking:
        # episode_id[env, pos] records which episode a transition belongs to.
        # Sequences must not span different episode_ids.
        self._episode_id = np.zeros((num_envs, size_per_env), dtype=np.int64)
        self._current_episode_id = np.zeros(num_envs, dtype=np.int64)

        # Buffer state
        self.ptr = 0
        self.filled_size = 0

    @property
    def size(self) -> int:
        """Total transitions stored across all environments."""
        return self.filled_size * self.num_envs

    @property
    def max_size(self) -> int:
        """Maximum capacity."""
        return self.size_per_env * self.num_envs

    def store_parallel(
        self,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
    ) -> None:
        """
        Store transitions from all parallel environments.

        Args:
            obs: Current observations [num_envs, obs_dim]
            action: Actions taken [num_envs, action_dim]
            reward: Rewards received [num_envs] or [num_envs, 1]
            next_obs: Next observations [num_envs, obs_dim]
            terminated: True termination flags [num_envs]
            truncated: Truncation flags [num_envs]
        """
        obs_np = np.asarray(obs)
        next_obs_np = np.asarray(next_obs)
        action_np = np.asarray(action)
        reward_np = np.asarray(reward)
        terminated_np = np.asarray(terminated, dtype=np.float32)
        truncated_np = np.asarray(truncated, dtype=np.float32)

        if reward_np.ndim == 1:
            reward_np = reward_np[:, None]
        if terminated_np.ndim == 1:
            terminated_np = terminated_np[:, None]
        if truncated_np.ndim == 1:
            truncated_np = truncated_np[:, None]

        # Store current transition
        self.obs_buf[:, self.ptr] = obs_np
        self.next_obs_buf[:, self.ptr] = next_obs_np
        self.action_buf[:, self.ptr] = action_np
        self.reward_buf[:, self.ptr] = reward_np
        self.terminated_buf[:, self.ptr] = terminated_np
        self._episode_id[:, self.ptr] = self._current_episode_id

        # Increment episode ID for environments that ended
        done = (terminated_np.squeeze(-1) > 0.5) | (truncated_np.squeeze(-1) > 0.5)
        self._current_episode_id[done] += 1

        # Advance pointer
        self.ptr = (self.ptr + 1) % self.size_per_env
        self.filled_size = min(self.filled_size + 1, self.size_per_env)

    def _sample_valid_indices(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample (env_idx, start_pos) pairs where the sequence of length
        `horizon` does not cross an episode boundary.

        Uses rejection sampling: sample candidates, filter valid ones, repeat.

        Returns:
            env_indices: [batch_size]
            start_positions: [batch_size]
        """
        # Maximum valid start position
        if self.filled_size >= self.size_per_env:
            # Buffer is full: any position could be a start, but we must
            # check episode boundaries. Avoid the `horizon` positions before ptr.
            max_logical_start = self.size_per_env - self.horizon
        else:
            max_logical_start = max(1, self.filled_size - self.horizon)

        collected_envs = []
        collected_pos = []
        remaining = batch_size

        max_attempts = 10
        for _ in range(max_attempts):
            # Over-sample to increase acceptance rate
            n_candidates = remaining * 3

            cand_envs = rng.integers(0, self.num_envs, size=n_candidates)

            if self.filled_size >= self.size_per_env:
                # Full buffer: sample logical offset and convert to physical position
                cand_logical = rng.integers(0, max_logical_start, size=n_candidates)
                cand_pos = (self.ptr + cand_logical) % self.size_per_env
            else:
                cand_pos = rng.integers(0, max_logical_start, size=n_candidates)

            # Validate: all positions in [start, start+horizon) must have same episode_id
            valid_mask = np.ones(n_candidates, dtype=bool)
            start_episode_ids = self._episode_id[cand_envs, cand_pos]

            for offset in range(1, self.horizon):
                check_pos = (cand_pos + offset) % self.size_per_env
                check_ids = self._episode_id[cand_envs, check_pos]
                valid_mask &= (check_ids == start_episode_ids)

            valid_envs = cand_envs[valid_mask]
            valid_pos = cand_pos[valid_mask]

            take = min(len(valid_envs), remaining)
            collected_envs.append(valid_envs[:take])
            collected_pos.append(valid_pos[:take])
            remaining -= take

            if remaining <= 0:
                break

        if remaining > 0:
            # Fallback: allow boundary-crossing sequences if we can't find enough valid ones
            fallback_envs = rng.integers(0, self.num_envs, size=remaining)
            if self.filled_size >= self.size_per_env:
                fallback_logical = rng.integers(0, max_logical_start, size=remaining)
                fallback_pos = (self.ptr + fallback_logical) % self.size_per_env
            else:
                fallback_pos = rng.integers(0, max(1, max_logical_start), size=remaining)
            collected_envs.append(fallback_envs)
            collected_pos.append(fallback_pos)

        env_indices = np.concatenate(collected_envs)[:batch_size]
        start_positions = np.concatenate(collected_pos)[:batch_size]
        return env_indices, start_positions

    def sample_batch(self, batch_size: int, key: jax.Array) -> SequenceBatch:
        """
        Sample a batch of trajectory sequences.

        Uses obs_buf for observations[0:H] and next_obs_buf for the final
        observation[H], ensuring correctness at episode boundaries.

        Args:
            batch_size: Number of sequences to sample
            key: JAX random key (used to seed NumPy RNG for index sampling)

        Returns:
            SequenceBatch with:
              observations: [horizon+1, batch_size, obs_dim]
              actions: [horizon, batch_size, action_dim]
              rewards: [horizon, batch_size, 1]
              terminated: [horizon, batch_size, 1]
        """
        if self.filled_size < self.horizon + 1:
            raise ValueError(
                f"Not enough data: need at least {self.horizon + 1} steps, "
                f"have {self.filled_size}"
            )

        seed = int(jax.random.randint(key, (), 0, 2 ** 31 - 1))
        rng = np.random.default_rng(seed)

        env_indices, start_positions = self._sample_valid_indices(batch_size, rng)

        # Positions for H transitions: [H, B]
        offsets = np.arange(self.horizon)
        positions = (start_positions[None, :] + offsets[:, None]) % self.size_per_env
        env_exp = np.broadcast_to(env_indices[None, :], positions.shape)

        # observations[0:H] from obs_buf at each transition start
        obs_main = self.obs_buf[env_exp, positions]  # [H, B, obs_dim]

        # observations[H] from next_obs_buf of the last transition
        last_pos = positions[-1]  # [B]
        obs_last = self.next_obs_buf[env_indices, last_pos]  # [B, obs_dim]

        obs_seq = np.concatenate([obs_main, obs_last[None]], axis=0)  # [H+1, B, obs_dim]
        action_seq = self.action_buf[env_exp, positions]  # [H, B, action_dim]
        reward_seq = self.reward_buf[env_exp, positions]  # [H, B, 1]
        terminated_seq = self.terminated_buf[env_exp, positions]  # [H, B, 1]

        return SequenceBatch(
            observations=jnp.asarray(obs_seq),
            actions=jnp.asarray(action_seq),
            rewards=jnp.asarray(reward_seq),
            terminated=jnp.asarray(terminated_seq),
        )

    def get_recent_actions(self, n: int) -> jax.Array:
        """Get most recent n actions."""
        if self.filled_size == 0:
            return jnp.zeros((0, self.action_dim))
        n = min(n, self.filled_size * self.num_envs)
        if self.filled_size >= self.size_per_env:
            recent_pos = (self.ptr - 1 - np.arange(min(n // self.num_envs, self.size_per_env))) % self.size_per_env
        else:
            recent_pos = np.arange(self.filled_size - 1, -1, -1)[:n // self.num_envs]
        actions = self.action_buf[:, recent_pos].reshape(-1, self.action_dim)
        return jnp.asarray(actions[:n])

    def clear(self) -> None:
        """Reset the buffer."""
        self.ptr = 0
        self.filled_size = 0
        self._current_episode_id[:] = 0

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Buffer statistics."""
        return {
            "filled_size": self.filled_size,
            "ptr": self.ptr,
            "capacity": self.size_per_env,
            "total_transitions": self.filled_size * self.num_envs,
            "fill_ratio": self.filled_size / max(1, self.size_per_env),
            "horizon": self.horizon,
        }