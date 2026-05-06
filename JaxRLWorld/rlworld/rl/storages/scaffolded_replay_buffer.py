"""
Scaffolded Sequence Replay Buffer.

Extends SequenceReplayBuffer with privileged observation storage.
Follows identical patterns: NumPy storage, episode boundary tracking,
rejection-sampled valid indices.

ScaffoldedSequenceBatch extends SequenceBatch with privileged_observations field.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rlworld.rl.storages.sequence_replay_buffer import (
    SequenceReplayBuffer,
)


class ScaffoldedSequenceBatch(NamedTuple):
    """
    Batch of trajectory sequences with privileged observations.

    Extends SequenceBatch layout:
    - observations:            [horizon+1, batch_size, obs_dim]
    - actions:                 [horizon,   batch_size, action_dim]
    - rewards:                 [horizon,   batch_size, 1]
    - terminated:              [horizon,   batch_size, 1]
    - privileged_observations: [horizon+1, batch_size, priv_dim]
    """

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    privileged_observations: jax.Array


class ScaffoldedSequenceReplayBuffer(SequenceReplayBuffer):
    """
    Sequence replay buffer with privileged observation storage.

    Inherits all base functionality (episode boundary tracking,
    valid index sampling, circular buffer) and adds parallel
    storage for privileged observations.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        privileged_obs_dim: int,
        size_per_env: int,
        horizon: int,
    ):
        super().__init__(
            num_envs=num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            size_per_env=size_per_env,
            horizon=horizon,
        )

        self.privileged_obs_dim = privileged_obs_dim

        # Privileged obs storage (mirrors obs_buf / next_obs_buf)
        self.priv_obs_buf = np.zeros((num_envs, size_per_env, privileged_obs_dim), dtype=np.float32)
        self.next_priv_obs_buf = np.zeros((num_envs, size_per_env, privileged_obs_dim), dtype=np.float32)

    def store_parallel(
        self,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_obs: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        privileged_obs: jax.Array = None,
        next_privileged_obs: jax.Array = None,
    ) -> None:
        """
        Store transitions with privileged observations.

        Privileged obs are stored at the same buffer position as the
        base transition (same ptr, same episode boundary tracking).
        """
        # Store privileged obs BEFORE super() advances ptr
        if privileged_obs is not None:
            self.priv_obs_buf[:, self.ptr] = np.asarray(privileged_obs)
        if next_privileged_obs is not None:
            self.next_priv_obs_buf[:, self.ptr] = np.asarray(next_privileged_obs)

        # Base class stores obs, action, reward, next_obs, terminated
        # and advances ptr + updates episode_id
        super().store_parallel(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
        )

    def sample_batch(self, batch_size: int, key: jax.Array) -> ScaffoldedSequenceBatch:
        """
        Sample batch with privileged observations.

        Uses base class index sampling (episode boundary aware),
        then extracts privileged obs at the same indices.
        """
        if self.filled_size < self.horizon + 1:
            raise ValueError(f"Not enough data: need at least {self.horizon + 1} steps, have {self.filled_size}")

        seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
        rng = np.random.default_rng(seed)

        env_indices, start_positions = self._sample_valid_indices(batch_size, rng)

        # Positions for H transitions: [H, B]
        offsets = np.arange(self.horizon)
        positions = (start_positions[None, :] + offsets[:, None]) % self.size_per_env
        env_exp = np.broadcast_to(env_indices[None, :], positions.shape)

        # Base observations (same logic as SequenceReplayBuffer.sample_batch)
        obs_main = self.obs_buf[env_exp, positions]  # [H, B, obs_dim]
        last_pos = positions[-1]
        obs_last = self.next_obs_buf[env_indices, last_pos]  # [B, obs_dim]
        obs_seq = np.concatenate([obs_main, obs_last[None]], axis=0)  # [H+1, B, obs_dim]

        action_seq = self.action_buf[env_exp, positions]
        reward_seq = self.reward_buf[env_exp, positions]
        terminated_seq = self.terminated_buf[env_exp, positions]

        # Privileged observations (same index pattern)
        priv_main = self.priv_obs_buf[env_exp, positions]  # [H, B, priv_dim]
        priv_last = self.next_priv_obs_buf[env_indices, last_pos]  # [B, priv_dim]
        priv_seq = np.concatenate([priv_main, priv_last[None]], axis=0)  # [H+1, B, priv_dim]

        return ScaffoldedSequenceBatch(
            observations=jnp.asarray(obs_seq),
            actions=jnp.asarray(action_seq),
            rewards=jnp.asarray(reward_seq),
            terminated=jnp.asarray(terminated_seq),
            privileged_observations=jnp.asarray(priv_seq),
        )

    def clear(self) -> None:
        """Reset buffer including privileged storage."""
        super().clear()
        self.priv_obs_buf[:] = 0.0
        self.next_priv_obs_buf[:] = 0.0
