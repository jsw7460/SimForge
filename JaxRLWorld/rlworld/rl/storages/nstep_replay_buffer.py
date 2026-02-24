from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch

import genesis as gs


@dataclass
class NStepReplayBatch:
    """
    Batch of transitions sampled from a replay buffer for off-policy algorithms.
    """

    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_actor_observations: torch.Tensor
    next_critic_observations: torch.Tensor
    dones: torch.Tensor
    truncated: torch.Tensor


@dataclass
class EstimatorNStepReplayBatch(NStepReplayBatch):
    """
    Batch of transitions sampled from a replay buffer for off-policy algorithms,
    with additional fields for state estimation.
    """

    estimator_observations: torch.Tensor = None
    robot_states: torch.Tensor = None


class NstepReplayBuffer:
    """
    FastTD3-style Replay Buffer optimized for parallel environments.

    This buffer uses a synchronized storage pattern where all environments write to the same
    buffer position, with data shape (num_envs, size_per_env, feature_dim) for better
    memory locality and vectorized operations.
    """

    def __init__(
        self,
        num_envs: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        size_per_env: int,
        device: str = "cuda:0",
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        """
        Initialize the FastTD3-style parallel replay buffer.
        Args:
            num_envs: Number of parallel environments
            actor_obs_dim: Dimension of actor observations
            critic_obs_dim: Dimension of critic observations
            act_dim: Dimension of actions
            size_per_env: Maximum size of buffer per environment
            device: Device to store tensors on
            n_steps: Number of steps for n-step returns (default: 1)
            gamma: Discount factor for n-step returns (default: 0.99)
        """
        self.num_envs = num_envs
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
        self.size_per_env = size_per_env
        self.device = device
        self.n_steps = n_steps
        self.gamma = gamma

        # Create buffers with shape (num_envs, size_per_env, feature_dim)
        self.float_dtype = gs.tc_float if hasattr(gs, 'tc_float') else torch.float32
        self.actor_obs_buf = torch.zeros(
            (num_envs, size_per_env, actor_obs_dim),
            dtype=self.float_dtype,
            device=device
        )
        self.critic_obs_buf = torch.zeros(
            (num_envs, size_per_env, critic_obs_dim),
            dtype=self.float_dtype,
            device=device
        )
        self.next_actor_obs_buf = torch.zeros(
            (num_envs, size_per_env, actor_obs_dim),
            dtype=self.float_dtype,
            device=device
        )
        self.next_critic_obs_buf = torch.zeros(
            (num_envs, size_per_env, critic_obs_dim),
            dtype=self.float_dtype,
            device=device
        )
        self.acts_buf = torch.zeros(
            (num_envs, size_per_env, act_dim),
            dtype=self.float_dtype,
            device=device
        )
        self.rews_buf = torch.zeros(
            (num_envs, size_per_env),
            dtype=self.float_dtype,
            device=device
        )
        self.done_buf = torch.zeros(
            (num_envs, size_per_env),
            dtype=self.float_dtype,
            device=device
        )

        # Add truncation buffer for proper episode boundary handling
        # Truncations indicate if the episode was truncated (not done naturally)
        self.truncations = torch.zeros(
            (num_envs, size_per_env),
            dtype=torch.bool,
            device=device
        )

        # Single pointer for synchronized updates
        self.ptr = 0
        self.filled_size = 0  # Tracks how many positions have been filled
        self.buffer_size = size_per_env  # For compatibility with FastTD3 style

    @property
    def size(self) -> int:
        """
        Get the current total number of transitions stored across all environments.

        Returns:
            Total number of experiences stored
        """
        return self.filled_size * self.num_envs

    @property
    def max_size(self) -> int:
        """
        Get the maximum capacity of the buffer.

        Returns:
            Maximum number of transitions that can be stored
        """
        return self.size_per_env * self.num_envs

    @torch.no_grad()
    def store_parallel(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_actor_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        done: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """
        Store transitions from multiple parallel environments at once.
        All environments write to the same buffer position for synchronized storage.

        Args:
            actor_obs: Actor observations from all environments [num_envs, actor_obs_dim]
            critic_obs: Critic observations from all environments [num_envs, critic_obs_dim]
            act: Actions from all environments [num_envs, act_dim]
            rew: Rewards from all environments [num_envs] or [num_envs, 1]
            next_actor_obs: Next actor observations from all environments [num_envs, actor_obs_dim]
            next_critic_obs: Next critic observations from all environments [num_envs, critic_obs_dim]
            done: Done flags from all environments [num_envs] or [num_envs, 1]
            truncated: Truncation flags from all environments [num_envs] or [num_envs, 1] (optional)
        """
        # Ensure rewards and dones have correct shape
        if rew.dim() == 2:
            rew = rew.squeeze(-1)
        if done.dim() == 2:
            done = done.squeeze(-1)
        if truncated is not None and truncated.dim() == 2:
            truncated = truncated.squeeze(-1)

        # Store data at current pointer position for all environments
        self.actor_obs_buf[:, self.ptr] = actor_obs
        self.critic_obs_buf[:, self.ptr] = critic_obs
        self.acts_buf[:, self.ptr] = act
        self.rews_buf[:, self.ptr] = rew
        self.next_actor_obs_buf[:, self.ptr] = next_actor_obs
        self.next_critic_obs_buf[:, self.ptr] = next_critic_obs
        self.done_buf[:, self.ptr] = done.float()
        self.truncations[:, self.ptr] = truncated

        # Update pointer with wraparound
        self.ptr = (self.ptr + 1) % self.size_per_env

        # Update filled size (capped at size_per_env)
        self.filled_size = min(self.filled_size + 1, self.size_per_env)

    def _compute_nstep_data(
        self,
        env_indices: torch.Tensor,
        start_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute n-step returns and get observations n steps later (FastTD3 style).

        Args:
            env_indices: Environment indices [batch_size]
            start_positions: Starting positions [batch_size]

        Returns:
            nstep_rewards: Computed n-step returns [batch_size]
            final_next_actor_obs: Actor observations n steps later [batch_size, actor_obs_dim]
            final_next_critic_obs: Critic observations n steps later [batch_size, critic_obs_dim]
            final_dones: Done flags indicating if episode ended within n steps [batch_size]
        """
        # Create sequence offsets for n-step indexing
        seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, -1)

        # Calculate all indices for the n-step sequence
        # Shape: [batch_size, n_steps]
        all_positions = (start_positions.unsqueeze(-1) + seq_offsets) % self.buffer_size

        # Prepare indices for gathering - need to combine env and position indices
        # Create expanded env indices that match shape [batch_size, n_steps]
        env_indices_expanded = env_indices.unsqueeze(-1).expand(-1, self.n_steps)

        # Gather all rewards, dones, and truncations for the sequence
        # We need to index with both env and position indices
        all_rewards = self.rews_buf[env_indices_expanded, all_positions]
        all_dones = self.done_buf[env_indices_expanded, all_positions]
        all_truncations = self.truncations[env_indices_expanded, all_positions]

        # Create masks for rewards *after* first done
        # Shift dones to the right (first reward should not be masked)
        all_dones_shifted = torch.cat([
            torch.zeros_like(all_dones[:, :1]),
            all_dones[:, :-1]
        ], dim=1)

        # Create cumulative product to zero out rewards after first done
        done_masks = torch.cumprod(1.0 - all_dones_shifted, dim=1)

        # Create discount factors
        discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device))

        # Apply masks and discounts to rewards
        masked_rewards = all_rewards * done_masks
        discounted_rewards = masked_rewards * discounts.view(1, -1)

        # Sum rewards along the n_step dimension
        n_step_rewards = discounted_rewards.sum(dim=1)

        # Find index of first done or truncation or last step for each sequence
        # Using argmax to find first occurrence (returns 0 if all are False)
        first_done = torch.argmax((all_dones > 0).float(), dim=1)
        first_trunc = torch.argmax((all_truncations > 0).float(), dim=1)

        # Handle case where there are no dones or truncations
        no_dones = all_dones.sum(dim=1) == 0
        no_truncs = all_truncations.sum(dim=1) == 0

        # When no dones or truncs, use the last index
        first_done = torch.where(no_dones, self.n_steps - 1, first_done)
        first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

        # Take the minimum (first) of done or truncation
        final_indices = torch.minimum(first_done, first_trunc)

        # Get positions for final observations
        final_positions = (start_positions + final_indices) % self.buffer_size

        # Gather final next observations and done flags
        final_next_actor_obs = self.next_actor_obs_buf[env_indices, final_positions]
        final_next_critic_obs = self.next_critic_obs_buf[env_indices, final_positions]
        final_dones = self.done_buf[env_indices, final_positions]
        final_truncs = self.truncations[env_indices, final_positions]

        return n_step_rewards, final_next_actor_obs, final_next_critic_obs, final_dones, final_truncs

    @torch.no_grad()
    def sample_batch(self, batch_size: int = 256) -> NStepReplayBatch:
        """
        Sample a batch of transitions from the buffer using FastTD3-style sampling.

        Args:
            batch_size: Total size of the batch to sample

        Returns:
            ReplayBatch object containing sampled tensors of shape (batch_size, feature_dim)
        """
        if self.filled_size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        # Sample indices using FastTD3 approach
        # env_indices, pos_indices = self._sample_indices(batch_size)
        batch_size_per_env = batch_size // self.num_envs
        remainder = batch_size % self.num_envs

        if self.filled_size >= self.buffer_size:
            # Buffer is full
            current_pos = self.ptr % self.buffer_size

            # Temporarily mark positions as truncated to avoid sampling across episodes
            # This is only needed for n-step returns
            if self.n_steps > 1:
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(
                    self.done_buf[:, current_pos - 1]
                )

            # Sample from full buffer
            indices = torch.randint(
                0,
                self.buffer_size,
                (self.num_envs, batch_size_per_env),
                device=self.device,
            )

        else:
            # Buffer not full - ensure n-step sequence doesn't exceed valid data
            max_start_idx = max(1, self.ptr - self.n_steps + 1) if self.n_steps > 1 else self.ptr
            indices = torch.randint(
                0,
                max_start_idx,
                (self.num_envs, batch_size_per_env),
                device=self.device,
            )

        # Flatten indices and create environment indices
        env_indices = torch.arange(self.num_envs, device=self.device).repeat_interleave(batch_size_per_env)
        pos_indices = indices.flatten()

        # Add remainder samples if needed
        if remainder > 0:
            extra_env_indices = torch.randint(0, self.num_envs, (remainder,), device=self.device)
            if self.ptr >= self.buffer_size:
                extra_pos_indices = torch.randint(0, self.buffer_size, (remainder,), device=self.device)
            else:
                max_start_idx = max(1, self.ptr - self.n_steps + 1) if self.n_steps > 1 else self.ptr
                extra_pos_indices = torch.randint(0, max_start_idx, (remainder,), device=self.device)

            env_indices = torch.cat([env_indices, extra_env_indices])
            pos_indices = torch.cat([pos_indices, extra_pos_indices])
            raise NotImplementedError("Remainder sampling not implemented in this snippet")

        # Extract starting observations and actions
        actor_obs = self.actor_obs_buf[env_indices, pos_indices]
        critic_obs = self.critic_obs_buf[env_indices, pos_indices]
        actions = self.acts_buf[env_indices, pos_indices]

        # Handle n-step vs 1-step cases
        if self.n_steps > 1:
            # Compute n-step returns and get observations n steps later
            nstep_rewards, next_actor_obs, next_critic_obs, dones, truncs = self._compute_nstep_data(
                env_indices, pos_indices
            )
            rewards = nstep_rewards
        else:
            # Standard 1-step
            next_actor_obs = self.next_actor_obs_buf[env_indices, pos_indices]
            next_critic_obs = self.next_critic_obs_buf[env_indices, pos_indices]
            rewards = self.rews_buf[env_indices, pos_indices]
            dones = self.done_buf[env_indices, pos_indices]
            truncs = self.truncations[env_indices, pos_indices]

        if self.filled_size >= self.buffer_size and self.n_steps > 1:
            # Restore truncations after sampling
            self.truncations[:, current_pos - 1] = curr_truncations

        # Return batch with proper shapes
        return NStepReplayBatch(
            actor_observations=actor_obs,
            critic_observations=critic_obs,
            actions=actions,
            rewards=rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards,
            next_actor_observations=next_actor_obs,
            next_critic_observations=next_critic_obs,
            dones=dones.unsqueeze(-1) if dones.dim() == 1 else dones,
            truncated=truncs.unsqueeze(-1) if truncs.dim() == 1 else truncs,
        )

    @torch.no_grad()
    def sample_batch_with_indices(self, batch_size: int = 256) -> Tuple[NStepReplayBatch, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer and return the indices.

        Args:
            batch_size: Total size of the batch to sample

        Returns:
            Tuple of (ReplayBatch, indices)
        """
        if self.filled_size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        # Sample indices using FastTD3 approach
        env_indices, pos_indices = self._sample_indices(batch_size)

        # Extract starting observations and actions
        actor_obs = self.actor_obs_buf[env_indices, pos_indices]
        critic_obs = self.critic_obs_buf[env_indices, pos_indices]
        actions = self.acts_buf[env_indices, pos_indices]

        # Handle n-step vs 1-step cases
        if self.n_steps > 1:
            # Compute n-step returns and get observations n steps later
            nstep_rewards, next_actor_obs, next_critic_obs, dones = self._compute_nstep_data(
                env_indices, pos_indices
            )
            rewards = nstep_rewards
        else:
            # Standard 1-step
            next_actor_obs = self.next_actor_obs_buf[env_indices, pos_indices]
            next_critic_obs = self.next_critic_obs_buf[env_indices, pos_indices]
            rewards = self.rews_buf[env_indices, pos_indices]
            dones = self.done_buf[env_indices, pos_indices]

        # Create batch
        batch = NStepReplayBatch(
            actor_observations=actor_obs,
            critic_observations=critic_obs,
            actions=actions,
            rewards=rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards,
            next_actor_observations=next_actor_obs,
            next_critic_observations=next_critic_obs,
            dones=dones.unsqueeze(-1) if dones.dim() == 1 else dones,
        )

        # Create flat indices for compatibility (env_idx * size_per_env + pos_idx)
        flat_indices = env_indices * self.size_per_env + pos_indices

        return batch, flat_indices

    def mini_batch_generator(self, num_mini_batches, num_epochs=1):
        """
        Generate mini-batches from the replay buffer.

        Args:
            num_mini_batches: Number of mini-batches to generate
            num_epochs: Number of epochs to generate mini-batches for

        Returns:
            Generator yielding ReplayBatch objects
        """
        # Calculate total samples and mini-batch size
        total_samples = self.filled_size * self.num_envs
        mini_batch_size = total_samples // num_mini_batches

        for epoch in range(num_epochs):
            # Sample all indices for this epoch using FastTD3 approach
            all_env_indices, all_pos_indices = self._sample_indices(total_samples)

            # Shuffle the indices
            perm = torch.randperm(total_samples, device=self.device)
            all_env_indices = all_env_indices[perm]
            all_pos_indices = all_pos_indices[perm]

            for i in range(num_mini_batches):
                start_idx = i * mini_batch_size
                end_idx = min((i + 1) * mini_batch_size, total_samples)

                env_indices = all_env_indices[start_idx:end_idx]
                pos_indices = all_pos_indices[start_idx:end_idx]

                # Extract starting observations and actions
                actor_obs = self.actor_obs_buf[env_indices, pos_indices]
                critic_obs = self.critic_obs_buf[env_indices, pos_indices]
                actions = self.acts_buf[env_indices, pos_indices]

                # Handle n-step vs 1-step cases
                if self.n_steps > 1:
                    # Compute n-step returns and get observations n steps later
                    nstep_rewards, next_actor_obs, next_critic_obs, dones = self._compute_nstep_data(
                        env_indices, pos_indices
                    )
                    rewards = nstep_rewards
                else:
                    # Standard 1-step
                    next_actor_obs = self.next_actor_obs_buf[env_indices, pos_indices]
                    next_critic_obs = self.next_critic_obs_buf[env_indices, pos_indices]
                    rewards = self.rews_buf[env_indices, pos_indices]
                    dones = self.done_buf[env_indices, pos_indices]

                # Yield batch
                yield NStepReplayBatch(
                    actor_observations=actor_obs,
                    critic_observations=critic_obs,
                    actions=actions,
                    rewards=rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards,
                    next_actor_observations=next_actor_obs,
                    next_critic_observations=next_critic_obs,
                    dones=dones.unsqueeze(-1) if dones.dim() == 1 else dones,
                )

    def clear(self) -> None:
        """Clear the buffer by resetting pointers."""
        self.ptr = 0
        self.filled_size = 0

    def save(self, path: str) -> None:
        """
        Save the replay buffer to a file.

        Args:
            path: Path to save the buffer
        """
        save_dict = {
            "actor_obs": self.actor_obs_buf.cpu().numpy(),
            "critic_obs": self.critic_obs_buf.cpu().numpy(),
            "acts": self.acts_buf.cpu().numpy(),
            "rews": self.rews_buf.cpu().numpy(),
            "next_actor_obs": self.next_actor_obs_buf.cpu().numpy(),
            "next_critic_obs": self.next_critic_obs_buf.cpu().numpy(),
            "done": self.done_buf.cpu().numpy(),
            "ptr": self.ptr,
            "filled_size": self.filled_size,
            "num_envs": self.num_envs,
            "size_per_env": self.size_per_env,
            "actor_obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "act_dim": self.act_dim,
        }
        np.savez(path, **save_dict)

    def load(self, path: str) -> None:
        """
        Load the replay buffer from a file.

        Args:
            path: Path to load the buffer from
        """
        data = np.load(path)

        # Check compatibility
        if (
            data["num_envs"] != self.num_envs
            or data["actor_obs_dim"] != self.actor_obs_dim
            or data["critic_obs_dim"] != self.critic_obs_dim
            or data["act_dim"] != self.act_dim
            or data["size_per_env"] != self.size_per_env
        ):
            raise ValueError("Loaded buffer config doesn't match current buffer")

        # Load data with correct shapes
        self.actor_obs_buf = torch.tensor(data["actor_obs"], device=self.device, dtype=self.float_dtype)
        self.critic_obs_buf = torch.tensor(data["critic_obs"], device=self.device, dtype=self.float_dtype)
        self.acts_buf = torch.tensor(data["acts"], device=self.device, dtype=self.float_dtype)
        self.rews_buf = torch.tensor(data["rews"], device=self.device, dtype=self.float_dtype)
        self.next_actor_obs_buf = torch.tensor(data["next_actor_obs"], device=self.device, dtype=self.float_dtype)
        self.next_critic_obs_buf = torch.tensor(data["next_critic_obs"], device=self.device, dtype=self.float_dtype)
        self.done_buf = torch.tensor(data["done"], device=self.device, dtype=self.float_dtype)
        self.ptr = int(data["ptr"])
        self.filled_size = int(data["filled_size"])

    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the buffer state.

        Returns:
            Dictionary with buffer statistics
        """
        return {
            "filled_size": self.filled_size,
            "ptr": self.ptr,
            "capacity": self.size_per_env,
            "fill_ratio": self.filled_size / max(1, self.size_per_env),
            "total_transitions": self.filled_size * self.num_envs,
        }


class NstepEstimatorReplayBuffer(NstepReplayBuffer):
    """
    Extended FastTD3-style Replay Buffer with state estimation support.

    Maintains the efficient (num_envs, size_per_env, feature_dim) storage pattern
    while adding support for estimator observations and robot states.
    """

    def __init__(
        self,
        num_envs: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        estimator_obs_dim: int,
        robot_state_dim: int,
        size_per_env: int,
        device: str = "cuda:0",
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        """
        Initialize the FastTD3-style parallel estimator replay buffer.

        Args:
            num_envs: Number of parallel environments
            actor_obs_dim: Dimension of actor observations
            critic_obs_dim: Dimension of critic observations
            act_dim: Dimension of actions
            estimator_obs_dim: Dimension of estimator inputs
            robot_state_dim: Dimension of robot state (ground truth)
            size_per_env: Maximum size of buffer per environment
            device: Device to store tensors on
            n_steps: Number of steps for n-step returns
            gamma: Discount factor for n-step returns
        """
        super().__init__(
            num_envs=num_envs,
            actor_obs_dim=actor_obs_dim,
            critic_obs_dim=critic_obs_dim,
            act_dim=act_dim,
            size_per_env=size_per_env,
            device=device,
            n_steps=n_steps,
            gamma=gamma,
        )

        self.estimator_obs_dim = estimator_obs_dim
        self.robot_state_dim = robot_state_dim

        # Additional buffers for estimator data
        self.estimator_obs_buf = torch.zeros(
            (num_envs, size_per_env, estimator_obs_dim),
            dtype=self.float_dtype,
            device=device
        )
        self.robot_state_buf = torch.zeros(
            (num_envs, size_per_env, robot_state_dim),
            dtype=self.float_dtype,
            device=device
        )

    @torch.no_grad()
    def store_parallel(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_actor_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        done: torch.Tensor,
        truncated: Optional[torch.Tensor] = None,
        estimator_obs: Optional[torch.Tensor] = None,
        robot_state: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store transitions with estimator data from multiple parallel environments.
        Args:
            Standard replay buffer arguments plus:
            truncated: Truncation flags [num_envs] or [num_envs, 1] (optional)
            estimator_obs: Estimator observations [num_envs, estimator_obs_dim]
            robot_state: Ground truth robot states [num_envs, robot_state_dim]
        """
        # Store standard data
        super().store_parallel(
            actor_obs, critic_obs, act, rew,
            next_actor_obs, next_critic_obs, done, truncated
        )

        # Store estimator-specific data (ptr already updated by parent)
        if estimator_obs is not None:
            # Use the previous ptr position since parent already incremented it
            store_idx = (self.ptr - 1) % self.size_per_env
            self.estimator_obs_buf[:, store_idx] = estimator_obs

        if robot_state is not None:
            store_idx = (self.ptr - 1) % self.size_per_env
            self.robot_state_buf[:, store_idx] = robot_state

    @torch.no_grad()
    def store_parallel_batch(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_actor_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        done: torch.Tensor,
        truncated: torch.Tensor,
        estimator_obs: Optional[torch.Tensor] = None,
        robot_state: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store batch of trajectories with shape [num_envs, trajectory_length, dims].
        This method efficiently stores entire trajectories from expert data without
        looping through individual timesteps.
        Args:
            actor_obs: [num_envs, traj_length, actor_obs_dim]
            critic_obs: [num_envs, traj_length, critic_obs_dim]
            act: [num_envs, traj_length, act_dim]
            rew: [num_envs, traj_length] or [num_envs, traj_length, 1]
            next_actor_obs: [num_envs, traj_length, actor_obs_dim]
            next_critic_obs: [num_envs, traj_length, critic_obs_dim]
            done: [num_envs, traj_length] or [num_envs, traj_length, 1]
            truncated: [num_envs, traj_length] or [num_envs, traj_length, 1] (optional)
            estimator_obs: [num_envs, traj_length, estimator_obs_dim] (optional)
            robot_state: [num_envs, traj_length, robot_state_dim] (optional)
        """
        # Ensure correct shapes for rewards and dones
        if rew.dim() == 3:
            rew = rew.squeeze(-1)  # [num_envs, traj_length]
        if done.dim() == 3:
            done = done.squeeze(-1)  # [num_envs, traj_length]
        if truncated is not None and truncated.dim() == 3:
            truncated = truncated.squeeze(-1)  # [num_envs, traj_length]

        batch_num_envs, traj_length = rew.shape

        # Verify num_envs matches
        if batch_num_envs != self.num_envs:
            raise ValueError(f"Batch num_envs {batch_num_envs} doesn't match buffer num_envs {self.num_envs}")

        # Check if trajectory fits in buffer
        if traj_length > self.size_per_env:
            raise ValueError(
                f"Trajectory length {traj_length} exceeds buffer size_per_env {self.size_per_env}. "
                f"Consider splitting the trajectory or increasing buffer size."
            )

        # Calculate indices for storing
        start_ptr = self.ptr
        end_ptr = start_ptr + traj_length

        if end_ptr <= self.size_per_env:
            # No wraparound - simple slice assignment
            self.actor_obs_buf[:, start_ptr:end_ptr] = actor_obs
            self.critic_obs_buf[:, start_ptr:end_ptr] = critic_obs
            self.acts_buf[:, start_ptr:end_ptr] = act
            self.rews_buf[:, start_ptr:end_ptr] = rew
            self.next_actor_obs_buf[:, start_ptr:end_ptr] = next_actor_obs
            self.next_critic_obs_buf[:, start_ptr:end_ptr] = next_critic_obs
            self.done_buf[:, start_ptr:end_ptr] = done.float()
            self.truncations[:, start_ptr:end_ptr] = truncated

            if estimator_obs is not None:
                self.estimator_obs_buf[:, start_ptr:end_ptr] = estimator_obs
            if robot_state is not None:
                self.robot_state_buf[:, start_ptr:end_ptr] = robot_state
        else:
            # Wraparound case - split into two parts
            first_part_len = self.size_per_env - start_ptr
            second_part_len = traj_length - first_part_len

            # First part: from start_ptr to end of buffer
            self.actor_obs_buf[:, start_ptr:self.size_per_env] = actor_obs[:, :first_part_len]
            self.critic_obs_buf[:, start_ptr:self.size_per_env] = critic_obs[:, :first_part_len]
            self.acts_buf[:, start_ptr:self.size_per_env] = act[:, :first_part_len]
            self.rews_buf[:, start_ptr:self.size_per_env] = rew[:, :first_part_len]
            self.next_actor_obs_buf[:, start_ptr:self.size_per_env] = next_actor_obs[:, :first_part_len]
            self.next_critic_obs_buf[:, start_ptr:self.size_per_env] = next_critic_obs[:, :first_part_len]
            self.done_buf[:, start_ptr:self.size_per_env] = done[:, :first_part_len].float()
            self.truncations[:, start_ptr:self.size_per_env] = truncated[:, :first_part_len]

            if estimator_obs is not None:
                self.estimator_obs_buf[:, start_ptr:self.size_per_env] = estimator_obs[:, :first_part_len]
            if robot_state is not None:
                self.robot_state_buf[:, start_ptr:self.size_per_env] = robot_state[:, :first_part_len]

            # Second part: from start of buffer
            self.actor_obs_buf[:, :second_part_len] = actor_obs[:, first_part_len:]
            self.critic_obs_buf[:, :second_part_len] = critic_obs[:, first_part_len:]
            self.acts_buf[:, :second_part_len] = act[:, first_part_len:]
            self.rews_buf[:, :second_part_len] = rew[:, first_part_len:]
            self.next_actor_obs_buf[:, :second_part_len] = next_actor_obs[:, first_part_len:]
            self.next_critic_obs_buf[:, :second_part_len] = next_critic_obs[:, first_part_len:]
            self.done_buf[:, :second_part_len] = done[:, first_part_len:].float()
            self.truncations[:, :second_part_len] = truncated[:, first_part_len:]

            if estimator_obs is not None:
                self.estimator_obs_buf[:, :second_part_len] = estimator_obs[:, first_part_len:]
            if robot_state is not None:
                self.robot_state_buf[:, :second_part_len] = robot_state[:, first_part_len:]

        # Update pointer with wraparound
        self.ptr = (self.ptr + traj_length) % self.size_per_env

        # Update filled size (capped at size_per_env)
        self.filled_size = min(self.filled_size + traj_length, self.size_per_env)

    @torch.no_grad()
    def sample_batch(self, batch_size: int = 256) -> EstimatorNStepReplayBatch:
        """
        Sample a batch of transitions including estimator data.

        Args:
            batch_size: Total size of the batch to sample

        Returns:
            EstimatorReplayBatch object containing sampled tensors
        """
        if self.filled_size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        # Sample indices using FastTD3 approach
        # env_indices, pos_indices = self._sample_indices(batch_size)
        batch_size_per_env = batch_size // self.num_envs
        remainder = batch_size % self.num_envs

        if self.filled_size >= self.buffer_size:
            # Buffer is full
            current_pos = self.ptr % self.buffer_size

            # Temporarily mark positions as truncated to avoid sampling across episodes
            # This is only needed for n-step returns
            if self.n_steps > 1:
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(
                    self.done_buf[:, current_pos - 1]
                )

            # Sample from full buffer
            indices = torch.randint(
                0,
                self.buffer_size,
                (self.num_envs, batch_size_per_env),
                device=self.device,
            )

        else:
            # Buffer not full - ensure n-step sequence doesn't exceed valid data
            max_start_idx = max(1, self.ptr - self.n_steps + 1) if self.n_steps > 1 else self.ptr
            indices = torch.randint(
                0,
                max_start_idx,
                (self.num_envs, batch_size_per_env),
                device=self.device,
            )

        # Flatten indices and create environment indices
        env_indices = torch.arange(self.num_envs, device=self.device).repeat_interleave(batch_size_per_env)
        pos_indices = indices.flatten()

        # Add remainder samples if needed
        # if remainder > 0:
        #     extra_env_indices = torch.randint(0, self.num_envs, (remainder,), device=self.device)
        #     if self.ptr >= self.buffer_size:
        #         extra_pos_indices = torch.randint(0, self.buffer_size, (remainder,), device=self.device)
        #     else:
        #         max_start_idx = max(1, self.ptr - self.n_steps + 1) if self.n_steps > 1 else self.ptr
        #         extra_pos_indices = torch.randint(0, max_start_idx, (remainder,), device=self.device)

        #     env_indices = torch.cat([env_indices, extra_env_indices])
        #     pos_indices = torch.cat([pos_indices, extra_pos_indices])
        #     raise NotImplementedError("Remainder sampling not implemented in this snippet")

        # Extract starting observations and actions
        actor_obs = self.actor_obs_buf[env_indices, pos_indices]
        critic_obs = self.critic_obs_buf[env_indices, pos_indices]
        actions = self.acts_buf[env_indices, pos_indices]
        estimator_obs = self.estimator_obs_buf[env_indices, pos_indices]
        robot_states = self.robot_state_buf[env_indices, pos_indices]

        # Handle n-step vs 1-step cases
        if self.n_steps > 1:
            # Compute n-step returns and get observations n steps later
            nstep_rewards, next_actor_obs, next_critic_obs, dones, truncs = self._compute_nstep_data(
                env_indices, pos_indices
            )
            rewards = nstep_rewards
        else:
            # Standard 1-step
            next_actor_obs = self.next_actor_obs_buf[env_indices, pos_indices]
            next_critic_obs = self.next_critic_obs_buf[env_indices, pos_indices]
            rewards = self.rews_buf[env_indices, pos_indices]
            dones = self.done_buf[env_indices, pos_indices]
            truncs = self.truncations[env_indices, pos_indices]

        if self.filled_size >= self.buffer_size and self.n_steps > 1:
            # Restore truncations after sampling
            self.truncations[:, current_pos - 1] = curr_truncations

        # Create batch
        return EstimatorNStepReplayBatch(
            actor_observations=actor_obs,
            critic_observations=critic_obs,
            actions=actions,
            rewards=rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards,
            next_actor_observations=next_actor_obs,
            next_critic_observations=next_critic_obs,
            dones=dones.unsqueeze(-1) if dones.dim() == 1 else dones,
            truncated=truncs.unsqueeze(-1) if truncs.dim() == 1 else truncs,
            estimator_observations=estimator_obs,
            robot_states=robot_states,
        )

    @torch.no_grad()
    def sample_batch_with_indices(self, batch_size: int = 256) -> Tuple[EstimatorNStepReplayBatch, torch.Tensor]:
        """
        Sample a batch with indices for prioritized replay.

        Args:
            batch_size: Total size of the batch to sample

        Returns:
            Tuple of (EstimatorReplayBatch, indices)
        """
        if self.filled_size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        # Sample indices using FastTD3 approach
        env_indices, pos_indices = self._sample_indices(batch_size)

        # Extract starting observations and actions
        actor_obs = self.actor_obs_buf[env_indices, pos_indices]
        critic_obs = self.critic_obs_buf[env_indices, pos_indices]
        actions = self.acts_buf[env_indices, pos_indices]
        estimator_obs = self.estimator_obs_buf[env_indices, pos_indices]
        robot_states = self.robot_state_buf[env_indices, pos_indices]

        # Handle n-step vs 1-step cases
        if self.n_steps > 1:
            # Compute n-step returns and get observations n steps later
            nstep_rewards, next_actor_obs, next_critic_obs, dones = self._compute_nstep_data(
                env_indices, pos_indices
            )
            rewards = nstep_rewards
        else:
            # Standard 1-step
            next_actor_obs = self.next_actor_obs_buf[env_indices, pos_indices]
            next_critic_obs = self.next_critic_obs_buf[env_indices, pos_indices]
            rewards = self.rews_buf[env_indices, pos_indices]
            dones = self.done_buf[env_indices, pos_indices]

        # Create batch
        batch = EstimatorNStepReplayBatch(
            actor_observations=actor_obs,
            critic_observations=critic_obs,
            actions=actions,
            rewards=rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards,
            next_actor_observations=next_actor_obs,
            next_critic_observations=next_critic_obs,
            dones=dones.unsqueeze(-1) if dones.dim() == 1 else dones,
            estimator_observations=estimator_obs,
            robot_states=robot_states,
        )

        # Create flat indices for compatibility (env_idx * size_per_env + pos_idx)
        flat_indices = env_indices * self.size_per_env + pos_indices

        return batch, flat_indices

    def mini_batch_generator(self, num_mini_batches, num_epochs=1):
        """
        Generate mini-batches with estimator data.

        Args:
            num_mini_batches: Number of mini-batches to generate
            num_epochs: Number of epochs to generate mini-batches for

        Returns:
            Generator yielding EstimatorReplayBatch objects
        """
        # Calculate total samples and mini-batch size
        total_samples = self.filled_size * self.num_envs
        mini_batch_size = total_samples // num_mini_batches

        for epoch in range(num_epochs):
            # Sample all indices for this epoch using FastTD3 approach
            all_env_indices, all_pos_indices = self._sample_indices(total_samples)

            # Shuffle the indices
            perm = torch.randperm(total_samples, device=self.device)
            all_env_indices = all_env_indices[perm]
            all_pos_indices = all_pos_indices[perm]

            for i in range(num_mini_batches):
                start_idx = i * mini_batch_size
                end_idx = min((i + 1) * mini_batch_size, total_samples)

                env_indices = all_env_indices[start_idx:end_idx]
                pos_indices = all_pos_indices[start_idx:end_idx]

                # Extract starting observations and actions
                actor_obs = self.actor_obs_buf[env_indices, pos_indices]
                critic_obs = self.critic_obs_buf[env_indices, pos_indices]
                actions = self.acts_buf[env_indices, pos_indices]
                estimator_obs = self.estimator_obs_buf[env_indices, pos_indices]
                robot_states = self.robot_state_buf[env_indices, pos_indices]

                # Handle n-step vs 1-step cases
                if self.n_steps > 1:
                    # Compute n-step returns and get observations n steps later
                    nstep_rewards, next_actor_obs, next_critic_obs, dones = self._compute_nstep_data(
                        env_indices, pos_indices
                    )
                    rewards = nstep_rewards
                else:
                    # Standard 1-step
                    next_actor_obs = self.next_actor_obs_buf[env_indices, pos_indices]
                    next_critic_obs = self.next_critic_obs_buf[env_indices, pos_indices]
                    rewards = self.rews_buf[env_indices, pos_indices]
                    dones = self.done_buf[env_indices, pos_indices]

                # Yield batch
                yield EstimatorNStepReplayBatch(
                    actor_observations=actor_obs,
                    critic_observations=critic_obs,
                    actions=actions,
                    rewards=rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards,
                    next_actor_observations=next_actor_obs,
                    next_critic_observations=next_critic_obs,
                    dones=dones.unsqueeze(-1) if dones.dim() == 1 else dones,
                    estimator_observations=estimator_obs,
                    robot_states=robot_states,
                )

    def save(self, path: str) -> None:
        """
        Save the replay buffer to a file.

        Args:
            path: Path to save the buffer
        """
        # Get parent's save dict
        save_dict = {
            "actor_obs": self.actor_obs_buf.cpu().numpy(),
            "critic_obs": self.critic_obs_buf.cpu().numpy(),
            "acts": self.acts_buf.cpu().numpy(),
            "rews": self.rews_buf.cpu().numpy(),
            "next_actor_obs": self.next_actor_obs_buf.cpu().numpy(),
            "next_critic_obs": self.next_critic_obs_buf.cpu().numpy(),
            "done": self.done_buf.cpu().numpy(),
            "ptr": self.ptr,
            "filled_size": self.filled_size,
            "num_envs": self.num_envs,
            "size_per_env": self.size_per_env,
            "actor_obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "act_dim": self.act_dim,
            # Add estimator-specific fields
            "estimator_obs": self.estimator_obs_buf.cpu().numpy(),
            "robot_state": self.robot_state_buf.cpu().numpy(),
            "estimator_obs_dim": self.estimator_obs_dim,
            "robot_state_dim": self.robot_state_dim,
        }
        np.savez(path, **save_dict)

    def load(self, path: str) -> None:
        """
        Load the replay buffer from a file.

        Args:
            path: Path to load the buffer from
        """
        data = np.load(path)

        # Check compatibility
        if (
            data["num_envs"] != self.num_envs
            or data["actor_obs_dim"] != self.actor_obs_dim
            or data["critic_obs_dim"] != self.critic_obs_dim
            or data["act_dim"] != self.act_dim
            or data["estimator_obs_dim"] != self.estimator_obs_dim
            or data["robot_state_dim"] != self.robot_state_dim
            or data["size_per_env"] != self.size_per_env
        ):
            raise ValueError("Loaded buffer config doesn't match current buffer")

        # Load standard data
        super().load(path)

        # Load estimator-specific data
        self.estimator_obs_buf = torch.tensor(
            data["estimator_obs"],
            device=self.device,
            dtype=self.float_dtype
        )
        self.robot_state_buf = torch.tensor(
            data["robot_state"],
            device=self.device,
            dtype=self.float_dtype
        )
