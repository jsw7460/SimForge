from functools import partial
from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(1, 2, 3))
def _split_batch(flat: jax.Array, a: int, c: int, k: int) -> Tuple[jax.Array, ...]:
    """Cut the uploaded batch back into its fields, in one program.

    Slicing outside ``jit`` is a launch per slice, and there are nine of
    them per gradient step — which for an off-policy algorithm is nine
    per environment step. Compiled, the nine cuts cost one.
    """
    o = [0, a, a + c, a + c + k, a + c + k + 1]
    o += [o[-1] + a, o[-1] + a + c, o[-1] + a + c + 1, o[-1] + a + c + 2, o[-1] + a + c + 3]
    return tuple(flat[:, o[i] : o[i + 1]] for i in range(9))


@jax.jit
def _pack_transition(*fields: jax.Array) -> jax.Array:
    """Lay one step's fields out side by side, ready for a single copy.

    Every field of a transition has to reach host storage, and a separate
    copy per field means a separate transfer per field. Under ``jit`` the
    whole concatenation is one program, so the step costs one copy
    instead of eight. Scalars-per-env arrive as ``[num_envs]`` and are
    widened to ``[num_envs, 1]`` so they line up with the vectors.
    """
    columns = [f.reshape(f.shape[0], -1).astype(jnp.float32) for f in fields]
    return jnp.concatenate(columns, axis=1)


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
        seed: int = 0,
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
            seed: Seed for this buffer's own index sampler. The storage
                lives in host memory and is indexed with NumPy, so the
                indices are drawn on the host too — drawing them on the
                accelerator instead would mean a blocking transfer per
                sampled batch, which for an off-policy algorithm is one
                per environment step.
        """
        self.num_envs = num_envs
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
        self.size_per_env = size_per_env
        self.total_size = num_envs * size_per_env
        self.n_steps = n_steps
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        # NumPy buffers for fast in-place writes: [num_envs, size_per_env, dim]
        self.actor_obs_buf = np.zeros((num_envs, size_per_env, actor_obs_dim), dtype=np.float32)
        self.critic_obs_buf = np.zeros((num_envs, size_per_env, critic_obs_dim), dtype=np.float32)
        self.next_actor_obs_buf = np.zeros((num_envs, size_per_env, actor_obs_dim), dtype=np.float32)
        self.next_critic_obs_buf = np.zeros((num_envs, size_per_env, critic_obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((num_envs, size_per_env, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((num_envs, size_per_env, 1), dtype=np.float32)
        self.terminated_buf = np.zeros((num_envs, size_per_env, 1), dtype=np.float32)
        self.truncated_buf = np.zeros((num_envs, size_per_env, 1), dtype=np.float32)

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
        **kwargs,
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
        # The storage is host memory, so every field has to come down
        # from the accelerator. Coming down one field at a time is eight
        # transfers per environment step — for an off-policy algorithm,
        # eight per step of training. Concatenating first makes it one;
        # the split below costs nothing, being views into that array.
        packed = _pack_transition(
            actor_obs, critic_obs, act, rew, next_actor_obs, next_critic_obs, terminated, truncated
        )
        flat = np.asarray(packed)

        a, c, k = self.actor_obs_dim, self.critic_obs_dim, self.act_dim
        cuts = np.cumsum([a, c, k, 1, a, c, 1])
        (
            actor_obs_np,
            critic_obs_np,
            act_np,
            rew_np,
            next_actor_obs_np,
            next_critic_obs_np,
            terminated_np,
            truncated_np,
        ) = np.split(flat, cuts, axis=1)

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
            Tuple of (all shapes verified against the [num_envs,
            size_per_env, dim] buffer layout; B = batch_size):

            - ``nstep_rewards``         [B, 1] — discounted reward sum,
              truncated at the first episode boundary in the window.
            - ``final_next_actor_obs``  [B, actor_obs_dim] — stored
              next_obs of the *last used* transition (= first boundary
              step when one exists, else step n-1). For a truncated
              transition this is the pre-reset observation the runner
              stored.
            - ``final_next_critic_obs`` [B, critic_obs_dim] — same
              position as above.
            - ``final_terminated``      [B, 1] — terminated flag of the
              last used transition. The trailing 1 comes from the
              ``[num_envs, size_per_env, 1]`` buffer: indexing axes 0/1
              with two [B] integer arrays leaves axis 2 intact.
            - ``final_truncated``       [B, 1] — same as above.
            - ``gamma_power``           [B, 1] — per-sample
              ``gamma ** effective_n`` for the bootstrap term.

            All [B, 1] shapes broadcast directly against the critic's
            [B, 1] Q outputs in the loss; do NOT squeeze them.
        """
        batch_size = env_indices.shape[0]

        # Create sequence offsets [0, 1, ..., n_steps-1]
        seq_offsets = np.arange(self.n_steps)

        # Calculate all positions for n-step sequence: [batch_size, n_steps]
        all_positions = (start_positions[:, None] + seq_offsets[None, :]) % self.size_per_env

        # Expand env_indices to match: [batch_size, n_steps]
        env_indices_expanded = np.broadcast_to(env_indices[:, None], (batch_size, self.n_steps))

        # Gather rewards, terminated, and truncated for all steps.
        # Buffers are [num_envs, size_per_env, 1]; the explicit 0 on the
        # last axis drops the trailing dim -> each is [batch_size, n_steps].
        all_rewards = self.rews_buf[env_indices_expanded, all_positions, 0]
        all_terminated = self.terminated_buf[env_indices_expanded, all_positions, 0]
        all_truncated = self.truncated_buf[env_indices_expanded, all_positions, 0]

        # Episode boundary = terminated OR truncated
        all_boundaries = all_terminated + all_truncated

        # Mask: include reward at boundary step, exclude after
        cumsum_boundaries = np.cumsum(all_boundaries, axis=1)
        cumsum_boundaries_shifted = np.concatenate([np.zeros((batch_size, 1)), cumsum_boundaries[:, :-1]], axis=1)
        done_masks = (cumsum_boundaries_shifted == 0).astype(np.float32)

        # Discount factors [γ^0, γ^1, ..., γ^(n-1)]
        discounts = np.power(self.gamma, np.arange(self.n_steps).astype(np.float32))

        # Apply mask and discounts
        masked_rewards = all_rewards * done_masks
        discounted_rewards = masked_rewards * discounts[None, :]

        # Sum to get n-step reward
        nstep_rewards = discounted_rewards.sum(axis=1, keepdims=True)

        # Find effective n (number of steps actually used). All [B].
        has_boundary = all_boundaries.sum(axis=1) > 0
        first_boundary_idx = np.argmax(all_boundaries > 0, axis=1)
        effective_n = np.where(has_boundary, first_boundary_idx + 1, self.n_steps)

        # Position of the last used transition: the first boundary step
        # when one exists, else start + n_steps - 1. Shape [B].
        last_used_position = (start_positions + effective_n - 1) % self.size_per_env

        # Get next_obs from the last used transition.
        # Buffer [num_envs, size_per_env, obs_dim] indexed on axes 0/1
        # with two [B] integer arrays -> [B, obs_dim].
        final_next_actor_obs = self.next_actor_obs_buf[env_indices, last_used_position]
        final_next_critic_obs = self.next_critic_obs_buf[env_indices, last_used_position]

        # Terminal flags from the last used transition.
        # Buffer [num_envs, size_per_env, 1]: axis 2 is not indexed, so
        # the trailing 1 survives -> [B, 1]. Matches the critic's [B, 1]
        # Q output in the loss without further reshaping.
        final_terminated = self.terminated_buf[env_indices, last_used_position]
        final_truncated = self.truncated_buf[env_indices, last_used_position]

        # gamma^effective_n for the bootstrap term: [B] -> [B, 1].
        gamma_power = np.power(self.gamma, effective_n.astype(np.float32))[:, None]

        return (
            nstep_rewards,
            final_next_actor_obs,
            final_next_critic_obs,
            final_terminated,
            final_truncated,
            gamma_power,
        )

    def sample_batch(self, batch_size: int) -> ReplayBatch:
        """
        Sample a batch of transitions from the buffer.
        Computes n-step returns if n_steps > 1.

        Indices come from this buffer's own NumPy generator (see
        ``seed``), not from a JAX key: the storage is host memory indexed
        with NumPy, so drawing indices on the accelerator would only add
        a blocking transfer to bring them back.

        Args:
            batch_size: Size of the batch to sample

        Returns:
            ReplayBatch object with n-step data (JAX arrays)
        """
        if self.filled_size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        if self.filled_size < self.n_steps:
            # An n-step window sampled now would read zero-initialized
            # slots past the filled region (silent garbage targets).
            raise ValueError(
                f"Cannot sample n-step ({self.n_steps}) windows from a buffer "
                f"with only {self.filled_size} filled steps per env; increase "
                f"learning_starts so collection covers at least n_steps."
            )

        env_indices = self.rng.integers(0, self.num_envs, size=batch_size)

        # Sample position indices
        if self.filled_size >= self.size_per_env:
            max_logical = self.size_per_env - (self.n_steps - 1)
            logical_start = self.rng.integers(0, max_logical, size=batch_size)
            pos_indices = (self.ptr + logical_start) % self.size_per_env
        else:
            max_start_idx = max(1, self.filled_size - self.n_steps + 1)
            pos_indices = self.rng.integers(0, max_start_idx, size=batch_size)

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

        # Up in one copy, for the same reason it comes down in one: nine
        # separate uploads is nine per gradient step, and an off-policy
        # algorithm takes a gradient step per environment step. The
        # slices below are views into the uploaded array.
        flat = jnp.asarray(
            np.concatenate(
                [
                    actor_obs,
                    critic_obs,
                    actions,
                    nstep_rewards,
                    next_actor_obs,
                    next_critic_obs,
                    terminated,
                    truncated,
                    gamma_power,
                ],
                axis=1,
                dtype=np.float32,
            )
        )
        return ReplayBatch(*_split_batch(flat, self.actor_obs_dim, self.critic_obs_dim, self.act_dim))

    def get_recent_actions(self, n: int) -> jax.Array:
        """Get the most recent n actions from the buffer."""
        if self.filled_size == 0:
            return jnp.zeros((0, self.act_dim))

        n = min(n, self.filled_size * self.num_envs)

        if self.filled_size >= self.size_per_env:
            recent_pos = (self.ptr - 1 - np.arange(min(n // self.num_envs, self.size_per_env))) % self.size_per_env
        else:
            recent_pos = np.arange(self.filled_size - 1, -1, -1)[: n // self.num_envs]

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
