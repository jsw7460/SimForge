"""Per-env time-contiguous trajectory buffer for NPMP rollouts.

Linear (cleared every iteration) — matches the PPO ``RolloutStorage``
pattern. Because the experts that supervise distillation are fixed,
there is no benefit to keeping stale rollout data across gradient
iterations, so we drop the circular-buffer complexity and just refill
each iteration.

Storage layout (all JAX arrays, allocated on the default device)::

    s         : (num_envs, max_steps, D_s)
    x         : (num_envs, max_steps, D_x)
    mu_E      : (num_envs, max_steps, A)
    ep_starts : (num_envs, max_steps)

A single ``write_idx`` advances every ``add`` call (one column written
per step across all envs). Sampling picks ``n_traj`` random
``(env_idx, start_idx)`` pairs and gathers contiguous length-``T``
windows so the encoder's AR(1) chain stays valid within each sample.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from rlworld.imitation.npmp.loss import NPMPBatch

__all__ = ["NPMPBuffer"]


class NPMPBuffer:
    """Linear per-env trajectory buffer keyed by (env, time)."""

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        s_dim: int,
        x_dim: int,
        action_dim: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.action_dim = action_dim

        self._s = jnp.zeros((num_envs, max_steps, s_dim), dtype=dtype)
        self._x = jnp.zeros((num_envs, max_steps, x_dim), dtype=dtype)
        self._mu_E = jnp.zeros((num_envs, max_steps, action_dim), dtype=dtype)
        self._ep_starts = jnp.zeros((num_envs, max_steps), dtype=jnp.bool_)

        self._write_idx = 0

    # ------------------------------------------------------------------
    # Add / clear
    # ------------------------------------------------------------------

    def add(
        self,
        s: jax.Array,  # (num_envs, D_s)
        x: jax.Array,  # (num_envs, D_x)
        mu_E: jax.Array,  # (num_envs, A)
        ep_starts: jax.Array,  # (num_envs,)
    ) -> None:
        """Append one step worth of per-env data at the current write head."""
        if self._write_idx >= self.max_steps:
            raise RuntimeError(
                f"NPMPBuffer is full ({self._write_idx}/{self.max_steps}). Call clear() before adding more transitions."
            )
        self._validate_step_shapes(s, x, mu_E, ep_starts)

        i = self._write_idx
        self._s = self._s.at[:, i].set(s)
        self._x = self._x.at[:, i].set(x)
        self._mu_E = self._mu_E.at[:, i].set(mu_E)
        self._ep_starts = self._ep_starts.at[:, i].set(ep_starts)
        self._write_idx += 1

    def clear(self) -> None:
        """Reset the write head. Storage is overwritten in-place by next adds."""
        self._write_idx = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_filled(self) -> int:
        """Number of time slots currently populated (per env)."""
        return self._write_idx

    @property
    def is_full(self) -> bool:
        return self._write_idx >= self.max_steps

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_trajectories(
        self,
        n_traj: int,
        traj_len: int,
        key: jax.Array,
    ) -> NPMPBatch:
        """Sample ``n_traj`` random length-``traj_len`` time-contiguous windows.

        Each sample is a ``(env_idx, start_idx)`` pair drawn uniformly
        with ``start_idx + traj_len <= num_filled``. Returned tensors
        have leading shape ``(n_traj, traj_len, ...)``.
        """
        if traj_len <= 0:
            raise ValueError(f"traj_len must be positive, got {traj_len}")
        if traj_len > self._write_idx:
            raise ValueError(
                f"traj_len={traj_len} exceeds num_filled={self._write_idx}; "
                "either lower traj_len or run more rollout steps before sampling."
            )

        max_start = self._write_idx - traj_len + 1

        k_env, k_start = jax.random.split(key)
        env_idx = jax.random.randint(k_env, (n_traj,), 0, self.num_envs)
        start_idx = jax.random.randint(k_start, (n_traj,), 0, max_start)

        # (n_traj, traj_len) absolute time indices.
        time_offsets = jnp.arange(traj_len)
        time_idx = start_idx[:, None] + time_offsets[None, :]
        env_idx_b = env_idx[:, None]

        s = self._s[env_idx_b, time_idx]
        x = self._x[env_idx_b, time_idx]
        mu_E = self._mu_E[env_idx_b, time_idx]
        ep_starts = self._ep_starts[env_idx_b, time_idx]

        # The first slot of every sampled window must be treated as an
        # episode boundary for the encoder's AR(1) chain — otherwise the
        # in-buffer ``ep_starts`` flag (which only fires at true env
        # resets) would mis-indicate that mid-rollout slices continue
        # from some earlier z_{t-1}.
        ep_starts = ep_starts.at[:, 0].set(True)

        return NPMPBatch(s=s, x=x, mu_E=mu_E, episode_starts=ep_starts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_step_shapes(
        self,
        s: jax.Array,
        x: jax.Array,
        mu_E: jax.Array,
        ep_starts: jax.Array,
    ) -> None:
        expected = (
            (s, (self.num_envs, self.s_dim), "s"),
            (x, (self.num_envs, self.x_dim), "x"),
            (mu_E, (self.num_envs, self.action_dim), "mu_E"),
            (ep_starts, (self.num_envs,), "ep_starts"),
        )
        for arr, shape, name in expected:
            if arr.shape != shape:
                raise ValueError(f"NPMPBuffer.add: expected {name} shape {shape}, got {tuple(arr.shape)}")
