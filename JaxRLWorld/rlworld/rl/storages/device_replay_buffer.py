"""Replay buffer whose storage lives on the accelerator.

``ReplayBuffer`` keeps its rings in host memory, which is the right
choice when the transitions are born there — a CPU Gymnasium env — and
the wrong one when they are born on the accelerator. With a GPU
simulator the host buffer makes every transition cross twice: down to be
stored, and back up to be sampled. On go2/newton/sac (8192 envs,
``num_steps_per_env=24``, ``num_gradient_steps=200``, batch 8192) that is
roughly 160 MB down and 1.4 GB up per training iteration, none of which
computes anything.

Keeping the rings on device removes both, and the sampled indices with
them: they are drawn where they are used instead of being drawn on the
accelerator and copied back to index host arrays.

What it costs is memory. The same preset holds 5M transitions across
8192 environments, which is about 4 GB of device memory that the
physics no longer has. Vision observations make it far worse. So this
is opt-in per preset (``replay_buffer_device``), and ``ReplayBuffer``
stays the default.

The two must stay interchangeable, and nothing about a wrong sample is
loud — it is a slightly wrong gradient. ``check_replay_buffer_parity``
fills both with identical transitions and compares their batches for
the same indices; run it after touching either.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from rlworld.rl.storages.replay_buffer import ReplayBatch


@partial(jax.jit, donate_argnums=(0,))
def _write_step(buffers: Tuple[jax.Array, ...], index: jax.Array, values: Tuple[jax.Array, ...]):
    """Write one timestep into every ring, in one donated dispatch.

    Same shape as ``RolloutStorage._write_step`` and for the same two
    reasons: one launch instead of eight, and ``donate_argnums`` lets
    XLA write the row where it stands rather than copying a ring the
    size of the whole buffer per step.
    """
    return tuple(buf.at[:, index].set(val) for buf, val in zip(buffers, values))


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _sample_indices(
    key: jax.Array,
    fill: Tuple[jax.Array, jax.Array],
    num_envs: int,
    size_per_env: int,
    n_steps: int,
    batch_size: int,
    full: bool,
) -> Tuple[jax.Array, jax.Array]:
    """Draw (env, position) pairs, on device.

    ``full`` is static — it flips once, when the ring first wraps, so it
    costs one extra compilation for the whole run. ``ptr`` and
    ``filled_size`` are traced: they change every step, and making them
    static would recompile every step.
    """
    ptr, filled_size = fill
    key_env, key_pos = jax.random.split(key)
    env_indices = jax.random.randint(key_env, (batch_size,), 0, num_envs)
    if full:
        max_logical = size_per_env - (n_steps - 1)
        logical_start = jax.random.randint(key_pos, (batch_size,), 0, max_logical)
        positions = (ptr + logical_start) % size_per_env
    else:
        max_start = jnp.maximum(1, filled_size - n_steps + 1)
        positions = jax.random.randint(key_pos, (batch_size,), 0, max_start)
    return env_indices, positions


@partial(jax.jit, static_argnums=(2, 3, 4))
def _gather_batch(
    buffers: Tuple[jax.Array, ...],
    indices: Tuple[jax.Array, jax.Array],
    size_per_env: int,
    n_steps: int,
    gamma: float,
) -> ReplayBatch:
    """Gather a batch, n-step returns included, in one program.

    A line-for-line port of ``ReplayBuffer._compute_nstep_data``; the
    semantics are documented there, and the parity gate is what keeps
    the two honest. Everything it does — ``arange``, ``cumsum``,
    ``argmax``, ``where``, advanced indexing — has the same meaning in
    JAX, so the port is mechanical rather than a reinterpretation.
    """
    actor_obs_buf, critic_obs_buf, acts_buf, rews_buf, next_actor_buf, next_critic_buf, term_buf, trunc_buf = buffers
    env_indices, start_positions = indices

    actor_obs = actor_obs_buf[env_indices, start_positions]
    critic_obs = critic_obs_buf[env_indices, start_positions]
    actions = acts_buf[env_indices, start_positions]

    if n_steps == 1:
        return ReplayBatch(
            actor_observations=actor_obs,
            critic_observations=critic_obs,
            actions=actions,
            rewards=rews_buf[env_indices, start_positions],
            next_actor_observations=next_actor_buf[env_indices, start_positions],
            next_critic_observations=next_critic_buf[env_indices, start_positions],
            terminated=term_buf[env_indices, start_positions],
            truncated=trunc_buf[env_indices, start_positions],
            gamma_power=jnp.full((env_indices.shape[0], 1), gamma, jnp.float32),
        )

    batch_size = env_indices.shape[0]
    seq = jnp.arange(n_steps)
    all_positions = (start_positions[:, None] + seq[None, :]) % size_per_env
    env_expanded = jnp.broadcast_to(env_indices[:, None], (batch_size, n_steps))

    all_rewards = rews_buf[env_expanded, all_positions, 0]
    all_terminated = term_buf[env_expanded, all_positions, 0]
    all_truncated = trunc_buf[env_expanded, all_positions, 0]
    all_boundaries = all_terminated + all_truncated

    # Include the reward at the boundary step, exclude everything after.
    cumsum = jnp.cumsum(all_boundaries, axis=1)
    shifted = jnp.concatenate([jnp.zeros((batch_size, 1)), cumsum[:, :-1]], axis=1)
    done_masks = (shifted == 0).astype(jnp.float32)

    discounts = jnp.power(gamma, jnp.arange(n_steps).astype(jnp.float32))
    nstep_rewards = (all_rewards * done_masks * discounts[None, :]).sum(axis=1, keepdims=True)

    has_boundary = all_boundaries.sum(axis=1) > 0
    first_boundary = jnp.argmax(all_boundaries > 0, axis=1)
    effective_n = jnp.where(has_boundary, first_boundary + 1, n_steps)
    last_used = (start_positions + effective_n - 1) % size_per_env

    return ReplayBatch(
        actor_observations=actor_obs,
        critic_observations=critic_obs,
        actions=actions,
        rewards=nstep_rewards,
        next_actor_observations=next_actor_buf[env_indices, last_used],
        next_critic_observations=next_critic_buf[env_indices, last_used],
        terminated=term_buf[env_indices, last_used],
        truncated=trunc_buf[env_indices, last_used],
        gamma_power=jnp.power(gamma, effective_n.astype(jnp.float32))[:, None],
    )


@dataclass(frozen=True)
class TracedSampler:
    """A sampler that ``jit`` can treat as a constant.

    The obvious way to hand a scan its sampling step is a closure, but
    equinox keeps every non-array argument static and compares them by
    value: a fresh closure each call is a fresh static argument, so the
    compilation cache misses every time and the scan recompiles instead
    of running. Frozen and built only from the numbers that determine
    the program, this compares equal across calls; the arrays it reads
    travel separately, as ``state``, and stay traced.
    """

    num_envs: int
    size_per_env: int
    n_steps: int
    batch_size: int
    full: bool
    gamma: float

    def __call__(self, state: Tuple[Any, Any], key: jax.Array) -> ReplayBatch:
        buffers, fill = state
        indices = _sample_indices(key, fill, self.num_envs, self.size_per_env, self.n_steps, self.batch_size, self.full)
        return _gather_batch(buffers, indices, self.size_per_env, self.n_steps, self.gamma)


class DeviceReplayBuffer:
    """Device-resident counterpart of :class:`ReplayBuffer`.

    Same interface and same semantics; see the module docstring for when
    to prefer it and what it costs.
    """

    #: Everything a sample touches lives on the device, so a whole
    #: update-to-data loop can be one traced program (see
    #: :meth:`batch_sampler`).
    supports_traced_sampling = True

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
        self.num_envs = num_envs
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
        self.size_per_env = size_per_env
        self.total_size = num_envs * size_per_env
        self.n_steps = n_steps
        self.gamma = gamma
        self.key = jax.random.PRNGKey(seed)

        shape = (num_envs, size_per_env)
        self.buffers = (
            jnp.zeros(shape + (actor_obs_dim,), jnp.float32),
            jnp.zeros(shape + (critic_obs_dim,), jnp.float32),
            jnp.zeros(shape + (act_dim,), jnp.float32),
            jnp.zeros(shape + (1,), jnp.float32),
            jnp.zeros(shape + (actor_obs_dim,), jnp.float32),
            jnp.zeros(shape + (critic_obs_dim,), jnp.float32),
            jnp.zeros(shape + (1,), jnp.float32),
            jnp.zeros(shape + (1,), jnp.float32),
        )

        self.ptr = 0
        self.filled_size = 0

    @property
    def size(self) -> int:
        return self.filled_size * self.num_envs

    @property
    def max_size(self) -> int:
        return self.total_size

    @staticmethod
    def _column(x: jax.Array, width: int) -> jax.Array:
        """A per-env field as ``[num_envs, width]``, scalars widened."""
        return x.reshape(x.shape[0], width).astype(jnp.float32)

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
        """Store one timestep for every environment. Nothing leaves the device."""
        values = (
            self._column(actor_obs, self.actor_obs_dim),
            self._column(critic_obs, self.critic_obs_dim),
            self._column(act, self.act_dim),
            self._column(rew, 1),
            self._column(next_actor_obs, self.actor_obs_dim),
            self._column(next_critic_obs, self.critic_obs_dim),
            self._column(terminated, 1),
            self._column(truncated, 1),
        )
        self.buffers = _write_step(self.buffers, jnp.asarray(self.ptr), values)
        self.ptr = (self.ptr + 1) % self.size_per_env
        self.filled_size = min(self.filled_size + 1, self.size_per_env)

    def sample_batch(self, batch_size: int) -> ReplayBatch:
        """Sample a batch. Indices are drawn on device and stay there."""
        if self.filled_size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        if self.filled_size < self.n_steps:
            raise ValueError(
                f"Cannot sample n-step ({self.n_steps}) windows from a buffer "
                f"with only {self.filled_size} filled steps per env; increase "
                f"learning_starts so collection covers at least n_steps."
            )

        self.key, subkey = jax.random.split(self.key)
        indices = _sample_indices(
            subkey,
            (jnp.asarray(self.ptr), jnp.asarray(self.filled_size)),
            self.num_envs,
            self.size_per_env,
            self.n_steps,
            batch_size,
            self.filled_size >= self.size_per_env,
        )
        return _gather_batch(self.buffers, indices, self.size_per_env, self.n_steps, self.gamma)

    def batch_sampler(self, batch_size: int) -> Tuple["TracedSampler", Tuple[Any, Any]]:
        """A sampler and its state, for sampling inside a traced program.

        Driving ``num_gradient_steps`` updates from Python costs three
        dispatches a pass and a gap between each; handing this to
        ``lax.scan`` makes the whole loop one program.

        Returns the sampler — hashable, so the compilation cache hits —
        and the arrays it reads. Nothing is stored while updating, so
        the rings and fill counters are snapshotted here rather than
        carried through the scan. The caller owns the key and must
        thread it, so consecutive updates see different batches.
        """
        sampler = TracedSampler(
            num_envs=self.num_envs,
            size_per_env=self.size_per_env,
            n_steps=self.n_steps,
            batch_size=batch_size,
            full=self.filled_size >= self.size_per_env,
            gamma=self.gamma,
        )
        state = (self.buffers, (jnp.asarray(self.ptr), jnp.asarray(self.filled_size)))
        return sampler, state

    def get_recent_actions(self, n: int) -> jax.Array:
        """The most recent ``n`` actions, for action-distribution logging."""
        if self.filled_size == 0:
            return jnp.zeros((0, self.act_dim))
        steps = min(max(1, n // self.num_envs), self.filled_size)
        start = (self.ptr - steps) % self.size_per_env
        rows = (start + jnp.arange(steps)) % self.size_per_env
        return self.buffers[2][:, rows].reshape(-1, self.act_dim)[:n]

    def state_dict(self) -> Dict[str, Any]:
        return {"ptr": self.ptr, "filled_size": self.filled_size}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ptr = state["ptr"]
        self.filled_size = state["filled_size"]
