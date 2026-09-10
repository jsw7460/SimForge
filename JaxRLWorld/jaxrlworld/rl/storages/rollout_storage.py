from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


@partial(jax.jit, donate_argnums=(0,))
def _write_step(buffers: tuple[jax.Array, ...], index: jax.Array, values: tuple[jax.Array, ...]):
    """Write one timestep into every buffer, in place, in one dispatch.

    Both properties matter, and neither is free.

    ONE dispatch: written as ten separate ``at[].set()`` statements
    outside ``jit``, this is ten XLA programs per environment step, each
    paying the launch overhead for a write of a single row.

    IN PLACE: ``at[].set()`` is functional. Outside ``jit`` there is
    nothing to alias the result onto the input, so each call allocates a
    fresh ``(num_steps, num_envs, ...)`` buffer and copies the whole
    thing — the cost of recording one step scales with the length of the
    entire rollout, and ten copies per step churn the allocator hard
    enough to show up as periodic multi-millisecond stalls.
    ``donate_argnums`` lets XLA alias the output onto the donated input
    and write the row where it stands.

    ``index`` is traced, not a Python int, so the step counter does not
    retrigger compilation on every step of the rollout.

    Donation invalidates the caller's references: the buffers passed in
    are dead once this returns, and using one raises rather than reading
    stale data. The only caller reassigns all ten immediately.
    """
    return tuple(jax.tree.map(lambda b, v: b.at[index].set(v), buffer, value) for buffer, value in zip(buffers, values))


StepLayout = tuple[tuple[str, int, int], ...]
"""Column layout of a packed step: ``(name, start, width)`` per field.

The torch->jax bridge carries every vector of a step in ONE ``(num_envs,
total)`` float32 tensor — actor and critic observation, reward, the
flags as 0/1, the terminal critic observation and the masks when the
step has them — because each separate conversion is its own copy and
its own dispatch. Names: ``actor``, ``critic``, ``reward``,
``terminated``, ``truncated``, and optionally ``final_critic``,
``bootstrap_mask``, ``trunc_no_reset``. Static, so it is part of the
compiled program's signature.
"""


def layout_column(packed: jax.Array, layout: StepLayout, name: str) -> jax.Array | None:
    """The named field of a packed step, or ``None`` if the step lacks it.

    Width-1 fields come back as ``(num_envs,)``. Inside ``jit`` these
    slices are free; outside they would each be a dispatch.
    """
    for field, start, width in layout:
        if field == name:
            return packed[:, start] if width == 1 else packed[:, start : start + width]
    return None


def _record_impl(
    buffers: tuple[jax.Array, ...],
    index: jax.Array,
    fields: tuple[jax.Array, ...],
    rewards: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    last_dones: jax.Array | None,
    bootstrap_values: jax.Array | None,
    bootstrap_mask: jax.Array | None,
    trunc_no_reset: jax.Array | None,
    gamma: float,
    recompute_gae: bool,
):
    """Everything ``PPO.process_env_step`` does to a step, as one program.

    The flag casts, ``dones``, the timeout bootstrap bonus, the
    episode-start row and the row write used to be a dozen separate
    eager JAX ops around :func:`_write_step`; each is its own XLA
    program, and the launch overhead of a dozen of them was most of
    what recording a step cost. The critic forward that produces
    ``bootstrap_values`` stays its own compiled call (it is a model).

    ``terminated`` / ``truncated`` / the masks may arrive as bool, as
    the uint8 the bridge used to carry, or as the 0/1 floats of a packed
    step; ``!= 0`` reads all three. Traced by :func:`_record_step` and
    :func:`_record_packed`, which donate the buffers and the row counter
    as :func:`_write_step` does.
    """
    terminated = terminated != 0
    truncated = truncated != 0
    dones = terminated | truncated
    mask = (truncated & ~terminated) if bootstrap_mask is None else (bootstrap_mask != 0)
    if recompute_gae:
        trunc_mask = mask if trunc_no_reset is None else (mask | (trunc_no_reset != 0))
    else:
        trunc_mask = None
        if bootstrap_values is not None:
            rewards = rewards + mask.astype(rewards.dtype) * (gamma * bootstrap_values)
    episode_starts = jnp.zeros_like(dones) if last_dones is None else (last_dones != 0)

    actor_obs, critic_obs, actions, values, log_probs, mu, sigma = fields
    row = (actor_obs, critic_obs, actions, rewards, dones, episode_starts, values, log_probs, mu, sigma, trunc_mask)
    written = tuple(
        buffer if value is None else jax.tree.map(lambda b, v: b.at[index].set(v), buffer, value)
        for buffer, value in zip(buffers, row)
    )
    return written, index + 1, dones


@partial(jax.jit, donate_argnums=(0, 1), static_argnames=("gamma", "recompute_gae"))
def _record_step(
    buffers: tuple[jax.Array, ...],
    index: jax.Array,
    fields: tuple[jax.Array, ...],
    rewards: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    last_dones: jax.Array | None,
    bootstrap_values: jax.Array | None,
    bootstrap_mask: jax.Array | None,
    trunc_no_reset: jax.Array | None,
    gamma: float,
    recompute_gae: bool,
):
    """:func:`_record_impl` on separately converted step tensors."""
    return _record_impl(
        buffers,
        index,
        fields,
        rewards,
        terminated,
        truncated,
        last_dones,
        bootstrap_values,
        bootstrap_mask,
        trunc_no_reset,
        gamma,
        recompute_gae,
    )


@partial(jax.jit, donate_argnums=(0, 1), static_argnames=("layout", "gamma", "recompute_gae"))
def _record_packed(
    buffers: tuple[jax.Array, ...],
    index: jax.Array,
    fields: tuple[jax.Array, ...],
    packed: jax.Array,
    last_dones: jax.Array | None,
    bootstrap_values: jax.Array | None,
    layout: StepLayout,
    gamma: float,
    recompute_gae: bool,
):
    """:func:`_record_impl` on a packed step (see :data:`StepLayout`).

    Also returns the step's actor and critic observation columns, so the
    next policy forward gets them without a slice dispatch of its own.
    """
    written, index, dones = _record_impl(
        buffers,
        index,
        fields,
        layout_column(packed, layout, "reward"),
        layout_column(packed, layout, "terminated"),
        layout_column(packed, layout, "truncated"),
        last_dones,
        bootstrap_values,
        layout_column(packed, layout, "bootstrap_mask"),
        layout_column(packed, layout, "trunc_no_reset"),
        gamma,
        recompute_gae,
    )
    return written, index, dones, layout_column(packed, layout, "actor"), layout_column(packed, layout, "critic")


ObsShape = tuple[int, ...] | dict[str, tuple[int, ...]]
"""One group's per-env shape, or a dict of them when the policy reads
several observation groups (a state vector plus one or more images)."""


def _obs_zeros(shape: ObsShape, prefix: tuple[int, ...]) -> jax.Array | dict[str, jax.Array]:
    """Allocate a buffer per group, each with ``prefix`` in front."""
    if isinstance(shape, dict):
        return {group: jnp.zeros(prefix + group_shape) for group, group_shape in shape.items()}
    return jnp.zeros(prefix + shape)


def _obs_reshape(obs: jax.Array | dict, shape: ObsShape, prefix: tuple[int, ...]):
    """Reshape every group to ``prefix + its own shape``."""
    if isinstance(shape, dict):
        return {group: obs[group].reshape(prefix + shape[group]) for group in shape}
    return obs.reshape(prefix + shape)


class RolloutBatch(NamedTuple):
    """Batch of rollout data for PPO update."""

    actor_observations: jax.Array
    critic_observations: jax.Array
    actions: jax.Array
    values: jax.Array
    advantages: jax.Array
    returns: jax.Array
    old_log_probs: jax.Array
    old_mu: jax.Array
    old_sigma: jax.Array


class Transition(NamedTuple):
    """Single transition data."""

    actor_obs: jax.Array
    critic_obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    values: jax.Array
    log_probs: jax.Array
    mu: jax.Array
    sigma: jax.Array
    episode_starts: jax.Array


class RolloutStorage:
    """Rollout storage with pre-allocated device buffers.

    Each per-step field is held as a single ``(num_steps, num_envs, ...)``
    JAX array allocated once at construction time. Adding a transition
    writes one row into every buffer through ``_write_step``, a single
    donated ``jit`` that updates them in place; clearing the rollout just
    resets the step counter, so the buffers themselves are reused across
    iterations and no ``jnp.stack`` happens at the boundary between
    collection and learning.

    Public API (PPO / PPO-DR3):
        - add_transition(...): record one timestep's data
        - compute_returns(...): GAE → fills ``advantages``/``returns``
        - normalize_advantages(): rsl_rl-style per-rollout normalization
        - get_flat_observations(): flatten obs for normalizer updates
        - get_flat_actions(): flatten actions for action-stat logging
        - get_flat_batch() + get_minibatch_indices(...): the update loop's
          data — one flat batch plus shuffled index rows into it
        - clear(): reset for the next rollout (no reallocation)
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        actor_obs_shape: ObsShape,
        critic_obs_shape: ObsShape,
        action_shape: tuple[int, ...],
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.actor_obs_shape = actor_obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.action_shape = action_shape

        self.step = 0
        # Device-side twin of ``step`` for ``record_step``: a traced row
        # index that never needs a host->device copy per step.
        self._index = jnp.zeros((), dtype=jnp.int32)
        self._allocate_buffers()

    # ---------------------------------------------------------------- alloc

    def _allocate_buffers(self) -> None:
        T, N = self.num_steps, self.num_envs
        self.actor_obs = _obs_zeros(self.actor_obs_shape, (T, N))
        self.critic_obs = _obs_zeros(self.critic_obs_shape, (T, N))
        self.actions = jnp.zeros((T, N) + self.action_shape)
        self.rewards = jnp.zeros((T, N))
        self.dones = jnp.zeros((T, N), dtype=jnp.bool_)
        self.episode_starts = jnp.zeros((T, N), dtype=jnp.bool_)
        self.values = jnp.zeros((T, N))
        self.log_probs = jnp.zeros((T, N))
        self.mu = jnp.zeros((T, N) + self.action_shape)
        self.sigma = jnp.zeros((T, N) + self.action_shape)
        # Truncation-without-reset mask, written by PPO.process_env_step only
        # when the per-epoch-GAE update path is active (stays all-False and
        # unread otherwise). Marks steps whose reward the update replaces
        # with V(s_t) and where GAE cuts (time-outs + command resamples).
        self.trunc_masks = jnp.zeros((T, N), dtype=jnp.bool_)
        # Filled by compute_returns()
        self.advantages: jax.Array | None = None
        self.returns: jax.Array | None = None

    # ------------------------------------------------------------ add/clear

    def add_transition(
        self,
        actor_obs: jax.Array,
        critic_obs: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        episode_starts: jax.Array,
        values: jax.Array,
        log_probs: jax.Array,
        mu: jax.Array,
        sigma: jax.Array,
    ) -> None:
        if self.step >= self.num_steps:
            raise RuntimeError("Storage overflow.")

        (
            self.actor_obs,
            self.critic_obs,
            self.actions,
            self.rewards,
            self.dones,
            self.episode_starts,
            self.values,
            self.log_probs,
            self.mu,
            self.sigma,
        ) = _write_step(
            (
                self.actor_obs,
                self.critic_obs,
                self.actions,
                self.rewards,
                self.dones,
                self.episode_starts,
                self.values,
                self.log_probs,
                self.mu,
                self.sigma,
            ),
            jnp.asarray(self.step),
            (
                actor_obs,
                critic_obs,
                actions,
                rewards,
                dones,
                episode_starts,
                values,
                log_probs,
                mu,
                sigma,
            ),
        )
        self.step += 1

    def record_step(
        self,
        actor_obs: jax.Array,
        critic_obs: jax.Array,
        actions: jax.Array,
        values: jax.Array,
        log_probs: jax.Array,
        mu: jax.Array,
        sigma: jax.Array,
        rewards: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        last_dones: jax.Array | None,
        bootstrap_values: jax.Array | None,
        bootstrap_mask: jax.Array | None,
        trunc_no_reset: jax.Array | None,
        gamma: float,
        recompute_gae: bool,
    ) -> jax.Array:
        """Record one step through :func:`_record_step`; returns ``dones``.

        The reward written is ``rewards`` plus the timeout bootstrap bonus
        when ``bootstrap_values`` is given (and ``recompute_gae`` is off);
        with ``recompute_gae`` the truncation mask row is written instead.
        """
        if self.step >= self.num_steps:
            raise RuntimeError("Storage overflow.")
        buffers = (
            self.actor_obs,
            self.critic_obs,
            self.actions,
            self.rewards,
            self.dones,
            self.episode_starts,
            self.values,
            self.log_probs,
            self.mu,
            self.sigma,
            self.trunc_masks,
        )
        (
            (
                self.actor_obs,
                self.critic_obs,
                self.actions,
                self.rewards,
                self.dones,
                self.episode_starts,
                self.values,
                self.log_probs,
                self.mu,
                self.sigma,
                self.trunc_masks,
            ),
            self._index,
            dones,
        ) = _record_step(
            buffers,
            self._index,
            (actor_obs, critic_obs, actions, values, log_probs, mu, sigma),
            rewards,
            terminated,
            truncated,
            last_dones,
            bootstrap_values,
            bootstrap_mask,
            trunc_no_reset,
            gamma=gamma,
            recompute_gae=recompute_gae,
        )
        self.step += 1
        return dones

    def _buffer_tuple(self) -> tuple:
        return (
            self.actor_obs,
            self.critic_obs,
            self.actions,
            self.rewards,
            self.dones,
            self.episode_starts,
            self.values,
            self.log_probs,
            self.mu,
            self.sigma,
            self.trunc_masks,
        )

    def _set_buffer_tuple(self, buffers: tuple) -> None:
        (
            self.actor_obs,
            self.critic_obs,
            self.actions,
            self.rewards,
            self.dones,
            self.episode_starts,
            self.values,
            self.log_probs,
            self.mu,
            self.sigma,
            self.trunc_masks,
        ) = buffers

    def record_step_packed(
        self,
        actor_obs: jax.Array,
        critic_obs: jax.Array,
        actions: jax.Array,
        values: jax.Array,
        log_probs: jax.Array,
        mu: jax.Array,
        sigma: jax.Array,
        packed: jax.Array,
        layout: StepLayout,
        last_dones: jax.Array | None,
        bootstrap_values: jax.Array | None,
        gamma: float,
        recompute_gae: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """:meth:`record_step` from a packed step; returns ``(dones,
        next_actor_vector, next_critic_vector)``."""
        if self.step >= self.num_steps:
            raise RuntimeError("Storage overflow.")
        buffers, self._index, dones, next_actor, next_critic = _record_packed(
            self._buffer_tuple(),
            self._index,
            (actor_obs, critic_obs, actions, values, log_probs, mu, sigma),
            packed,
            last_dones,
            bootstrap_values,
            layout=layout,
            gamma=gamma,
            recompute_gae=recompute_gae,
        )
        self._set_buffer_tuple(buffers)
        self.step += 1
        return dones, next_actor, next_critic

    def clear(self) -> None:
        """Reset for next rollout. Buffers are reused; advantages/returns dropped."""
        self.step = 0
        self._index = jnp.zeros((), dtype=jnp.int32)
        self.advantages = None
        self.returns = None

    # ------------------------------------------------------ GAE / advantage

    def compute_returns(
        self,
        last_values: jax.Array,
        last_dones: jax.Array,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and returns."""
        advantages, returns = compute_gae(
            rewards=self.rewards,
            values=self.values,
            episode_starts=self.episode_starts,
            last_values=last_values,
            last_dones=last_dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self.advantages = advantages
        self.returns = returns

    def normalize_advantages(self) -> None:
        """Per-rollout advantage normalization (rsl_rl default).

        Must be called after ``compute_returns``.
        """
        if self.advantages is None:
            raise RuntimeError("normalize_advantages() called before compute_returns().")
        adv = self.advantages
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    # --------------------------------------------------------- public reads

    def get_flat_observations(self) -> tuple[jax.Array, jax.Array]:
        """Return ``(flat_actor_obs, flat_critic_obs)`` flattened to
        ``[num_steps * num_envs, *obs_shape]`` for normalizer updates.

        Image groups are left out: the normalizers read the state
        vector alone (images are never normalized), and flattening a
        vision rollout's image storage here materialised a copy of the
        largest tensors in the run at the moment VRAM is tightest.
        """

        def vectors_only(shape: ObsShape) -> ObsShape:
            if isinstance(shape, dict):
                return {g: s for g, s in shape.items() if len(s) == 1}
            return shape

        flat_actor = _obs_reshape(self.actor_obs, vectors_only(self.actor_obs_shape), (-1,))
        flat_critic = _obs_reshape(self.critic_obs, vectors_only(self.critic_obs_shape), (-1,))
        return flat_actor, flat_critic

    def get_flat_actions(self) -> jax.Array:
        """Return all actions flattened to ``[num_steps * num_envs, *action_shape]``."""
        return self.actions.reshape((-1,) + self.action_shape)

    # ----------------------------------------------------------- minibatch

    def get_flat_batch(self) -> RolloutBatch:
        """The whole rollout as ONE flat batch, ``[T*N, ...]`` per field.

        Observation reshapes are views — nothing here copies the rollout.
        Minibatching happens later, inside the update's scan, by indexing
        this flat batch with ``get_minibatch_indices``: the shuffled
        minibatches are never materialized all at once, which is what
        used to blow device memory on image observations (the old
        pre-gathered layout held ``num_epochs`` full copies of the
        rollout).
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("get_flat_batch() called before compute_returns().")
        batch_size = self.num_envs * self.num_steps
        return RolloutBatch(
            actor_observations=_obs_reshape(self.actor_obs, self.actor_obs_shape, (batch_size,)),
            critic_observations=_obs_reshape(self.critic_obs, self.critic_obs_shape, (batch_size,)),
            actions=self.actions.reshape((batch_size,) + self.action_shape),
            values=self.values.reshape(batch_size),
            advantages=self.advantages.reshape(batch_size),
            returns=self.returns.reshape(batch_size),
            old_log_probs=self.log_probs.reshape(batch_size),
            old_mu=self.mu.reshape((batch_size,) + self.action_shape),
            old_sigma=self.sigma.reshape((batch_size,) + self.action_shape),
        )

    def get_minibatch_indices(
        self,
        num_minibatches: int,
        num_epochs: int,
        key: jax.Array,
    ) -> jax.Array:
        """Shuffled minibatch indices into the flat batch.

        Shape ``(num_epochs * num_minibatches, minibatch_size)`` — the
        leading axis is what the scan-based update iterates over.  Key
        handling and permutation order are kept exactly as the old
        pre-gathered path (one independent permutation per epoch, split
        sequentially into minibatches), so row ``i`` selects the very
        same samples the old layout put in stacked batch ``i``.
        """
        batch_size = self.num_envs * self.num_steps
        minibatch_size = batch_size // num_minibatches

        # One independent permutation per epoch.
        keys = jax.random.split(key, num_epochs)
        perms = jax.vmap(lambda k: jax.random.permutation(k, batch_size))(keys)

        return perms.reshape(num_epochs * num_minibatches, minibatch_size)


# ==================== Functional GAE ====================


@jax.jit
def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    episode_starts: jax.Array,
    last_values: jax.Array,
    last_dones: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Generalized Advantage Estimation.

    Args:
        rewards: [num_steps, num_envs]
        values: [num_steps, num_envs]
        episode_starts: [num_steps, num_envs] — True if step is the first of a
            new episode (i.e. previous step was done). Used to prevent
            bootstrapping across episode boundaries.
        last_values: [num_envs] — value of the state AFTER the last collected step.
        last_dones: [num_envs] — done flag at the last collected step.
        gamma: discount factor
        gae_lambda: GAE λ

    Returns:
        advantages: [num_steps, num_envs]
        returns: advantages + values
    """
    num_steps = rewards.shape[0]

    # next_episode_start[t] = episode_starts[t+1] for t < T-1, else last_dones.
    # Equivalent to dones[t]: shifting episode_starts forward by 1 step recovers
    # the original done sequence at every position except the boundary.
    episode_starts_padded = jnp.concatenate(
        [
            episode_starts[1:],
            last_dones[None],
        ],
        axis=0,
    )

    def scan_fn(carry, t):
        gae = carry
        step = num_steps - 1 - t

        current_rewards = rewards[step]
        current_values = values[step]

        next_episode_start = episode_starts_padded[step]
        next_non_terminal = 1.0 - next_episode_start.astype(jnp.float32)

        next_values = jax.lax.cond(
            step == num_steps - 1,
            lambda: last_values,
            lambda: values[step + 1],
        )

        delta = current_rewards + gamma * next_values * next_non_terminal - current_values
        gae = delta + gamma * gae_lambda * next_non_terminal * gae

        return gae, gae

    init_carry = jnp.zeros_like(last_values)
    _, advantages_reversed = jax.lax.scan(scan_fn, init_carry, jnp.arange(num_steps))

    advantages = jnp.flip(advantages_reversed, axis=0)
    returns = advantages + values

    return advantages, returns
