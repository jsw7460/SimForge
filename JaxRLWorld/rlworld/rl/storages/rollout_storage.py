from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp


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
    issues an ``at[step].set(...)`` functional update that XLA can fuse
    into in-place writes; clearing the rollout just resets the step
    counter, so the buffers themselves are reused across iterations and
    no ``jnp.stack`` happens at the boundary between collection and
    learning.

    Public API (PPO / PPO-DR3):
        - add_transition(...): record one timestep's data
        - compute_returns(...): GAE → fills ``advantages``/``returns``
        - normalize_advantages(): rsl_rl-style per-rollout normalization
        - get_flat_observations(): flatten obs for normalizer updates
        - get_flat_actions(): flatten actions for action-stat logging
        - get_stacked_batches(...): shuffled minibatches for the update loop
        - clear(): reset for the next rollout (no reallocation)
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        actor_obs_shape: tuple[int, ...],
        critic_obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.actor_obs_shape = actor_obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.action_shape = action_shape

        self.step = 0
        self._allocate_buffers()

    # ---------------------------------------------------------------- alloc

    def _allocate_buffers(self) -> None:
        T, N = self.num_steps, self.num_envs
        self.actor_obs = jnp.zeros((T, N) + self.actor_obs_shape)
        self.critic_obs = jnp.zeros((T, N) + self.critic_obs_shape)
        self.actions = jnp.zeros((T, N) + self.action_shape)
        self.rewards = jnp.zeros((T, N))
        self.dones = jnp.zeros((T, N), dtype=jnp.bool_)
        self.episode_starts = jnp.zeros((T, N), dtype=jnp.bool_)
        self.values = jnp.zeros((T, N))
        self.log_probs = jnp.zeros((T, N))
        self.mu = jnp.zeros((T, N) + self.action_shape)
        self.sigma = jnp.zeros((T, N) + self.action_shape)
        # Filled by compute_returns()
        self.advantages: Optional[jax.Array] = None
        self.returns: Optional[jax.Array] = None

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

        s = self.step
        self.actor_obs = self.actor_obs.at[s].set(actor_obs)
        self.critic_obs = self.critic_obs.at[s].set(critic_obs)
        self.actions = self.actions.at[s].set(actions)
        self.rewards = self.rewards.at[s].set(rewards)
        self.dones = self.dones.at[s].set(dones)
        self.episode_starts = self.episode_starts.at[s].set(episode_starts)
        self.values = self.values.at[s].set(values)
        self.log_probs = self.log_probs.at[s].set(log_probs)
        self.mu = self.mu.at[s].set(mu)
        self.sigma = self.sigma.at[s].set(sigma)
        self.step += 1

    def clear(self) -> None:
        """Reset for next rollout. Buffers are reused; advantages/returns dropped."""
        self.step = 0
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
            raise RuntimeError(
                "normalize_advantages() called before compute_returns()."
            )
        adv = self.advantages
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    # --------------------------------------------------------- public reads

    def get_flat_observations(self) -> tuple[jax.Array, jax.Array]:
        """Return ``(flat_actor_obs, flat_critic_obs)`` flattened to
        ``[num_steps * num_envs, *obs_shape]`` for normalizer updates."""
        flat_actor = self.actor_obs.reshape((-1,) + self.actor_obs_shape)
        flat_critic = self.critic_obs.reshape((-1,) + self.critic_obs_shape)
        return flat_actor, flat_critic

    def get_flat_actions(self) -> jax.Array:
        """Return all actions flattened to ``[num_steps * num_envs, *action_shape]``."""
        return self.actions.reshape((-1,) + self.action_shape)

    # ----------------------------------------------------------- minibatch

    def get_stacked_batches(
        self,
        num_minibatches: int,
        num_epochs: int,
        key: jax.Array,
    ) -> RolloutBatch:
        """Get stacked, shuffled minibatches for the PPO update loop.

        Output shape per field: ``(num_epochs * num_minibatches, minibatch_size, ...)``.
        The leading axis is the dimension that the scan-based update
        iterates over.
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError(
                "get_stacked_batches() called before compute_returns()."
            )
        batch_size = self.num_envs * self.num_steps
        minibatch_size = batch_size // num_minibatches
        num_total_batches = num_epochs * num_minibatches

        # Flatten [T, N, ...] → [T*N, ...]
        flat_actor_obs = self.actor_obs.reshape((batch_size,) + self.actor_obs_shape)
        flat_critic_obs = self.critic_obs.reshape((batch_size,) + self.critic_obs_shape)
        flat_actions = self.actions.reshape((batch_size,) + self.action_shape)
        flat_values = self.values.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        flat_log_probs = self.log_probs.reshape(batch_size)
        flat_mu = self.mu.reshape((batch_size,) + self.action_shape)
        flat_sigma = self.sigma.reshape((batch_size,) + self.action_shape)

        # One independent permutation per epoch.
        keys = jax.random.split(key, num_epochs)
        perms = jax.vmap(lambda k: jax.random.permutation(k, batch_size))(keys)

        shuf_actor_obs = jax.vmap(lambda p: flat_actor_obs[p])(perms)
        shuf_critic_obs = jax.vmap(lambda p: flat_critic_obs[p])(perms)
        shuf_actions = jax.vmap(lambda p: flat_actions[p])(perms)
        shuf_values = jax.vmap(lambda p: flat_values[p])(perms)
        shuf_advantages = jax.vmap(lambda p: flat_advantages[p])(perms)
        shuf_returns = jax.vmap(lambda p: flat_returns[p])(perms)
        shuf_log_probs = jax.vmap(lambda p: flat_log_probs[p])(perms)
        shuf_mu = jax.vmap(lambda p: flat_mu[p])(perms)
        shuf_sigma = jax.vmap(lambda p: flat_sigma[p])(perms)

        # (num_epochs, num_minibatches, minibatch_size, ...)
        def reshape_batches(arr):
            return arr.reshape((num_epochs, num_minibatches, minibatch_size) + arr.shape[2:])

        # (num_epochs * num_minibatches, minibatch_size, ...)
        def merge_epochs(arr):
            return arr.reshape((num_total_batches, minibatch_size) + arr.shape[3:])

        return RolloutBatch(
            actor_observations=merge_epochs(reshape_batches(shuf_actor_obs)),
            critic_observations=merge_epochs(reshape_batches(shuf_critic_obs)),
            actions=merge_epochs(reshape_batches(shuf_actions)),
            values=merge_epochs(reshape_batches(shuf_values)),
            advantages=merge_epochs(reshape_batches(shuf_advantages)),
            returns=merge_epochs(reshape_batches(shuf_returns)),
            old_log_probs=merge_epochs(reshape_batches(shuf_log_probs)),
            old_mu=merge_epochs(reshape_batches(shuf_mu)),
            old_sigma=merge_epochs(reshape_batches(shuf_sigma)),
        )


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
    episode_starts_padded = jnp.concatenate([
        episode_starts[1:],
        last_dones[None],
    ], axis=0)

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
