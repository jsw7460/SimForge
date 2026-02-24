from typing import NamedTuple, Generator
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
    """Rollout storage using Python lists during collection."""

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
        self._init_lists()
        self._storage = None  # Built after rollout

    def _init_lists(self):
        """Initialize empty lists for collection."""
        self._actor_obs_list = []
        self._critic_obs_list = []
        self._actions_list = []
        self._rewards_list = []
        self._dones_list = []
        self._episode_starts_list = []
        self._values_list = []
        self._log_probs_list = []
        self._mu_list = []
        self._sigma_list = []

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
        """Add transition to lists (O(1) append)."""
        if self.step >= self.num_steps:
            raise RuntimeError("Storage overflow.")

        self._actor_obs_list.append(actor_obs)
        self._critic_obs_list.append(critic_obs)
        self._actions_list.append(actions)
        self._rewards_list.append(rewards)
        self._dones_list.append(dones)
        self._episode_starts_list.append(episode_starts)
        self._values_list.append(values)
        self._log_probs_list.append(log_probs)
        self._mu_list.append(mu)
        self._sigma_list.append(sigma)
        self.step += 1

    def finalize(self) -> None:
        """Stack lists into JAX arrays (call after rollout complete)."""
        self._storage = {
            "actor_obs": jnp.stack(self._actor_obs_list),
            "critic_obs": jnp.stack(self._critic_obs_list),
            "actions": jnp.stack(self._actions_list),
            "rewards": jnp.stack(self._rewards_list),
            "dones": jnp.stack(self._dones_list),
            "episode_starts": jnp.stack(self._episode_starts_list),
            "values": jnp.stack(self._values_list),
            "log_probs": jnp.stack(self._log_probs_list),
            "mu": jnp.stack(self._mu_list),
            "sigma": jnp.stack(self._sigma_list),
            "advantages": None,
            "returns": None,
        }

    def compute_returns(
        self,
        last_values: jax.Array,
        last_dones: jax.Array,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE (call finalize() first)."""
        advantages, returns = compute_gae(
            rewards=self._storage["rewards"],
            values=self._storage["values"],
            episode_starts=self._storage["episode_starts"],
            last_values=last_values,
            last_dones=last_dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self._storage["advantages"] = advantages
        self._storage["returns"] = returns

    def get_stacked_batches(
        self,
        num_minibatches: int,
        num_epochs: int,
        key: jax.Array,
    ) -> RolloutBatch:
        """Get stacked batches for update."""
        batch_size = self.num_envs * self.num_steps
        minibatch_size = batch_size // num_minibatches
        num_total_batches = num_epochs * num_minibatches

        # Flatten
        flat_actor_obs = self._storage["actor_obs"].reshape((batch_size,) + self.actor_obs_shape)
        flat_critic_obs = self._storage["critic_obs"].reshape((batch_size,) + self.critic_obs_shape)
        flat_actions = self._storage["actions"].reshape((batch_size,) + self.action_shape)
        flat_values = self._storage["values"].reshape(batch_size)
        flat_advantages = self._storage["advantages"].reshape(batch_size)
        flat_returns = self._storage["returns"].reshape(batch_size)
        flat_log_probs = self._storage["log_probs"].reshape(batch_size)
        flat_mu = self._storage["mu"].reshape((batch_size,) + self.action_shape)
        flat_sigma = self._storage["sigma"].reshape((batch_size,) + self.action_shape)

        # Generate all permutations
        keys = jax.random.split(key, num_epochs)
        perms = jax.vmap(lambda k: jax.random.permutation(k, batch_size))(keys)

        # Shuffle and reshape
        shuf_actor_obs = jax.vmap(lambda p: flat_actor_obs[p])(perms)
        shuf_critic_obs = jax.vmap(lambda p: flat_critic_obs[p])(perms)
        shuf_actions = jax.vmap(lambda p: flat_actions[p])(perms)
        shuf_values = jax.vmap(lambda p: flat_values[p])(perms)
        shuf_advantages = jax.vmap(lambda p: flat_advantages[p])(perms)
        shuf_returns = jax.vmap(lambda p: flat_returns[p])(perms)
        shuf_log_probs = jax.vmap(lambda p: flat_log_probs[p])(perms)
        shuf_mu = jax.vmap(lambda p: flat_mu[p])(perms)
        shuf_sigma = jax.vmap(lambda p: flat_sigma[p])(perms)

        # Reshape to (num_epochs, num_minibatches, minibatch_size, ...)
        def reshape_batches(arr):
            shape = arr.shape
            new_shape = (num_epochs, num_minibatches, minibatch_size) + shape[2:]
            return arr.reshape(new_shape)

        # Merge to (num_total_batches, minibatch_size, ...)
        def merge_epochs(arr):
            shape = arr.shape
            new_shape = (num_total_batches, minibatch_size) + shape[3:]
            return arr.reshape(new_shape)

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

    def clear(self) -> None:
        """Reset for next rollout."""
        self.step = 0
        self._init_lists()
        self._storage = None

# ==================== Functional API ====================

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
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards [num_steps, num_envs]
        values: Value estimates [num_steps, num_envs]
        episode_starts: Flags indicating if a new episode started at each step [num_steps, num_envs].
                        Used to prevent bootstrapping across episode boundaries.
                        episode_starts[t] = dones[t-1] (previous step's done flag).
        last_values: Value of last state [num_envs]
        last_dones: Done flag of last state [num_envs]
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: GAE advantages [num_steps, num_envs]
        returns: Returns (advantages + values) [num_steps, num_envs]
    """
    num_steps = rewards.shape[0]

    # Prepend a dummy row for episode_starts[step+1] indexing
    # episode_starts_shifted[step] = episode_starts[step+1] for step < num_steps-1
    # episode_starts_shifted[num_steps-1] = last_dones
    episode_starts_padded = jnp.concatenate([
        episode_starts[1:],  # [1, 2, ..., num_steps-1]
        last_dones[None],  # last_dones as the final "next" episode start
    ], axis=0)

    def scan_fn(carry, t):
        gae = carry
        step = num_steps - 1 - t

        current_rewards = rewards[step]
        current_values = values[step]

        next_episode_start = episode_starts_padded[step]
        next_non_terminal = 1.0 - next_episode_start.astype(jnp.float32)

        # next_values
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


# ==================== Storage State (for functional style) ====================

class StorageState(NamedTuple):
    """Immutable storage state for functional API."""
    actor_obs: jax.Array  # [num_steps, num_envs, obs_dim]
    critic_obs: jax.Array  # [num_steps, num_envs, obs_dim]
    actions: jax.Array  # [num_steps, num_envs, action_dim]
    rewards: jax.Array  # [num_steps, num_envs]
    dones: jax.Array  # [num_steps, num_envs]
    values: jax.Array  # [num_steps, num_envs]
    log_probs: jax.Array  # [num_steps, num_envs]
    mu: jax.Array  # [num_steps, num_envs, action_dim]
    sigma: jax.Array  # [num_steps, num_envs, action_dim]
    step: int  # Current step


def create_storage_state(
    num_steps: int,
    num_envs: int,
    actor_obs_shape: tuple[int, ...],
    critic_obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
) -> StorageState:
    """Create initial storage state."""
    return StorageState(
        actor_obs=jnp.zeros((num_steps, num_envs, *actor_obs_shape)),
        critic_obs=jnp.zeros((num_steps, num_envs, *critic_obs_shape)),
        actions=jnp.zeros((num_steps, num_envs, *action_shape)),
        rewards=jnp.zeros((num_steps, num_envs)),
        dones=jnp.zeros((num_steps, num_envs), dtype=jnp.bool_),
        values=jnp.zeros((num_steps, num_envs)),
        log_probs=jnp.zeros((num_steps, num_envs)),
        mu=jnp.zeros((num_steps, num_envs, *action_shape)),
        sigma=jnp.zeros((num_steps, num_envs, *action_shape)),
        step=0,
    )


def add_transition_to_state(
    state: StorageState,
    transition: Transition,
) -> StorageState:
    """Add transition to storage state (functional style)."""
    step = state.step
    return StorageState(
        actor_obs=state.actor_obs.at[step].set(transition.actor_obs),
        critic_obs=state.critic_obs.at[step].set(transition.critic_obs),
        actions=state.actions.at[step].set(transition.actions),
        rewards=state.rewards.at[step].set(transition.rewards),
        dones=state.dones.at[step].set(transition.dones),
        values=state.values.at[step].set(transition.values),
        log_probs=state.log_probs.at[step].set(transition.log_probs),
        mu=state.mu.at[step].set(transition.mu),
        sigma=state.sigma.at[step].set(transition.sigma),
        step=step + 1,
    )
