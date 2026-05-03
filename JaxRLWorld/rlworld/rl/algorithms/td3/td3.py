from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlworld.rl.algorithms.base import (
    OffPolicyAlgorithm,
    ActInput,
    copy_params,
)
from rlworld.rl.algorithms.td3.metrics import (
    TD3Metrics,
    TD3CriticMetrics,
    TD3ActorMetrics,
    TD3BatchMetrics,
)
from rlworld.rl.algorithms.td3.update import (
    act_deterministic,
    act_with_noise,
    update_targets,
    get_value,
    update_critics,
    update_actor,
)
from rlworld.rl.modules.policies.td3_ac import TD3ActorCritic
from rlworld.rl.storages.replay_buffer import ReplayBuffer, ReplayBatch


class TD3TrainState(NamedTuple):
    """Training state for TD3 containing all model and optimizer states."""
    model: TD3ActorCritic
    # Target network params
    target_actor_params: Any
    target_critic1_params: Any
    target_critic2_params: Any
    # Target network static parts (for proper reconstruction)
    target_actor_static: Any
    target_critic1_static: Any
    target_critic2_static: Any
    # Optimizer states
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    key: jax.Array


@dataclass
class TD3TransitionBuffer:
    """Buffer for current transition."""
    actor_observations: jax.Array = None
    critic_observations: jax.Array = None
    actions: jax.Array = None
    rewards: jax.Array = None
    dones: jax.Array = None
    values: jax.Array = None

    def clear(self):
        self.actor_observations = None
        self.critic_observations = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.values = None


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3) algorithm implementation with JAX.

    Features:
    - Twin Q-networks to mitigate overestimation
    - Target policy smoothing (clipped noise on target actions)
    - Delayed policy updates
    - Exploration noise during action selection
    """

    ActInput = ActInput

    def __init__(
        self,
        actor_critic: TD3ActorCritic,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        max_grad_norm: float = 10.0,
        key: jax.Array = None,
        **kwargs,
    ):
        """
        Initialize TD3 algorithm.

        Args:
            actor_critic: TD3 Actor-Critic network
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic networks
            gamma: Discount factor
            tau: Target network soft update rate (Polyak averaging)
            batch_size: Batch size for updates
            policy_delay: Update actor every N critic updates (default 2)
            exploration_noise: Stddev of Gaussian exploration noise
            target_policy_noise: Stddev of noise added to target actions
            target_noise_clip: Clipping range for target noise
            max_grad_norm: Maximum gradient norm for clipping
            key: JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        super().__init__(
            actor_critic=actor_critic,
            gamma=gamma,
            tau=tau,
            key=key,
        )

        self.actor_critic = actor_critic
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.max_grad_norm = max_grad_norm

        # Create optimizers
        self._create_optimizers()

        # Initialize training state
        self._init_train_state(key)

        # Observation normalization
        self.obs_normalization = actor_critic.actor_obs_normalizer is not None

        # Storage (initialized via init_storage)
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.transition: Optional[TD3TransitionBuffer] = None

        # Update counter
        self.total_it = 0

        # Print configuration
        self._print_config()

    def _create_optimizers(self):
        """Create optimizers for actor and critics."""
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.actor_lr),
        )

        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.critic_lr),
        )

    def _init_train_state(self, key: jax.Array):
        """Initialize training state."""
        key, subkey = jax.random.split(key)

        # Partition current networks into params and static
        actor_params, actor_static = eqx.partition(
            self.actor_critic.actor, eqx.is_inexact_array
        )
        critic1_params, critic1_static = eqx.partition(
            self.actor_critic.critic1, eqx.is_inexact_array
        )
        critic2_params, critic2_static = eqx.partition(
            self.actor_critic.critic2, eqx.is_inexact_array
        )

        # Initialize target network parameters (copy from main networks)
        target_actor_params = copy_params(actor_params)
        target_critic1_params = copy_params(critic1_params)
        target_critic2_params = copy_params(critic2_params)

        # Static parts are shared (same architecture)
        target_actor_static = actor_static
        target_critic1_static = critic1_static
        target_critic2_static = critic2_static

        # Initialize optimizer states
        actor_opt_state = self.actor_optimizer.init(actor_params)
        critic_opt_state = self.critic_optimizer.init((critic1_params, critic2_params))

        self.train_state = TD3TrainState(
            model=self.actor_critic,
            target_actor_params=target_actor_params,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            target_actor_static=target_actor_static,
            target_critic1_static=target_critic1_static,
            target_critic2_static=target_critic2_static,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            key=subkey,
        )

    def _print_config(self):
        """Print TD3 configuration."""
        print("\n🔍 TD3 Configuration:")
        print(f"  Actor LR: {self.actor_lr}")
        print(f"  Critic LR: {self.critic_lr}")
        print(f"  Tau: {self.tau}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Policy delay: {self.policy_delay}")
        print(f"  Exploration noise: {self.exploration_noise}")
        print(f"  Target policy noise: {self.target_policy_noise}")
        print(f"  Target noise clip: {self.target_noise_clip}")

        print(f"\n🔍 Network architecture:")
        print(f"  Actor obs dim: {self.actor_critic.actor_obs_dim}")
        print(f"  Critic obs dim: {self.actor_critic.critic_obs_dim}")
        print(f"  Action dim: {self.actor_critic.num_actions}")

    @property
    def model(self) -> TD3ActorCritic:
        """Get current model."""
        return self.train_state.model

    def init_storage(self, cfg: Dict[str, Any]) -> None:
        """Initialize replay buffer."""
        self.replay_buffer = ReplayBuffer(
            num_envs=cfg["num_envs"],
            actor_obs_dim=cfg["actor_obs_shape"][0],
            critic_obs_dim=cfg["critic_obs_shape"][0],
            act_dim=cfg["actions_shape"][0],
            size_per_env=cfg["size_per_env"],
            n_steps=cfg["n_steps"],
            gamma=self.gamma,
        )
        self.transition = TD3TransitionBuffer()

    def act(self, obs: ActInput, deterministic: bool = False) -> jax.Array:
        """
        Select action given observation.

        Args:
            obs: Current observation (ActInput)
            deterministic: If True, use deterministic action; otherwise add exploration noise

        Returns:
            Selected action
        """
        model = self.train_state.model
        key = self.train_state.key

        if deterministic:
            actions, values, new_key = act_deterministic(
                model, obs.actor_obs, obs.critic_obs, key
            )
        else:
            actions, values, new_key = act_with_noise(
                model, obs.actor_obs, obs.critic_obs, self.exploration_noise, key
            )

        self.train_state = self.train_state._replace(key=new_key)

        self.transition.actions = actions
        self.transition.actor_observations = obs.actor_obs
        self.transition.critic_observations = obs.critic_obs
        self.transition.values = values

        return actions

    def process_env_step(
        self,
        rewards: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        infos: Dict[str, Any],
    ) -> None:
        """Process environment step and store transition."""
        self.transition.rewards = rewards
        self.transition.dones = terminated | truncated
        self.transition.clear()

    def store_transition(
        self,
        actor_obs: jax.Array,
        critic_obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_actor_obs: jax.Array,
        next_critic_obs: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.store_parallel(
            actor_obs=actor_obs,
            critic_obs=critic_obs,
            act=action,
            rew=reward,
            next_actor_obs=next_actor_obs,
            next_critic_obs=next_critic_obs,
            terminated=terminated,
            truncated=truncated,
        )

    def sample_batch(self, batch_size: int, key: jax.Array) -> ReplayBatch:
        """Sample batch from replay buffer."""
        return self.replay_buffer.sample_batch(batch_size, key)

    def update(self, batch: ReplayBatch) -> TD3Metrics:
        """Update all networks using provided batch."""
        self.total_it += 1

        # Update normalizer stats from batch data
        if self.obs_normalization:
            new_actor_norm = self.train_state.model.actor_obs_normalizer.update(
                batch.actor_observations
            )
            new_critic_norm = self.train_state.model.critic_obs_normalizer.update(
                batch.critic_observations
            )
            new_actor_norm = new_actor_norm.update(batch.next_actor_observations)
            new_critic_norm = new_critic_norm.update(batch.next_critic_observations)
            new_model = eqx.tree_at(
                lambda m: (m.actor_obs_normalizer, m.critic_obs_normalizer),
                self.train_state.model,
                (new_actor_norm, new_critic_norm),
            )
            self.train_state = self.train_state._replace(model=new_model)

        key = self.train_state.key
        key, critic_key, actor_key = jax.random.split(key, 3)

        # Update critics
        new_model, new_critic_opt_state, critic_info = update_critics(
            self.train_state.model,
            self.train_state.target_actor_params,
            self.train_state.target_actor_static,
            self.train_state.target_critic1_params,
            self.train_state.target_critic1_static,
            self.train_state.target_critic2_params,
            self.train_state.target_critic2_static,
            self.train_state.critic_opt_state,
            batch,
            self.critic_optimizer,
            self.gamma,
            self.target_policy_noise,
            self.target_noise_clip,
            critic_key,
        )

        self.train_state = self.train_state._replace(
            model=new_model,
            critic_opt_state=new_critic_opt_state,
            key=key,
        )

        # Update actor (with policy delay)
        if self.total_it % self.policy_delay == 0:
            new_model, new_actor_opt_state, actor_info = update_actor(
                self.train_state.model,
                self.train_state.actor_opt_state,
                self.actor_optimizer,
                batch,
                actor_key,
            )

            actor_loss = float(actor_info["actor_loss"])
            action_mean = float(actor_info["action_mean"])
            action_std = float(actor_info["action_std"])

            self.train_state = self.train_state._replace(
                model=new_model,
                actor_opt_state=new_actor_opt_state,
            )

            # Update target networks
            self._update_target_networks()
        else:
            actor_loss = 0.0
            action_mean = 0.0
            action_std = 0.0

        jax.block_until_ready(self.train_state.model)

        # Build metrics
        metrics = self._build_metrics(critic_info, actor_loss, action_mean, action_std, batch)

        return metrics

    def _build_metrics(
        self,
        critic_info: Dict[str, jax.Array],
        actor_loss: float,
        action_mean: float,
        action_std: float,
        batch: ReplayBatch,
    ) -> TD3Metrics:
        """Build metrics from update results."""
        return TD3Metrics(
            critic=TD3CriticMetrics(
                loss=float(critic_info["critic_loss"]),
                critic1_loss=float(critic_info["critic1_loss"]),
                critic2_loss=float(critic_info["critic2_loss"]),
                q1_mean=float(critic_info["q1_value"]),
                q2_mean=float(critic_info["q2_value"]),
                q1_std=float(critic_info["current_q1_std"]),
                q2_std=float(critic_info["current_q2_std"]),
                q_target_mean=float(critic_info["target_q_value"]),
            ),
            actor=TD3ActorMetrics(
                loss=actor_loss,
                action_mean=action_mean,
                action_std=action_std,
            ),
            batch=TD3BatchMetrics(
                reward_mean=float(batch.rewards.mean()),
                action_mean=float(batch.actions.mean()),
                action_std=float(batch.actions.std()),
            ),
            total_updates=self.total_it,
        )

    def _update_target_networks(self) -> None:
        """Update target networks with Polyak averaging."""
        new_target_actor, new_target_critic1, new_target_critic2 = update_targets(
            self.train_state.model,
            self.train_state.target_actor_params,
            self.train_state.target_critic1_params,
            self.train_state.target_critic2_params,
            self.tau,
        )

        self.train_state = self.train_state._replace(
            target_actor_params=new_target_actor,
            target_critic1_params=new_target_critic1,
            target_critic2_params=new_target_critic2,
        )

    def get_value(self, actor_obs: jax.Array, critic_obs: jax.Array) -> jax.Array:
        """Get value estimate for observations."""
        key = self.train_state.key
        key, subkey = jax.random.split(key)
        self.train_state = self.train_state._replace(key=key)
        return get_value(self.train_state.model, actor_obs, critic_obs, subkey)

    def save_train_state(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Save TD3 training state."""
        import os

        model_path = os.path.join(checkpoint_dir, "model.eqx")
        eqx.tree_serialise_leaves(model_path, self.train_state.model)

        target_path = os.path.join(checkpoint_dir, "target_networks.eqx")
        target_networks = {
            "actor": self.train_state.target_actor_params,
            "critic1": self.train_state.target_critic1_params,
            "critic2": self.train_state.target_critic2_params,
        }
        eqx.tree_serialise_leaves(target_path, target_networks)

        return {
            "alg_class": self.__class__.__name__,
            "alg_key": np.array(self.train_state.key),
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "total_it": self.total_it,
        }

    def load_train_state(self, checkpoint_dir: str, metadata: Dict[str, Any]) -> None:
        """Load TD3 training state."""
        import os

        model_path = os.path.join(checkpoint_dir, "model.eqx")
        new_model = eqx.tree_deserialise_leaves(model_path, self.train_state.model)

        target_path = os.path.join(checkpoint_dir, "target_networks.eqx")
        target_networks_template = {
            "actor": self.train_state.target_actor_params,
            "critic1": self.train_state.target_critic1_params,
            "critic2": self.train_state.target_critic2_params,
        }
        target_networks = eqx.tree_deserialise_leaves(target_path, target_networks_template)

        # Re-partition to get params and static
        new_actor_params, new_actor_static = eqx.partition(
            new_model.actor, eqx.is_inexact_array
        )
        new_critic1_params, new_critic1_static = eqx.partition(
            new_model.critic1, eqx.is_inexact_array
        )
        new_critic2_params, new_critic2_static = eqx.partition(
            new_model.critic2, eqx.is_inexact_array
        )

        # Re-initialize optimizer states
        actor_opt_state = self.actor_optimizer.init(new_actor_params)
        critic_opt_state = self.critic_optimizer.init(
            (new_critic1_params, new_critic2_params)
        )

        self.train_state = TD3TrainState(
            model=new_model,
            target_actor_params=target_networks["actor"],
            target_critic1_params=target_networks["critic1"],
            target_critic2_params=target_networks["critic2"],
            target_actor_static=new_actor_static,
            target_critic1_static=new_critic1_static,
            target_critic2_static=new_critic2_static,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            key=jnp.array(metadata["alg_key"], dtype=jnp.uint32),
        )

        self.actor_lr = metadata.get("actor_lr", self.actor_lr)
        self.critic_lr = metadata.get("critic_lr", self.critic_lr)
        self.total_it = metadata.get("total_it", 0)