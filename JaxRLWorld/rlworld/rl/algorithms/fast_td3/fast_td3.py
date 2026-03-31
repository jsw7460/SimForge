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
from rlworld.rl.algorithms.fast_td3.metrics import (
    FastTD3Metrics,
    FastTD3CriticMetrics,
    FastTD3ActorMetrics,
    FastTD3BatchMetrics,
)
from rlworld.rl.algorithms.fast_td3.update import (
    act_deterministic,
    act_with_noise,
    update_targets,
    get_value,
    update_critics,
    update_actor,
    init_noise_scales,
    resample_noise_on_done,
)
from rlworld.rl.modules.policies.fast_td3_ac import FastTD3ActorCritic
from rlworld.rl.storages.replay_buffer import ReplayBuffer, ReplayBatch
from rlworld.rl.modules.normalization import EmpiricalNormalization


class FastTD3TrainState(NamedTuple):
    """Training state for FastTD3."""
    model: FastTD3ActorCritic
    # Target network params
    target_actor_params: Any
    target_critic1_params: Any
    target_critic2_params: Any
    # Target network static parts
    target_actor_static: Any
    target_critic1_static: Any
    target_critic2_static: Any
    # Optimizer states
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    # Per-environment noise scales
    noise_scales: jax.Array
    key: jax.Array


@dataclass
class FastTD3TransitionBuffer:
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


class FastTD3(OffPolicyAlgorithm):
    """
    FastTD3: Simple, Fast, and Capable RL for Humanoid Control.

    Key features over standard TD3:
    - Distributional critics (C51) for better value estimation
    - Large batch size (32768) for stable learning
    - Per-environment mixed exploration noise
    - Optimized for parallel simulation

    Reference: https://younggyo.me/fast_td3
    """

    ActInput = ActInput

    def __init__(
        self,
        actor_critic: FastTD3ActorCritic,
        num_envs: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 32768,
        policy_delay: int = 2,
        noise_min: float = 0.05,
        noise_max: float = 0.4,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        use_cdq: bool = True,
        use_target_actor: bool = False,
        max_grad_norm: float = 10.0,
        key: jax.Array = None,
        **kwargs,
    ):
        """
        Initialize FastTD3 algorithm.

        Args:
            actor_critic: FastTD3 Actor-Critic network with distributional critics
            num_envs: Number of parallel environments
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critics
            gamma: Discount factor
            tau: Target network soft update rate
            batch_size: Batch size (default 32768 for FastTD3)
            policy_delay: Update actor every N critic updates
            noise_min: Minimum exploration noise scale
            noise_max: Maximum exploration noise scale
            target_policy_noise: Noise added to target actions
            target_noise_clip: Clipping for target noise
            use_cdq: Use Clipped Double Q-learning (min) vs average
            max_grad_norm: Maximum gradient norm
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
        self.num_envs = num_envs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.use_cdq = use_cdq
        self.use_target_actor = use_target_actor
        self.max_grad_norm = max_grad_norm

        # Check if model has normalizers enabled
        self.obs_normalization = actor_critic.actor_obs_normalizer is not None

        # Create optimizers
        self._create_optimizers()

        # Initialize training state
        self._init_train_state(key)

        # Storage
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.transition: Optional[FastTD3TransitionBuffer] = None

        # Update counter
        self.total_it = 0

        # Print configuration
        self._print_config()

    def _create_optimizers(self):
        """Create optimizers for actor and critics."""
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adamw(learning_rate=self.actor_lr, weight_decay=0.1),
        )

        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adamw(learning_rate=self.critic_lr, weight_decay=0.1),
        )
    def _init_train_state(self, key: jax.Array):
        """Initialize training state including noise scales."""
        key, noise_key, subkey = jax.random.split(key, 3)

        # Partition networks
        actor_params, actor_static = eqx.partition(
            self.actor_critic.actor, eqx.is_inexact_array
        )
        critic1_params, critic1_static = eqx.partition(
            self.actor_critic.critic1, eqx.is_inexact_array
        )
        critic2_params, critic2_static = eqx.partition(
            self.actor_critic.critic2, eqx.is_inexact_array
        )

        # Initialize target networks
        target_actor_params = copy_params(actor_params)
        target_critic1_params = copy_params(critic1_params)
        target_critic2_params = copy_params(critic2_params)

        # Initialize optimizer states
        actor_opt_state = self.actor_optimizer.init(actor_params)
        critic_opt_state = self.critic_optimizer.init((critic1_params, critic2_params))

        # Initialize per-environment noise scales
        noise_scales = init_noise_scales(
            self.num_envs, self.noise_min, self.noise_max, noise_key
        )

        # NOTE: Normalizers are now stored inside self.actor_critic (model)
        # instead of in train_state

        self.train_state = FastTD3TrainState(
            model=self.actor_critic,
            target_actor_params=target_actor_params,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            target_actor_static=actor_static,
            target_critic1_static=critic1_static,
            target_critic2_static=critic2_static,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            noise_scales=noise_scales,
            key=subkey,
        )

    def _print_config(self):
        """Print FastTD3 configuration."""
        print("\n🚀 FastTD3 Configuration:")
        print(f"  Actor LR: {self.actor_lr}")
        print(f"  Critic LR: {self.critic_lr}")
        print(f"  Tau: {self.tau}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Policy delay: {self.policy_delay}")
        print(f"  Noise range: [{self.noise_min}, {self.noise_max}]")
        print(f"  Target policy noise: {self.target_policy_noise}")
        print(f"  Target noise clip: {self.target_noise_clip}")
        print(f"  Use CDQ: {self.use_cdq}")
        print(f"  Use target actor: {self.use_target_actor}")
        print(f"  Obs normalization: {self.obs_normalization}")

        print(f"\n📊 Distributional RL (C51):")
        print(f"  Num atoms: {self.actor_critic.num_atoms}")
        print(f"  V_min: {self.actor_critic.v_min}")
        print(f"  V_max: {self.actor_critic.v_max}")

        print(f"\n🔍 Network architecture:")
        print(f"  Actor obs dim: {self.actor_critic.actor_obs_dim}")
        print(f"  Critic obs dim: {self.actor_critic.critic_obs_dim}")
        print(f"  Action dim: {self.actor_critic.num_actions}")

    @property
    def model(self) -> FastTD3ActorCritic:
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
        self.transition = FastTD3TransitionBuffer()

    def act(self, obs: ActInput, deterministic: bool = False) -> jax.Array:
        """
        Select action with optional exploration noise.

        NOTE: Observation normalization is now handled inside model.act()
        via _normalize_actor_obs().
        """
        actor_obs = obs.actor_obs
        critic_obs = obs.critic_obs

        if deterministic:
            actions, values, new_key = act_deterministic(
                self.train_state.model,
                actor_obs,
                critic_obs,
                self.train_state.key,
            )
        else:
            actions, values, new_key = act_with_noise(
                self.train_state.model,
                actor_obs,
                critic_obs,
                self.train_state.noise_scales,
                self.train_state.key,
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
        """Process environment step and update noise scales for done envs.

        NOTE: Observation normalizer is NOT updated here. It is updated
        in update() from batch data, avoiding per-step model/train_state
        recreation (matching PPO's approach).
        """
        dones = terminated | truncated

        # Resample noise for done environments
        new_noise_scales, new_key = resample_noise_on_done(
            self.train_state.noise_scales,
            dones,
            self.noise_min,
            self.noise_max,
            self.train_state.key,
        )

        self.train_state = self.train_state._replace(
            noise_scales=new_noise_scales,
            key=new_key,
        )

        self.transition.rewards = rewards
        self.transition.dones = dones
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

    def update(self, batch: ReplayBatch) -> FastTD3Metrics:
        """
        Update all networks using provided batch.

        NOTE: Observation normalization is now handled inside model methods
        (_normalize_actor_obs, _normalize_critic_obs), so we no longer need
        to pass mean/std parameters to update functions.
        """
        self.total_it += 1

        # Update normalizer stats from batch data (matches original FastTD3 behavior
        # where normalize_obs() updates running stats during both collection AND training)
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

        # Update critics (distributional)
        # NOTE: Normalization is handled inside compute_critic_loss via model methods
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
            self.use_cdq,
            self.use_target_actor,
            critic_key,
        )

        self.train_state = self.train_state._replace(
            model=new_model,
            critic_opt_state=new_critic_opt_state,
            key=key,
        )

        # Update target networks
        self._update_target_networks()

        # Update actor (with policy delay)
        if self.total_it % self.policy_delay == 0:
            # NOTE: Normalization is handled inside compute_actor_loss via model methods
            new_model, new_actor_opt_state, actor_info = update_actor(
                self.train_state.model,
                self.train_state.actor_opt_state,
                self.actor_optimizer,
                batch,
                self.use_cdq,
                actor_key,
            )

            actor_loss = float(actor_info["actor_loss"])
            action_mean = float(actor_info["action_mean"])
            action_std = float(actor_info["action_std"])
            actor_q_value = float(actor_info["actor_q_value"])

            self.train_state = self.train_state._replace(
                model=new_model,
                actor_opt_state=new_actor_opt_state,
            )
        else:
            actor_loss = 0.0
            action_mean = 0.0
            action_std = 0.0
            actor_q_value = 0.0

        jax.block_until_ready(self.train_state.model)

        # Build metrics
        metrics = self._build_metrics(
            critic_info, actor_loss, action_mean, action_std, actor_q_value, batch
        )

        return metrics

    def _build_metrics(
        self,
        critic_info: Dict[str, jax.Array],
        actor_loss: float,
        action_mean: float,
        action_std: float,
        actor_q_value: float,
        batch: ReplayBatch,
    ) -> FastTD3Metrics:
        """Build metrics from update results."""
        return FastTD3Metrics(
            critic=FastTD3CriticMetrics(
                loss=float(critic_info["critic_loss"]),
                critic1_loss=float(critic_info["critic1_loss"]),
                critic2_loss=float(critic_info["critic2_loss"]),
                q1_mean=float(critic_info["q1_value"]),
                q2_mean=float(critic_info["q2_value"]),
                q1_std=float(critic_info["q1_std"]),
                q2_std=float(critic_info["q2_std"]),
                q_target_mean=float(critic_info["target_q_value"]),
                q1_max=float(critic_info["q1_max"]),
                q1_min=float(critic_info["q1_min"]),
            ),
            actor=FastTD3ActorMetrics(
                loss=actor_loss,
                action_mean=action_mean,
                action_std=action_std,
                q_value=actor_q_value,
            ),
            batch=FastTD3BatchMetrics(
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
            self.use_target_actor,
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
        """
        Save FastTD3 training state.

        NOTE: Normalizers are now saved as part of model.eqx since they are
        stored inside FastTD3ActorCritic.
        """
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
            "noise_scales": np.array(self.train_state.noise_scales),
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "total_it": self.total_it,
            "num_atoms": self.actor_critic.num_atoms,
            "v_min": self.actor_critic.v_min,
            "v_max": self.actor_critic.v_max,
            "obs_normalization": self.obs_normalization,
        }

    def load_train_state(self, checkpoint_dir: str, metadata: Dict[str, Any]) -> None:
        """
        Load FastTD3 training state.

        NOTE: Normalizers are now loaded as part of model.eqx since they are
        stored inside FastTD3ActorCritic.
        """
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

        # Re-partition
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

        # Restore noise scales
        noise_scales = jnp.array(metadata.get("noise_scales", self.train_state.noise_scales))

        self.train_state = FastTD3TrainState(
            model=new_model,
            target_actor_params=target_networks["actor"],
            target_critic1_params=target_networks["critic1"],
            target_critic2_params=target_networks["critic2"],
            target_actor_static=new_actor_static,
            target_critic1_static=new_critic1_static,
            target_critic2_static=new_critic2_static,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            noise_scales=noise_scales,
            key=jnp.array(metadata["alg_key"], dtype=jnp.uint32),
        )

        self.actor_lr = metadata.get("actor_lr", self.actor_lr)
        self.critic_lr = metadata.get("critic_lr", self.critic_lr)
        self.total_it = metadata.get("total_it", 0)