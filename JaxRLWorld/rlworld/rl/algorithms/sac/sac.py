import os
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlworld.rl.algorithms.base import (
    ActInput,
    OffPolicyAlgorithm,
    copy_params,
    polyak_update,
)
from rlworld.rl.algorithms.metrics import ActorMetrics
from rlworld.rl.algorithms.sac.metrics import (
    SACAlphaMetrics,
    SACBatchMetrics,
    SACCriticMetrics,
    SACMetrics,
)
from rlworld.rl.algorithms.sac.update import (
    act_deterministic,
    act_stochastic,
    get_value,
    update_actor,
    update_alpha,
    update_critics,
)
from rlworld.rl.modules.policies.sac_ac import SACActorCritic
from rlworld.rl.storages.replay_buffer import ReplayBatch, ReplayBuffer


@eqx.filter_jit
def _update_normalizers(
    model: SACActorCritic,
    actor_obs: jax.Array,
    critic_obs: jax.Array,
) -> SACActorCritic:
    """JIT-compiled normalizer update (PPO-style)."""
    new_actor_normalizer = model.actor_obs_normalizer.update(actor_obs)
    new_critic_normalizer = model.critic_obs_normalizer.update(critic_obs)
    return eqx.tree_at(
        lambda m: (m.actor_obs_normalizer, m.critic_obs_normalizer),
        model,
        (new_actor_normalizer, new_critic_normalizer),
    )


class SACTrainState(NamedTuple):
    """Training state for SAC containing all model and optimizer states."""

    model: SACActorCritic
    target_critic1_params: Any
    target_critic2_params: Any
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    alpha_opt_state: optax.OptState | None
    log_ent_coef: jax.Array | None
    key: jax.Array


@dataclass
class SACTransitionBuffer:
    """Buffer for current transition."""

    actor_observations: jax.Array = None
    critic_observations: jax.Array = None
    actions: jax.Array = None
    rewards: jax.Array = None
    dones: jax.Array = None
    values: jax.Array = None
    action_mean: jax.Array = None

    def clear(self):
        self.actor_observations = None
        self.critic_observations = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.values = None
        self.action_mean = None


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) algorithm implementation with JAX.

    Features:
    - Twin Q-networks to mitigate overestimation
    - Automatic entropy coefficient tuning
    - Separate learning rates for actor/critic/alpha
    - Policy delay option
    """

    ActInput = ActInput

    def __init__(
        self,
        actor_critic: SACActorCritic,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        policy_delay: int = 1,
        max_grad_norm: float = 10.0,
        key: jax.Array = None,
        **kwargs,
    ):
        """
        Initialize SAC algorithm.

        Args:
            actor_critic: SAC Actor-Critic network
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic networks
            alpha_lr: Learning rate for entropy coefficient
            gamma: Discount factor
            tau: Target network soft update rate (Polyak averaging)
            batch_size: Batch size for updates
            ent_coef: Entropy coefficient ('auto' for automatic tuning or fixed float)
            target_entropy: Target entropy ('auto' for -dim(action_space) or fixed float)
            policy_delay: Update actor every N critic updates (default 1)
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
        self.alpha_lr = alpha_lr
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.max_grad_norm = max_grad_norm

        # Setup entropy tuning
        self.ent_coef_config = ent_coef
        self.target_entropy_config = target_entropy
        self._setup_entropy_tuning()

        # Create optimizers
        self._create_optimizers()

        # Initialize training state
        self._init_train_state(key)

        # Observation normalization
        self.obs_normalization = actor_critic.actor_obs_normalizer is not None

        # Storage (initialized via init_storage)
        self.replay_buffer: ReplayBuffer | None = None
        self.transition: SACTransitionBuffer | None = None

        # Update counter
        self.total_it = 0

        # Print configuration
        self._print_config()

    def _setup_entropy_tuning(self):
        """Setup entropy coefficient for automatic or fixed tuning."""
        if self.target_entropy_config == "auto":
            self.target_entropy = float(-self.actor_critic.num_actions)
        else:
            self.target_entropy = float(self.target_entropy_config)

        if isinstance(self.ent_coef_config, str) and self.ent_coef_config.startswith("auto"):
            self.auto_entropy = True
            init_value = 1.0
            if "_" in self.ent_coef_config:
                init_value = float(self.ent_coef_config.split("_")[1])
                assert init_value > 0.0, "Initial ent_coef must be > 0"
            self.init_ent_coef = init_value
        else:
            self.auto_entropy = False
            self.fixed_ent_coef = float(self.ent_coef_config)

    def _create_optimizers(self):
        """Create optimizers for actor, critics, and entropy coefficient."""
        self.actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.actor_lr),
        )

        self.critic_optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.critic_lr),
        )

        if self.auto_entropy:
            self.alpha_optimizer = optax.adam(learning_rate=self.alpha_lr)
        else:
            self.alpha_optimizer = None

    def _init_train_state(self, key: jax.Array):
        """Initialize training state."""
        key, subkey = jax.random.split(key)

        # Initialize target critic parameters
        critic1_params, _ = eqx.partition(self.actor_critic.critic1, eqx.is_inexact_array)
        critic2_params, _ = eqx.partition(self.actor_critic.critic2, eqx.is_inexact_array)
        target_critic1_params = copy_params(critic1_params)
        target_critic2_params = copy_params(critic2_params)

        # Initialize optimizer states
        actor_params, _ = eqx.partition(self.actor_critic.actor, eqx.is_inexact_array)
        log_std_params, _ = eqx.partition(self.actor_critic.log_std_net, eqx.is_inexact_array)
        actor_opt_state = self.actor_optimizer.init((actor_params, log_std_params))
        critic_opt_state = self.critic_optimizer.init((critic1_params, critic2_params))

        # Initialize entropy coefficient
        if self.auto_entropy:
            log_ent_coef = jnp.log(jnp.array(self.init_ent_coef))
            alpha_opt_state = self.alpha_optimizer.init(log_ent_coef)
        else:
            log_ent_coef = None
            alpha_opt_state = None

        self.train_state = SACTrainState(
            model=self.actor_critic,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            alpha_opt_state=alpha_opt_state,
            log_ent_coef=log_ent_coef,
            key=subkey,
        )

    def _print_config(self):
        """Print SAC configuration."""
        print("\n🔍 SAC Configuration:")
        print(f"  Actor LR: {self.actor_lr}")
        print(f"  Critic LR: {self.critic_lr}")
        print(f"  Alpha LR: {self.alpha_lr}")
        print(f"  Tau: {self.tau}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Policy delay: {self.policy_delay}")
        print(f"  Target entropy: {self.target_entropy}")

        if self.auto_entropy:
            print(f"  Initial ent_coef: {self.init_ent_coef}")
        else:
            print(f"  Fixed ent_coef: {self.fixed_ent_coef}")

        print("\n🔍 Network architecture:")
        print(f"  Actor obs dim: {self.actor_critic.actor_obs_dim}")
        print(f"  Critic obs dim: {self.actor_critic.critic_obs_dim}")
        print(f"  Action dim: {self.actor_critic.num_actions}")

    @property
    def model(self) -> SACActorCritic:
        """Get current model."""
        return self.train_state.model

    def _get_ent_coef(self) -> jax.Array:
        """Get current entropy coefficient."""
        if self.auto_entropy:
            return jnp.exp(self.train_state.log_ent_coef)
        else:
            return jnp.array(self.fixed_ent_coef)

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
        self.transition = SACTransitionBuffer()

    def act(self, obs: ActInput, deterministic: bool = False) -> jax.Array:
        """
        Select action given observation.

        Args:
            obs: Current observation (ActInput)
            deterministic: If True, use mean action; otherwise sample

        Returns:
            Selected action
        """
        model = self.train_state.model
        key = self.train_state.key

        if deterministic:
            actions, values, new_key = act_deterministic(model, obs.actor_obs, obs.critic_obs, key)
        else:
            actions, values, new_key = act_stochastic(model, obs.actor_obs, obs.critic_obs, key)

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

    def update_normalizers(self, actor_obs: jax.Array, critic_obs: jax.Array) -> None:
        """Update obs normalizers with collected env-time observations.

        Called by the runner once per collection cycle so that every env-time
        observation contributes to the running statistics exactly once. This
        replaces the previous behavior of updating from sampled replay
        batches, which biased the statistics toward whatever transitions
        happened to be sampled most often.
        """
        if not self.obs_normalization:
            return
        new_model = _update_normalizers(self.train_state.model, actor_obs, critic_obs)
        self.train_state = self.train_state._replace(model=new_model)

    def update(self, batch: ReplayBatch) -> SACMetrics:
        """Update all networks using provided batch."""
        self.total_it += 1

        key = self.train_state.key
        key, critic_key, actor_key = jax.random.split(key, 3)

        ent_coef = self._get_ent_coef()

        # Update critics
        new_model, new_critic_opt_state, critic_info = update_critics(
            self.train_state.model,
            self.train_state.target_critic1_params,
            self.train_state.target_critic2_params,
            self.train_state.critic_opt_state,
            batch,
            self.critic_optimizer,
            self.gamma,
            ent_coef,
            critic_key,
        )

        self.train_state = self.train_state._replace(
            model=new_model,
            critic_opt_state=new_critic_opt_state,
            key=key,
        )

        # Update actor (with policy delay)
        if self.total_it % self.policy_delay == 0:
            new_model, new_actor_opt_state, log_prob, actor_info = update_actor(
                self.train_state.model,
                self.train_state.actor_opt_state,
                self.actor_optimizer,
                batch,
                ent_coef,
                actor_key,
            )

            actor_loss = float(actor_info["actor_loss"])
            entropy = float(actor_info["entropy"])

            self.train_state = self.train_state._replace(
                model=new_model,
                actor_opt_state=new_actor_opt_state,
            )

            # Update alpha (if auto-tuning)
            if self.auto_entropy:
                new_log_ent_coef, new_alpha_opt_state, alpha_loss, alpha_value = update_alpha(
                    self.train_state.log_ent_coef,
                    self.train_state.alpha_opt_state,
                    self.alpha_optimizer,
                    log_prob,
                    self.target_entropy,
                )

                alpha_loss = float(alpha_loss)
                alpha_value = float(alpha_value)

                self.train_state = self.train_state._replace(
                    log_ent_coef=new_log_ent_coef,
                    alpha_opt_state=new_alpha_opt_state,
                )
            else:
                alpha_loss = 0.0
                alpha_value = float(ent_coef)

            # Update target networks
            self._update_target_networks()
        else:
            actor_loss = 0.0
            entropy = 0.0
            alpha_loss = 0.0
            alpha_value = float(ent_coef)

        # Note: no explicit block_until_ready — _build_metrics calls float()
        # on the device-side losses, which forces the same sync naturally.
        # Build metrics
        metrics = self._build_metrics(critic_info, actor_loss, entropy, alpha_loss, alpha_value, batch)

        return metrics

    def _build_metrics(
        self,
        critic_info: Dict[str, jax.Array],
        actor_loss: float,
        entropy: float,
        alpha_loss: float,
        alpha_value: float,
        batch: ReplayBatch,
    ) -> SACMetrics:
        """Build metrics from update results."""
        return SACMetrics(
            critic=SACCriticMetrics(
                loss=float(critic_info["critic_loss"]),
                critic1_loss=float(critic_info["critic1_loss"]),
                critic2_loss=float(critic_info["critic2_loss"]),
                q1_mean=float(critic_info["q1_value"]),
                q2_mean=float(critic_info["q2_value"]),
                q1_std=float(critic_info["current_q1_std"]),
                q2_std=float(critic_info["current_q2_std"]),
                q_target_mean=float(critic_info["target_q_value"]),
            ),
            actor=ActorMetrics(
                loss=actor_loss,
                entropy=entropy,
            ),
            alpha=SACAlphaMetrics(
                value=alpha_value,
                loss=alpha_loss,
                target_entropy=float(self.target_entropy),
                entropy_gap=entropy - float(self.target_entropy),
            ),
            batch=SACBatchMetrics(
                reward_mean=float(batch.rewards.mean()),
                reward_std=float(batch.rewards.std()),
                reward_min=float(batch.rewards.min()),
                reward_max=float(batch.rewards.max()),
                action_mean=float(batch.actions.mean()),
                action_std=float(batch.actions.std()),
                terminated_ratio=float(batch.terminated.mean()),
            ),
            total_updates=self.total_it,
        )

    def _update_target_networks(self) -> None:
        """Update target critics with Polyak averaging."""
        critic1_params, _ = eqx.partition(self.train_state.model.critic1, eqx.is_inexact_array)
        critic2_params, _ = eqx.partition(self.train_state.model.critic2, eqx.is_inexact_array)

        new_target_critic1_params = polyak_update(
            critic1_params,
            self.train_state.target_critic1_params,
            self.tau,
        )
        new_target_critic2_params = polyak_update(
            critic2_params,
            self.train_state.target_critic2_params,
            self.tau,
        )

        self.train_state = self.train_state._replace(
            target_critic1_params=new_target_critic1_params,
            target_critic2_params=new_target_critic2_params,
        )

    def get_value(self, critic_obs: jax.Array) -> jax.Array:
        """Get value estimate for observations."""
        key = self.train_state.key
        key, subkey = jax.random.split(key)
        self.train_state = self.train_state._replace(key=key)
        return get_value(self.train_state.model, critic_obs, subkey)

    def save_train_state(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Save SAC training state."""
        model_path = os.path.join(checkpoint_dir, "model.eqx")
        eqx.tree_serialise_leaves(model_path, self.train_state.model)

        target_path = os.path.join(checkpoint_dir, "target_critics.eqx")
        target_critics = {
            "critic1": self.train_state.target_critic1_params,
            "critic2": self.train_state.target_critic2_params,
        }
        eqx.tree_serialise_leaves(target_path, target_critics)

        return {
            "alg_class": self.__class__.__name__,
            "alg_key": np.array(self.train_state.key),
            "log_ent_coef": np.array(self.train_state.log_ent_coef)
            if self.train_state.log_ent_coef is not None
            else None,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "total_it": self.total_it,
        }

    def load_train_state(self, checkpoint_dir: str, metadata: Dict[str, Any]) -> None:
        """Load SAC training state."""
        model_path = os.path.join(checkpoint_dir, "model.eqx")
        new_model = eqx.tree_deserialise_leaves(model_path, self.train_state.model)

        target_path = os.path.join(checkpoint_dir, "target_critics.eqx")
        target_critics_template = {
            "critic1": self.train_state.target_critic1_params,
            "critic2": self.train_state.target_critic2_params,
        }
        target_critics = eqx.tree_deserialise_leaves(target_path, target_critics_template)

        log_ent_coef = None
        if metadata.get("log_ent_coef") is not None:
            log_ent_coef = jnp.array(metadata["log_ent_coef"])

        new_actor_params, _ = eqx.partition(new_model.actor, eqx.is_inexact_array)
        new_log_std_params, _ = eqx.partition(new_model.log_std_net, eqx.is_inexact_array)
        new_critic1_params, _ = eqx.partition(new_model.critic1, eqx.is_inexact_array)
        new_critic2_params, _ = eqx.partition(new_model.critic2, eqx.is_inexact_array)

        actor_opt_state = self.actor_optimizer.init((new_actor_params, new_log_std_params))
        critic_opt_state = self.critic_optimizer.init((new_critic1_params, new_critic2_params))

        alpha_opt_state = None
        if self.auto_entropy and log_ent_coef is not None:
            alpha_opt_state = self.alpha_optimizer.init(log_ent_coef)

        self.train_state = SACTrainState(
            model=new_model,
            target_critic1_params=target_critics["critic1"],
            target_critic2_params=target_critics["critic2"],
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            alpha_opt_state=alpha_opt_state,
            log_ent_coef=log_ent_coef,
            key=jnp.array(metadata["alg_key"], dtype=jnp.uint32),
        )

        self.actor_lr = metadata.get("actor_lr", self.actor_lr)
        self.critic_lr = metadata.get("critic_lr", self.critic_lr)
        self.total_it = metadata.get("total_it", 0)
