from typing import Union, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.architectures.mlp import MLPCritic
from rlworld.rl.modules.architectures.space_time_transformer.critic import (
    SpaceTimeTransformerCritic,
)
from rlworld.rl.modules.distributions import GaussianDistribution, SquashedGaussianDistribution
from rlworld.rl.modules.normalization import EmpiricalNormalization
from .base_ac import BaseActorCritic

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree

__all__ = [
    "PPOActorCritic",
    "StdNetwork",
    "ConstantStd",
    "LearnableLogStd",
]


# ==================== Std Modules ====================

class LearnableStd(eqx.Module):
    std: jax.Array

    def __init__(self, num_actions: int, init_std: float):
        self.std = jnp.full(num_actions, init_std)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_shape = x.shape[:-1]
        return jnp.broadcast_to(self.std, batch_shape + self.std.shape)


class StdNetwork(eqx.Module):
    """Neural network for learning state-dependent action standard deviations."""
    linear: eqx.nn.Linear
    min_std: float = eqx.field(static=True)
    max_std: float = eqx.field(static=True)

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        init_std: float = 1.0,
        min_std: float = 0.05,
        max_std: float = 2.0,
        *,
        key: jax.Array,
    ):
        self.min_std = min_std
        self.max_std = max_std
        self.linear = eqx.nn.Linear(num_inputs, num_outputs, key=key)

        target_bias = jnp.log(jnp.exp(init_std - min_std) - 1)

        self.linear = eqx.tree_at(
            lambda l: l.weight,
            self.linear,
            self.linear.weight * 0.005,
        )
        self.linear = eqx.tree_at(
            lambda l: l.bias,
            self.linear,
            jnp.full_like(self.linear.bias, target_bias),
        )

    def _forward_single(self, x: jax.Array) -> jax.Array:
        return jnp.clip(jax.nn.softplus(self.linear(x)) + self.min_std, max=self.max_std)

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim == 1:
            return self._forward_single(x)
        else:
            return jax.vmap(self._forward_single)(x)


class ConstantStd(eqx.Module):
    """Fixed (non-learnable) standard deviation."""
    std: jax.Array

    def __init__(self, num_actions: int, init_std: float):
        self.std = jnp.full(num_actions, init_std)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_shape = x.shape[:-1]
        return jnp.broadcast_to(self.std, batch_shape + self.std.shape)


class LearnableLogStd(eqx.Module):
    """Learnable state-independent log standard deviation."""
    log_std: jax.Array
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)

    def __init__(self, num_actions: int, init_std: float,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        self.log_std = jnp.full(num_actions, jnp.log(init_std))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def __call__(self, x: jax.Array) -> jax.Array:
        log_std = jnp.clip(self.log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)
        batch_shape = x.shape[:-1]
        return jnp.broadcast_to(std, batch_shape + std.shape)


# ==================== PPO Actor-Critic ====================

class PPOActorCritic(BaseActorCritic):
    """
    PPO Actor-Critic with Gaussian/SquashedGaussian distributions.

    Equivalent to PyTorch PPOActorCritic.
    """
    std_module: StdNetwork | ConstantStd | LearnableLogStd | LearnableStd

    distribution_type: str = eqx.field(static=True)
    std_type: str = eqx.field(static=True)
    init_noise_std: float = eqx.field(static=True)

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_class_name: str = "MLPActor",
        critic_class_name: str = "MLPCritic",
        init_noise_std: float = 1.0,
        std_type: str = "state_dependent",
        distribution_type: str = "gaussian",
        kinematic_tree: Union["KinematicTree", None] = None,
        actuated_joint_names: "list[str] | None" = None,
        obs_normalization: bool = False,
        *,
        key: jax.Array,
        **kwargs,
    ):
        """
        Args:
            num_actor_obs: Actor observation dimension
            num_critic_obs: Critic observation dimension
            num_actions: Action dimension
            actor_class_name: Name of actor class ("MLPActor", "SpaceTimeTransformerActor", etc.)
            init_noise_std: Initial action standard deviation
            std_type: "state_dependent", "state_independent", or "fixed"
            distribution_type: "gaussian" or "squashed_gaussian"
            kinematic_tree: Optional kinematic tree for dynamics-aware actors
            obs_normalization: If true, normalize observations
            key: JAX random key
            **kwargs: Must contain "actor_kwargs" and "critic_kwargs"
        """
        self.actor_obs_dim = num_actor_obs
        self.critic_obs_dim = num_critic_obs
        self.num_actions = num_actions
        self.distribution_type = distribution_type
        self.is_squashed = (distribution_type == "squashed_gaussian")
        self.std_type = std_type
        self.init_noise_std = init_noise_std
        self.is_recurrent = False

        key_actor, key_critic, key_std = jax.random.split(key, 3)

        # Build networks
        self._build_networks(
            actor_class_name,
            critic_class_name,
            kinematic_tree,
            actuated_joint_names,
            key_actor,
            key_critic,
            **kwargs,
        )

        # Initialize std
        self._initialize_std(key_std)

        # Initialize observation normalizers
        if obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(shape=num_actor_obs)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=num_critic_obs)
        else:
            self.actor_obs_normalizer = None
            self.critic_obs_normalizer = None

        print(f"🎲 PPO Actor-Critic: actor={actor_class_name}, distribution={distribution_type}, std={std_type}")
        print(f"📏 Obs normalization: {obs_normalization}")

    def _build_networks(
        self,
        actor_class_name: str,
        critic_class_name: str,
        kinematic_tree: "KinematicTree | None",
        actuated_joint_names: "list[str] | None",
        key_actor: jax.Array,
        key_critic: jax.Array,
        **kwargs,
    ):
        """Build actor and critic networks."""
        critic_kwargs = kwargs["critic_kwargs"]
        if critic_class_name == "MLPCritic":
            self.critic = MLPCritic(
                num_obs=self.critic_obs_dim,
                hidden_dims=critic_kwargs["hidden_dims"],
                activation=critic_kwargs["activation"],
                use_layer_norm=critic_kwargs.get("use_layer_norm", False),
                ortho_init=critic_kwargs["ortho_init"],
                key=key_critic,
            )
        elif critic_class_name == "SpaceTimeTransformerCritic":
            if kinematic_tree is None:
                raise ValueError(
                    "SpaceTimeTransformerCritic requires kinematic_tree."
                )
            stc_kwargs = {
                k: v for k, v in critic_kwargs.items()
                if k not in ("num_obs", "key", "kinematic_tree")
            }
            self.critic = SpaceTimeTransformerCritic(
                kinematic_tree=kinematic_tree,
                num_obs=self.critic_obs_dim,
                key=key_critic,
                **stc_kwargs,
            )
        else:
            raise ValueError(f"Unknown critic_class_name: {critic_class_name!r}")

        # Build actor using common logic
        actor_kwargs = kwargs.get("actor_kwargs", {})
        self.actor = self._build_actor_common(
            actor_class_name=actor_class_name,
            actor_obs_dim=self.actor_obs_dim,
            num_actions=self.num_actions,
            actor_kwargs=actor_kwargs,
            kinematic_tree=kinematic_tree,
            actuated_joint_names=actuated_joint_names,
            key=key_actor,
        )

    def _initialize_std(self, key: jax.Array):
        """Initialize the standard deviation based on the selected type."""
        if self.std_type == "state_dependent":
            self.std_module = StdNetwork(
                num_inputs=self.actor_obs_dim,
                num_outputs=self.num_actions,
                init_std=self.init_noise_std,
                min_std=0.05,
                key=key,
            )
            print(f"📊 Using state-dependent std (neural network)")

        elif self.std_type == "state_independent":
            self.std_module = LearnableLogStd(
                num_actions=self.num_actions,
                init_std=self.init_noise_std,
            )
            print(f"🎚️  Using state-independent log_std (learnable)")

        elif self.std_type == "fixed":
            self.std_module = ConstantStd(
                num_actions=self.num_actions,
                init_std=self.init_noise_std,
            )
            print(f"🔒 Using fixed std (constant={self.init_noise_std:.4f})")

        elif self.std_type == "scalar":
            self.std_module = LearnableStd(
                num_actions=self.num_actions,
                init_std=self.init_noise_std,
            )
            print(f"🎚️ Using scalar std (learnable, no log transform)")

        else:
            raise ValueError(f"Unknown std_type: {self.std_type}")

    def get_current_std(self, observations: jax.Array | None = None) -> jax.Array:
        """Get the current standard deviation based on type."""
        if self.std_type == "state_dependent":
            return self.std_module(observations)
        elif self.std_type == "state_independent":
            log_std = jnp.clip(self.std_module.log_std,
                               self.std_module.log_std_min,
                               self.std_module.log_std_max)
            std = jnp.exp(log_std)
            if observations is not None:
                return jnp.broadcast_to(std, (observations.shape[0],) + std.shape)
            return std
        elif self.std_type == "scalar":
            std = self.std_module.std
            if observations is not None:
                return jnp.broadcast_to(std, (observations.shape[0],) + std.shape)
            return std
        elif self.std_type == "fixed":
            std = self.std_module.std
            if observations is not None:
                return jnp.broadcast_to(std, (observations.shape[0],) + std.shape)
            return std

    def get_distribution(
        self, actor_obs: jax.Array, *, key: jax.Array
    ) -> tuple[GaussianDistribution | SquashedGaussianDistribution, dict]:
        """Create action distribution from observations."""
        normalized_obs = self._normalize_actor_obs(actor_obs)

        if normalized_obs.ndim == 2:
            keys = jax.random.split(key, normalized_obs.shape[0])
            mean, aux = jax.vmap(self.actor)(normalized_obs, key=keys)
        else:
            mean, aux = self.actor(normalized_obs, key=key)

        std = self.get_current_std(normalized_obs)
        # std = jnp.clip(std, 1e-3, 5.0)

        if self.distribution_type == "gaussian":
            return GaussianDistribution(mean, std), aux
        elif self.distribution_type == "squashed_gaussian":
            return SquashedGaussianDistribution(mean, std), aux
        else:
            raise ValueError(f"Unknown distribution_type: {self.distribution_type}")

    def act(
        self,
        actor_obs: jax.Array,
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jax.Array, dict]:
        """Sample action from policy."""
        dist, aux = self.get_distribution(actor_obs, key=key)

        if deterministic:
            if self.distribution_type == "squashed_gaussian":
                return jnp.tanh(dist.mean), aux
            return dist.mean, aux

        return dist.sample(key), aux

    def get_actions_log_prob(self, actions: jax.Array) -> jax.Array:
        """Get log probability of actions (requires get_distribution called first)."""
        # Note: In JAX we need to recompute distribution
        raise NotImplementedError("Use evaluate_actions instead")

    def evaluate_actions(
        self,
        actor_obs: jax.Array,
        actions: jax.Array,
        *,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, dict]:
        """Evaluate log probability and entropy for given actions.

        For squashed distributions, actions should be raw (pre-tanh) values.
        This avoids numerically unstable arctanh inversion (matches Brax).
        """
        dist, aux = self.get_distribution(actor_obs, key=key)
        if dist.is_squashed:
            log_prob = dist.log_prob_raw(actions)
        else:
            log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, aux

    def evaluate_value(self, critic_obs: jax.Array) -> tuple[jax.Array, dict]:
        """Evaluate value function."""
        normalized_obs = self._normalize_critic_obs(critic_obs)
        value, aux = self.critic(normalized_obs)
        return value, aux

    def act_and_value(
        self,
        actor_obs: jax.Array,
        critic_obs: jax.Array,
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
        """Get action, value, log_prob, and entropy in one call."""
        dist, actor_aux = self.get_distribution(actor_obs, key=key)

        if deterministic:
            if self.distribution_type == "squashed_gaussian":
                action = jnp.tanh(dist.mean)
            else:
                action = dist.mean
        else:
            action = dist.sample(key)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value, critic_aux = self.evaluate_value(critic_obs)

        # Merge aux dicts
        aux = {**actor_aux, **critic_aux}
        return action, value, log_prob, entropy, aux

    def act_inference(self, actor_obs: jax.Array, *, key: jax.Array) -> tuple[jax.Array, dict]:
        """Get deterministic action for inference."""
        normalized_obs = self._normalize_actor_obs(actor_obs)
        actions_mean, aux = self.actor(normalized_obs, key=key)

        if self.distribution_type == "squashed_gaussian":
            actions_mean = jnp.tanh(actions_mean)

        return actions_mean, aux

    def post_update_step(self, *args, **kwargs):
        """Placeholder for compatibility."""
        if hasattr(self.actor, 'post_update_step'):
            self.actor.post_update_step(*args, **kwargs)

    @property
    def extra_to_log(self) -> dict:
        """Extra metrics to log."""
        extra = {}
        if hasattr(self.actor, 'extra_to_log'):
            extra.update(**self.actor.extra_to_log)
        return extra
