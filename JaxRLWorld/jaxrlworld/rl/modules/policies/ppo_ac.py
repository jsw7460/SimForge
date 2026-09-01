from typing import TYPE_CHECKING, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxrlworld.rl.configs.common_config_classes import (
    ActorCfg,
    CriticCfg,
    DistributionType,
    StdType,
    VisionActorCfg,
    VisionCriticCfg,
)
from jaxrlworld.rl.modules.architectures.actor_registry import build_actor, build_critic
from jaxrlworld.rl.modules.distributions import GaussianDistribution, SquashedGaussianDistribution
from jaxrlworld.rl.modules.normalization import EmpiricalNormalization

from .base_ac import BaseActorCritic

if TYPE_CHECKING:
    from jaxrlworld.rl.configs.robots.kinematic_tree import KinematicTree

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

    def __init__(self, num_actions: int, init_std: float, log_std_min: float = -5.0, log_std_max: float = 2.0):
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

    distribution_type: DistributionType = eqx.field(static=True)
    std_type: StdType = eqx.field(static=True)
    init_noise_std: float = eqx.field(static=True)

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_cfg: ActorCfg,
        critic_cfg: CriticCfg,
        init_noise_std: float = 1.0,
        std_type: StdType = StdType.STATE_DEPENDENT,
        distribution_type: DistributionType = DistributionType.GAUSSIAN,
        kinematic_tree: Union["KinematicTree", None] = None,
        actuated_joint_names: "list[str] | None" = None,
        obs_normalization: bool = False,
        obs_shapes: "dict[str, tuple[int, ...]] | None" = None,
        actor_vector_group: str = "actor",
        critic_vector_group: str = "critic",
        *,
        key: jax.Array,
    ):
        """
        Args:
            num_actor_obs: Actor observation dimension
            num_critic_obs: Critic observation dimension
            num_actions: Action dimension
            actor_cfg: Typed actor cfg (MLPActorCfg / SpaceTimeTransformerActorCfg / ...)
            critic_cfg: Typed critic cfg
            init_noise_std: Initial action standard deviation
            std_type: StdType enum
            distribution_type: DistributionType enum
            kinematic_tree: Optional kinematic tree for dynamics-aware actors
            obs_normalization: If true, normalize observations
            obs_shapes: Per-env shape of every observation group. Required
                by vision cfgs, which read image groups the flat
                ``num_actor_obs`` cannot describe.
            actor_vector_group: Group holding the actor's state vector.
            critic_vector_group: Group holding the critic's state vector.
            key: JAX random key
        """
        self.actor_obs_dim = num_actor_obs
        self.critic_obs_dim = num_critic_obs
        self.num_actions = num_actions
        self.distribution_type = distribution_type
        self.is_squashed = distribution_type == DistributionType.SQUASHED_GAUSSIAN
        self.std_type = std_type
        self.init_noise_std = init_noise_std
        self.is_recurrent = False

        key_actor, key_critic, key_std = jax.random.split(key, 3)

        # A vision cfg reads a dict of observation groups; every other
        # cfg reads one array, and must not learn to.
        actor_is_vision = isinstance(actor_cfg, VisionActorCfg)
        critic_is_vision = isinstance(critic_cfg, VisionCriticCfg)
        if actor_is_vision != critic_is_vision:
            raise ValueError(
                "Actor and critic must agree on whether observations are a dict of groups: "
                f"got {type(actor_cfg).__name__} with {type(critic_cfg).__name__}."
            )
        if actor_is_vision:
            if obs_shapes is None:
                raise ValueError("A vision cfg needs obs_shapes.")
            if std_type == StdType.STATE_DEPENDENT:
                raise ValueError(
                    "state_dependent std is not defined for a vision policy: the std network would have to "
                    "read either the image or the state vector alone, and both are silent choices. "
                    "Use std_type='scalar', as mjlab's vision task does."
                )
        self.actor_vector_group = actor_vector_group if actor_is_vision else None
        self.critic_vector_group = critic_vector_group if critic_is_vision else None

        # Build actor + critic via the cfg-type-keyed builders.
        self.actor = build_actor(
            actor_cfg,
            num_obs=num_actor_obs,
            num_actions=num_actions,
            key=key_actor,
            kinematic_tree=kinematic_tree,
            actuated_joint_names=actuated_joint_names,
            obs_shapes=obs_shapes,
            vector_group=self.actor_vector_group,
        )
        self.critic = build_critic(
            critic_cfg,
            num_obs=num_critic_obs,
            key=key_critic,
            kinematic_tree=kinematic_tree,
            obs_shapes=obs_shapes,
            vector_group=self.critic_vector_group,
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

        print(
            f"🎲 PPO Actor-Critic: actor={type(actor_cfg).__name__}, distribution={distribution_type}, std={std_type}"
        )
        print(f"📏 Obs normalization: {obs_normalization}")

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
            print("📊 Using state-dependent std (neural network)")

        elif self.std_type == "state_independent":
            self.std_module = LearnableLogStd(
                num_actions=self.num_actions,
                init_std=self.init_noise_std,
            )
            print("🎚️  Using state-independent log_std (learnable)")

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
            print("🎚️ Using scalar std (learnable, no log transform)")

        else:
            raise ValueError(f"Unknown std_type: {self.std_type}")

    def get_current_std(self, observations: jax.Array | None = None) -> jax.Array:
        """Get the current standard deviation based on type."""
        if self.std_type == "state_dependent":
            return self.std_module(observations)
        elif self.std_type == "state_independent":
            log_std = jnp.clip(self.std_module.log_std, self.std_module.log_std_min, self.std_module.log_std_max)
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
        vector = self._actor_vector(normalized_obs)

        if vector.ndim == 2:
            keys = jax.random.split(key, vector.shape[0])
            mean, aux = jax.vmap(self.actor)(normalized_obs, key=keys)
        else:
            mean, aux = self.actor(normalized_obs, key=key)

        std = self.get_current_std(vector)
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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
        """Evaluate log probability, entropy, and base-Gaussian (mu, sigma) for given actions.

        For squashed distributions, actions should be raw (pre-tanh) values.
        This avoids numerically unstable arctanh inversion (matches Brax).
        Returned (mu, sigma) refer to the pre-squash Gaussian, matching the
        (old_mu, old_sigma) stored at rollout time so analytical KL is well-defined.
        """
        dist, aux = self.get_distribution(actor_obs, key=key)
        if dist.is_squashed:
            log_prob = dist.log_prob_raw(actions)
        else:
            log_prob = dist.log_prob(actions)
        # Squashed entropy is MC-estimated and needs an rng (split off the actor key);
        # the Gaussian path ignores it. key is always provided on the PPO update path.
        ent_key = None if key is None else jax.random.fold_in(key, 0)
        entropy = dist.entropy(ent_key)
        return log_prob, entropy, dist.mean, dist.std, aux

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
        entropy = dist.entropy(jax.random.fold_in(key, 1))
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
        if hasattr(self.actor, "post_update_step"):
            self.actor.post_update_step(*args, **kwargs)

    @property
    def extra_to_log(self) -> dict:
        """Extra metrics to log."""
        extra = {}
        if hasattr(self.actor, "extra_to_log"):
            extra.update(**self.actor.extra_to_log)
        return extra
