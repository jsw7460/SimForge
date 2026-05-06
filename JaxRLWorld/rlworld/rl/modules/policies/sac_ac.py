from typing import TYPE_CHECKING, Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.distributions import GaussianDistribution, SquashedGaussianDistribution
from rlworld.rl.modules.normalization import EmpiricalNormalization
from rlworld.rl.modules.utils import get_activation

from .base_ac import BaseActorCritic

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = ["SACActorCritic", "SACLogStdNetwork"]


# ==================== Log Std Network ====================


class SACLogStdNetwork(eqx.Module):
    """
    Neural network for learning state-dependent log standard deviations.

    Used in SAC for exploration control.
    """

    linears: list
    activation: str = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        hidden_dims: list[int],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        *,
        key: jax.Array,
    ):
        """
        Args:
            num_inputs: Input dimension (actor observation dim)
            num_outputs: Output dimension (action dim)
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            init_noise_std: Initial noise standard deviation (not log)
            log_std_min: Minimum log_std value (clipping)
            log_std_max: Maximum log_std value (clipping)
            key: JAX random key
        """
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.activation = activation

        # Convert to log space
        log_std_init = jnp.log(init_noise_std)

        # Build MLP layers
        self.linears = []
        dims = [num_inputs] + list(hidden_dims) + [num_outputs]

        for i in range(len(dims) - 1):
            key, subkey = jax.random.split(key)
            layer = eqx.nn.Linear(dims[i], dims[i + 1], key=subkey)

            # Initialize output layer bias to log_std_init
            if i == len(dims) - 2:
                layer = eqx.tree_at(
                    lambda l: l.bias,
                    layer,
                    jnp.full_like(layer.bias, log_std_init),
                )

            self.linears.append(layer)

    def _forward_single(self, x: jax.Array) -> jax.Array:
        """Forward pass for single input."""
        act_fn = get_activation(self.activation)

        for i, linear in enumerate(self.linears):
            x = linear(x)
            # Apply activation except for output layer
            if i < len(self.linears) - 1:
                x = act_fn(x)
        # Clamp log_std
        return jnp.clip(x, self.log_std_min, self.log_std_max)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with automatic batching."""
        if x.ndim == 1:
            return self._forward_single(x)
        else:
            return jax.vmap(self._forward_single)(x)


# ==================== Twin Q-Network ====================


class SACQNetwork(eqx.Module):
    """
    Q-network for SAC.

    Takes (observation, action) as input and outputs Q-value.
    """

    linears: list
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        activation: str = "elu",
        *,
        key: jax.Array,
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            key: JAX random key
        """
        input_dim = obs_dim + action_dim
        self.activation = activation
        self.linears = []
        dims = [input_dim] + list(hidden_dims) + [1]

        for i in range(len(dims) - 1):
            key, subkey = jax.random.split(key)
            layer = eqx.nn.Linear(dims[i], dims[i + 1], key=subkey)
            self.linears.append(layer)

    def _forward_single(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Forward pass for single input."""
        x = jnp.concatenate([obs, action], axis=-1)
        act_fn = get_activation(self.activation)

        for i, linear in enumerate(self.linears):
            x = linear(x)
            # Apply activation except for output layer
            if i < len(self.linears) - 1:
                x = act_fn(x)
        return x

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Forward pass with automatic batching."""
        if obs.ndim == 1:
            return self._forward_single(obs, action)
        else:
            return jax.vmap(self._forward_single)(obs, action)


# ==================== SAC Actor-Critic ====================


class SACActorCritic(BaseActorCritic):
    """
    SAC Actor-Critic with separate log_std network (compatible with all actors).

    Features:
    - Twin Q-networks (critic1, critic2) for reduced overestimation
    - Separate log_std network for state-dependent exploration
    - Squashed Gaussian distribution for bounded actions
    """

    log_std_net: SACLogStdNetwork
    critic1: SACQNetwork
    critic2: SACQNetwork

    distribution_type: str = eqx.field(static=True)
    init_noise_std: float = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)

    # Runtime state (not part of model parameters)
    _distribution: Any = eqx.field(static=True, default=None)
    _action_mean: Any = eqx.field(static=True, default=None)

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_class_name: str = "MLPActor",
        distribution_type: str = "squashed_gaussian",
        init_noise_std: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        kinematic_tree: "KinematicTree | None" = None,
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
            actor_class_name: Name of actor class ("MLPActor", etc.)
            distribution_type: "gaussian" or "squashed_gaussian"
            init_noise_std: Initial noise standard deviation (not log)
            log_std_min: Minimum log_std (clipping)
            log_std_max: Maximum log_std (clipping)
            kinematic_tree: Optional kinematic tree for dynamics-aware actors
            obs_normalization: Whether to enable observation normalization
            key: JAX random key
            **kwargs: Must contain "actor_kwargs" and "critic_kwargs"
        """
        self.actor_obs_dim = num_actor_obs
        self.critic_obs_dim = num_critic_obs
        self.num_actions = num_actions
        self.distribution_type = distribution_type
        self.is_squashed = distribution_type == "squashed_gaussian"
        self.init_noise_std = init_noise_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.is_recurrent = False

        # Split keys
        key, key_actor, key_log_std, key_critic1, key_critic2 = jax.random.split(key, 5)

        # Build networks
        self._build_networks(
            actor_class_name=actor_class_name,
            kinematic_tree=kinematic_tree,
            key_actor=key_actor,
            key_log_std=key_log_std,
            key_critic1=key_critic1,
            key_critic2=key_critic2,
            **kwargs,
        )

        # Placeholder critic (not used, but required by BaseActorCritic)
        # We use critic1 and critic2 instead
        self.critic = self.critic1

        # Observation normalization
        if obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(shape=num_actor_obs)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=num_critic_obs)
        else:
            self.actor_obs_normalizer = None
            self.critic_obs_normalizer = None

        print(f"🎭 SAC Actor-Critic: distribution={distribution_type}")
        print(f"🤖 Actor: {actor_class_name}")
        print(f"📏 Obs normalization: {obs_normalization}")

    def _build_networks(
        self,
        actor_class_name: str,
        kinematic_tree: "KinematicTree | None",
        key_actor: jax.Array,
        key_log_std: jax.Array,
        key_critic1: jax.Array,
        key_critic2: jax.Array,
        **kwargs,
    ):
        """Build actor, log_std network, and twin critics."""
        actor_kwargs = kwargs.get("actor_kwargs", {})
        critic_kwargs = kwargs.get("critic_kwargs", {})

        # Build actor
        self.actor = self._build_actor_common(
            actor_class_name=actor_class_name,
            actor_obs_dim=self.actor_obs_dim,
            num_actions=self.num_actions,
            actor_kwargs=actor_kwargs,
            kinematic_tree=kinematic_tree,
            key=key_actor,
        )
        # Build separate log_std network
        self.log_std_net = SACLogStdNetwork(
            num_inputs=self.actor_obs_dim,
            num_outputs=self.num_actions,
            hidden_dims=actor_kwargs.get("hidden_dims", [256, 256]),
            activation=actor_kwargs.get("activation", "elu"),
            init_noise_std=self.init_noise_std,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            key=key_log_std,
        )

        # Build twin critics
        self.critic1 = SACQNetwork(
            obs_dim=self.critic_obs_dim,
            action_dim=self.num_actions,
            hidden_dims=critic_kwargs.get("hidden_dims", [256, 256]),
            activation=critic_kwargs.get("activation", "elu"),
            key=key_critic1,
        )

        self.critic2 = SACQNetwork(
            obs_dim=self.critic_obs_dim,
            action_dim=self.num_actions,
            hidden_dims=critic_kwargs.get("hidden_dims", [256, 256]),
            activation=critic_kwargs.get("activation", "elu"),
            key=key_critic2,
        )

    def _get_actor_distribution(
        self,
        observations: jax.Array,
        *,
        key: jax.Array,
    ) -> Tuple[GaussianDistribution | SquashedGaussianDistribution, jax.Array, dict]:
        """
        Compute actor distribution from observations.

        Returns:
            Tuple of (distribution, mean, aux_dict)
        """
        normalized_obs = self._normalize_actor_obs(observations)

        # Get mean from actor
        if normalized_obs.ndim == 2:
            keys = jax.random.split(key, normalized_obs.shape[0])
            mean, aux = jax.vmap(self.actor)(normalized_obs, key=keys)
        else:
            mean, aux = self.actor(normalized_obs, key=key)

        # Get log_std from separate network
        log_std = self.log_std_net(normalized_obs)
        std = jnp.exp(log_std)

        # Create distribution
        if self.distribution_type == "gaussian":
            dist = GaussianDistribution(mean, std)
        elif self.distribution_type == "squashed_gaussian":
            dist = SquashedGaussianDistribution(mean, std)
        else:
            raise ValueError(f"Unknown distribution_type: {self.distribution_type}")

        return dist, mean, aux

    def act(
        self,
        observations: jax.Array,
        *,
        key: jax.Array,
        deterministic: bool = False,
    ) -> Tuple[jax.Array, dict]:
        """
        Sample action from policy.

        Args:
            observations: Actor observations
            key: JAX random key
            deterministic: If True, return mean action

        Returns:
            Tuple of (actions, aux_dict)
        """
        dist, mean, aux = self._get_actor_distribution(observations, key=key)

        if deterministic:
            if self.distribution_type == "squashed_gaussian":
                actions = jnp.tanh(mean)
            else:
                actions = mean
        else:
            actions = dist.sample(key)

        return actions, aux

    def act_with_log_prob(
        self,
        observations: jax.Array,
        *,
        key: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, dict]:
        """
        Sample action and compute log probability.

        Args:
            observations: Actor observations
            key: JAX random key

        Returns:
            Tuple of (actions, log_prob, aux_dict)
        """
        dist, mean, aux = self._get_actor_distribution(observations, key=key)

        # Use rsample_with_log_prob for SquashedGaussian (more numerically stable)
        if isinstance(dist, SquashedGaussianDistribution):
            actions, log_prob = dist.rsample_with_log_prob(key)
        else:
            actions = dist.sample(key)
            log_prob = dist.log_prob(actions)

        return actions, log_prob, aux

    def get_actions_log_prob(
        self,
        observations: jax.Array,
        actions: jax.Array,
        *,
        key: jax.Array,
    ) -> jax.Array:
        """
        Compute log probability of given actions.

        Args:
            observations: Actor observations
            actions: Actions to evaluate
            key: JAX random key

        Returns:
            Log probabilities
        """
        dist, _, _ = self._get_actor_distribution(observations, key=key)
        return dist.log_prob(actions)

    def critic1_forward(
        self,
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """
        Forward pass for first Q-network.

        Args:
            observations: Critic observations
            actions: Actions

        Returns:
            Q-values from critic1
        """
        normalized_obs = self._normalize_critic_obs(observations)
        return self.critic1(normalized_obs, actions)

    def critic2_forward(
        self,
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """
        Forward pass for second Q-network.

        Args:
            observations: Critic observations
            actions: Actions

        Returns:
            Q-values from critic2
        """
        normalized_obs = self._normalize_critic_obs(observations)
        return self.critic2(normalized_obs, actions)

    def evaluate(
        self,
        actor_observations: jax.Array,
        critic_observations: jax.Array,
        *,
        key: jax.Array,
    ) -> jax.Array:
        """
        Evaluate state value using minimum of two critics.

        Args:
            actor_observations: Actor observations
            critic_observations: Critic observations
            key: JAX random key

        Returns:
            Minimum Q-value (state value estimate)
        """
        actions, _ = self.act(actor_observations, key=key, deterministic=True)
        q1 = self.critic1_forward(critic_observations, actions)
        q2 = self.critic2_forward(critic_observations, actions)
        return jnp.minimum(q1, q2)

    def evaluate_value(self, critic_obs: jax.Array) -> Tuple[jax.Array, dict]:
        """
        Evaluate value function (for compatibility with base class).

        Note: This uses critic1 only. For proper SAC value estimation,
        use evaluate() which takes the minimum of both critics.

        Args:
            critic_obs: Critic observations

        Returns:
            Tuple of (value, aux_dict)
        """
        # For SAC, we need an action to evaluate Q-value
        # This is a simplified version that returns zeros
        # Use evaluate() for proper value estimation
        normalized_obs = self._normalize_critic_obs(critic_obs)
        # Return zeros as placeholder - SAC doesn't use V(s) directly
        if normalized_obs.ndim == 1:
            return jnp.zeros((1,)), {}
        else:
            return jnp.zeros((normalized_obs.shape[0], 1)), {}

    def act_inference(self, actor_obs: jax.Array, *, key: jax.Array) -> tuple[jax.Array, dict]:
        """Deterministic action for inference."""
        actions, aux = self.act(actor_obs, key=key, deterministic=True)
        return actions, aux

    @property
    def extra_to_log(self) -> dict:
        """Extra metrics to log."""
        extra = {}

        # Gate values (for gated encoders)
        if hasattr(self.actor, "encoder") and hasattr(self.actor.encoder, "last_gate"):
            gate = self.actor.encoder.last_gate
            if gate is not None:
                gate_mean = gate.mean(axis=0)

                extra["gate/mean"] = float(gate.mean())
                extra["gate/std"] = float(gate.std())

                for i in range(gate_mean.shape[0]):
                    extra[f"gate/per_action/action_{i:02d}"] = float(gate_mean[i])

        return extra
