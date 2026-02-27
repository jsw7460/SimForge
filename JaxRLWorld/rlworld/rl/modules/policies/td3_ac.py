from typing import TYPE_CHECKING, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import math

from rlworld.rl.modules.utils import MLP, orthogonal_init_mlp
from .base_ac import BaseActorCritic

if TYPE_CHECKING:
    pass

__all__ = ["TD3ActorCritic", "TD3QNetwork"]


# ==================== Q-Network ====================


class TD3QNetwork(eqx.Module):
    """Q-network for TD3."""
    net: MLP

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
        ortho_init: bool = True,
        output_gain: float = 1.0,
        *,
        key: jax.Array,
    ):
        input_dim = obs_dim + action_dim
        key1, key2 = jax.random.split(key)

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            output_activation=None,
            use_layer_norm=False,
            key=key1,
        )

        if ortho_init:
            gain_map = {
                "relu": math.sqrt(2),
                "elu": math.sqrt(2),
                "tanh": 1.0,
                "sigmoid": 1.0,
                "selu": 1.0,
            }
            hidden_gain = gain_map.get(activation, math.sqrt(2))
            self.net = orthogonal_init_mlp(
                self.net,
                hidden_gain=hidden_gain,
                output_gain=output_gain,
                key=key2,
            )

    def _forward_single(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], axis=-1)
        return self.net(x)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        if obs.ndim == 1:
            return self._forward_single(obs, action)
        else:
            return jax.vmap(self._forward_single)(obs, action)


# ==================== TD3 Actor-Critic ====================


class TD3ActorCritic(BaseActorCritic):
    """
    TD3 Actor-Critic with deterministic policy.

    Features:
    - Deterministic actor (outputs action directly, no sampling)
    - Twin Q-networks (critic1, critic2) for reduced overestimation
    - No log_std network (unlike SAC)
    - Actions bounded to [-1, 1] via tanh
    """
    critic1: TD3QNetwork
    critic2: TD3QNetwork

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_class_name: str = "MLPActor",
        kinematic_tree: "KinematicTree | None" = None,
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
            kinematic_tree: Optional kinematic tree for dynamics-aware actors
            key: JAX random key
            **kwargs: Must contain "actor_kwargs" and "critic_kwargs"
        """
        self.actor_obs_dim = num_actor_obs
        self.critic_obs_dim = num_critic_obs
        self.num_actions = num_actions
        self.is_squashed = True
        self.is_recurrent = False

        # Split keys
        key, key_actor, key_critic1, key_critic2 = jax.random.split(key, 4)

        # Build networks
        self._build_networks(
            actor_class_name=actor_class_name,
            kinematic_tree=kinematic_tree,
            key_actor=key_actor,
            key_critic1=key_critic1,
            key_critic2=key_critic2,
            **kwargs,
        )

        # Placeholder critic (required by BaseActorCritic)
        self.critic = self.critic1

        print(f"🎭 TD3 Actor-Critic: deterministic policy")
        print(f"🤖 Actor: {actor_class_name}")

    def _build_networks(
        self,
        actor_class_name: str,
        kinematic_tree: "KinematicTree | None",
        key_actor: jax.Array,
        key_critic1: jax.Array,
        key_critic2: jax.Array,
        **kwargs,
    ):
        """Build deterministic actor and twin critics."""
        actor_kwargs = kwargs["actor_kwargs"]
        critic_kwargs = kwargs["critic_kwargs"]

        # Build actor (deterministic - outputs action mean directly)
        self.actor = self._build_actor_common(
            actor_class_name=actor_class_name,
            actor_obs_dim=self.actor_obs_dim,
            num_actions=self.num_actions,
            actor_kwargs=actor_kwargs,
            kinematic_tree=kinematic_tree,
            key=key_actor,
        )

        # Build twin critics
        critic_hidden = critic_kwargs["hidden_dims"]
        critic_activation = critic_kwargs["activation"]
        ortho_init = critic_kwargs["ortho_init"]
        output_gain = critic_kwargs.get("output_gain", 0.01)

        self.critic1 = TD3QNetwork(
            obs_dim=self.critic_obs_dim,
            action_dim=self.num_actions,
            hidden_dims=critic_hidden,
            activation=critic_activation,
            ortho_init=ortho_init,
            output_gain=output_gain,
            key=key_critic1,
        )

        self.critic2 = TD3QNetwork(
            obs_dim=self.critic_obs_dim,
            action_dim=self.num_actions,
            hidden_dims=critic_hidden,
            activation=critic_activation,
            ortho_init=ortho_init,
            output_gain=output_gain,
            key=key_critic2,
        )

    def act(
        self,
        observations: jax.Array,
        *,
        key: jax.Array,
        deterministic: bool = True,
    ) -> Tuple[jax.Array, dict]:
        """
        Get action from deterministic policy.

        Args:
            observations: Actor observations
            key: JAX random key (unused for deterministic policy)
            deterministic: Ignored (TD3 is always deterministic)

        Returns:
            Tuple of (actions, aux_dict)
        """
        normalized_obs = self._normalize_actor_obs(observations)

        # Get action from actor
        if normalized_obs.ndim == 2:
            keys = jax.random.split(key, normalized_obs.shape[0])
            actions, aux = jax.vmap(self.actor)(normalized_obs, key=keys)
        else:
            actions, aux = self.actor(normalized_obs, key=key)

        # Apply tanh to bound actions to [-1, 1]
        actions = jnp.tanh(actions)

        return actions, aux

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
        actions, _ = self.act(actor_observations, key=key)
        q1 = self.critic1_forward(critic_observations, actions)
        q2 = self.critic2_forward(critic_observations, actions)
        return jnp.minimum(q1, q2)

    def evaluate_value(self, critic_obs: jax.Array) -> Tuple[jax.Array, dict]:
        """
        Evaluate value function (for compatibility with base class).

        Note: TD3 doesn't use V(s) directly. Returns zeros as placeholder.

        Args:
            critic_obs: Critic observations

        Returns:
            Tuple of (value, aux_dict)
        """
        normalized_obs = self._normalize_critic_obs(critic_obs)
        if normalized_obs.ndim == 1:
            return jnp.zeros((1,)), {}
        else:
            return jnp.zeros((normalized_obs.shape[0], 1)), {}

    @property
    def extra_to_log(self) -> dict:
        """Extra metrics to log."""
        extra = {}

        # Gate values (for gated encoders)
        if hasattr(self.actor, 'encoder') and hasattr(self.actor.encoder, 'last_gate'):
            gate = self.actor.encoder.last_gate
            if gate is not None:
                gate_mean = gate.mean(axis=0)

                extra["gate/mean"] = float(gate.mean())
                extra["gate/std"] = float(gate.std())

                for i in range(gate_mean.shape[0]):
                    extra[f"gate/per_action/action_{i:02d}"] = float(gate_mean[i])

        return extra
