import math
from typing import TYPE_CHECKING, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.normalization import EmpiricalNormalization
from rlworld.rl.modules.policies.base_ac import BaseActorCritic
from rlworld.rl.modules.utils import MLP, orthogonal_init_mlp

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree

__all__ = ["FastTD3ActorCritic", "DistributionalQNetwork"]


# ==================== Distributional Q-Network (C51) ====================


class DistributionalQNetwork(eqx.Module):
    """Distributional Q-network using C51 algorithm."""

    net: MLP
    num_atoms: int
    v_min: float
    v_max: float

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        activation: str = "relu",
        ortho_init: bool = True,
        output_gain: float = 1.0,
        init_value_range: tuple[float, float] | None = None,  # e.g., (-500, -200)
        *,
        key: jax.Array,
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions (e.g., [1024, 512, 256])
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value support
            v_max: Maximum value support
            activation: Activation function
            ortho_init: Whether to use orthogonal initialization for hidden layers.
                        NOTE: If init_value_range is set, output layer will be
                        overwritten regardless of this setting.
            output_gain: Output layer initialization gain (ignored if init_value_range is set)
            init_value_range: If set, initialize output layer so that the probability
                              distribution is concentrated around this range.
                              e.g., (-500, -200) means initial Q-distribution peaks
                              between -500 and -200.
                              NOTE: This overwrites the output layer weights/bias,
                              so ortho_init and output_gain have no effect on output layer.
            key: JAX random key
        """
        input_dim = obs_dim + action_dim
        key1, key2, key3 = jax.random.split(key, 3)

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_atoms,
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

        # Initialize output layer bias to concentrate probability on init_value_range
        if init_value_range is not None:
            output_layer = self.net.linears[-1]

            # Zero out weights for near-constant output
            small_weight = jax.random.normal(key3, output_layer.weight.shape) * 0.01

            # Create bias that peaks in the target range
            support = jnp.linspace(v_min, v_max, num_atoms)
            range_low, range_high = init_value_range
            range_center = (range_low + range_high) / 2.0
            range_width = (range_high - range_low) / 2.0

            # Gaussian-like bias centered on target range
            bias = jnp.exp(-0.5 * ((support - range_center) / range_width) ** 2)
            bias = bias * 3.0  # Scale to make softmax more peaked

            new_output_layer = eqx.tree_at(
                lambda l: (l.weight, l.bias),
                output_layer,
                (small_weight, bias),
            )
            self.net = eqx.tree_at(
                lambda n: n.linears[-1],
                self.net,
                new_output_layer,
            )

    @property
    def delta_z(self) -> float:
        """Atom spacing."""
        return (self.v_max - self.v_min) / (self.num_atoms - 1)

    @property
    def support(self) -> jax.Array:
        """Value support atoms."""
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    def _forward_single(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Forward pass for single observation."""
        x = jnp.concatenate([obs, action], axis=-1)
        return self.net(x)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """
        Forward pass returning logits.

        Args:
            obs: Observations [..., obs_dim]
            action: Actions [..., action_dim]

        Returns:
            Logits [..., num_atoms]
        """
        if obs.ndim == 1:
            return self._forward_single(obs, action)
        else:
            return jax.vmap(self._forward_single)(obs, action)

    def get_probs(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Get probability distribution over atoms."""
        logits = self(obs, action)
        return jax.nn.softmax(logits, axis=-1)

    def get_value(self, probs: jax.Array) -> jax.Array:
        """
        Compute expected value from probability distribution.

        Args:
            probs: Probability distribution [..., num_atoms]

        Returns:
            Expected Q-value [...,]
        """
        return jnp.sum(probs * self.support, axis=-1)

    def get_q_value(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Get Q-value directly from observations and actions."""
        probs = self.get_probs(obs, action)
        return self.get_value(probs)


def project_distribution(
    next_probs: jax.Array,
    rewards: jax.Array,
    bootstrap: jax.Array,
    discount: jax.Array,
    num_atoms: int,
    v_min: float,
    v_max: float,
) -> jax.Array:
    """
    Project target distribution onto fixed support.

    Args:
        next_probs: Next state probabilities [batch, num_atoms]
        rewards: Rewards [batch,]
        bootstrap: Bootstrap mask (1 if not terminal) [batch,]
        discount: Discount factor (gamma^n for n-step) [batch,]
        num_atoms: Number of atoms
        v_min: Minimum value
        v_max: Maximum value

    Returns:
        Projected distribution [batch, num_atoms]
    """
    delta_z = (v_max - v_min) / (num_atoms - 1)
    support = jnp.linspace(v_min, v_max, num_atoms)

    # Compute target support: r + gamma * z
    # rewards: [batch,], bootstrap: [batch,], discount: [batch,], support: [num_atoms,]
    target_z = rewards[:, None] + bootstrap[:, None] * discount[:, None] * support[None, :]
    target_z = jnp.clip(target_z, v_min, v_max)

    # Compute projection indices
    b = (target_z - v_min) / delta_z
    l = jnp.floor(b).astype(jnp.int32)
    u = jnp.ceil(b).astype(jnp.int32)

    # Handle l == u edge case: ensure l != u so probability mass is not lost
    is_int = l == u
    l = jnp.where(is_int & (l > 0), l - 1, l)
    u = jnp.where(is_int & (u < num_atoms - 1), u + 1, u)

    # Clamp indices
    l = jnp.clip(l, 0, num_atoms - 1)
    u = jnp.clip(u, 0, num_atoms - 1)

    # Distribute probability mass
    batch_size = rewards.shape[0]
    proj_dist = jnp.zeros((batch_size, num_atoms))

    # Lower projection
    lower_weight = next_probs * (u.astype(jnp.float32) - b)
    # Upper projection
    upper_weight = next_probs * (b - l.astype(jnp.float32))

    # Scatter add using segment_sum approach
    def scatter_add_row(carry, inputs):
        proj_row, l_idx, u_idx, l_weight, u_weight = inputs
        proj_row = proj_row.at[l_idx].add(l_weight)
        proj_row = proj_row.at[u_idx].add(u_weight)
        return None, proj_row

    _, proj_dist = jax.lax.scan(
        scatter_add_row,
        None,
        (jnp.zeros((batch_size, num_atoms)), l, u, lower_weight, upper_weight),
    )

    return proj_dist


def project_distribution_batched(
    next_probs: jax.Array,
    rewards: jax.Array,
    bootstrap: jax.Array,
    discount: jax.Array,
    num_atoms: int,
    v_min: float,
    v_max: float,
) -> jax.Array:
    """
    Vectorized projection of target distribution.
    """
    delta_z = (v_max - v_min) / (num_atoms - 1)
    support = jnp.linspace(v_min, v_max, num_atoms)

    # Squeeze to [batch] if needed
    rewards = jnp.squeeze(rewards)
    bootstrap = jnp.squeeze(bootstrap)
    discount = jnp.squeeze(discount)

    # Compute target support: r + gamma * z [batch, num_atoms]
    target_z = rewards[:, None] + bootstrap[:, None] * discount[:, None] * support[None, :]
    target_z = jnp.clip(target_z, v_min, v_max)

    # Compute projection indices [batch, num_atoms]
    b = (target_z - v_min) / delta_z
    l = jnp.floor(b).astype(jnp.int32)
    u = jnp.ceil(b).astype(jnp.int32)

    # Handle l == u edge case: ensure l != u so probability mass is not lost
    # When l == u and l > 0: shift l down
    # When l == u and l == 0: shift u up
    is_int = l == u
    l = jnp.where(is_int & (l > 0), l - 1, l)
    u = jnp.where(is_int & (u < num_atoms - 1), u + 1, u)

    # Clamp indices
    l = jnp.clip(l, 0, num_atoms - 1)
    u = jnp.clip(u, 0, num_atoms - 1)

    # Compute weights [batch, num_atoms]
    u_weight = b - l.astype(jnp.float32)
    l_weight = 1.0 - u_weight

    # Weighted probabilities [batch, num_atoms]
    l_weighted_probs = next_probs * l_weight
    u_weighted_probs = next_probs * u_weight

    # Use one-hot for scatter [batch, num_atoms, num_atoms]
    l_one_hot = jax.nn.one_hot(l, num_atoms)
    u_one_hot = jax.nn.one_hot(u, num_atoms)

    # proj[batch, dst] = sum over src of (weighted_prob[batch, src] * one_hot[batch, src, dst])
    proj_dist = jnp.sum(l_weighted_probs[:, :, None] * l_one_hot, axis=1) + jnp.sum(
        u_weighted_probs[:, :, None] * u_one_hot, axis=1
    )

    return proj_dist


# ==================== FastTD3 Actor-Critic ====================


class FastTD3ActorCritic(BaseActorCritic):
    """
    FastTD3 Actor-Critic with Distributional Critics (C51).

    Key differences from standard TD3:
    - Uses distributional critics (C51) instead of scalar Q-values
    - Larger network architecture: Critic [1024, 512, 256], Actor [512, 256, 128]
    - Supports mixed exploration noise per environment

    Observation normalization is handled by BaseActorCritic via
    actor_obs_normalizer and critic_obs_normalizer attributes.
    """

    critic1: DistributionalQNetwork
    critic2: DistributionalQNetwork

    # Distributional RL parameters
    num_atoms: int
    v_min: float
    v_max: float

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        is_squashed: bool = True,
        actor_class_name: str = "MLPActor",
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
            num_atoms: Number of atoms for C51
            v_min: Minimum value support
            v_max: Maximum value support
            actor_class_name: Name of actor class
            kinematic_tree: Optional kinematic tree
            obs_normalization: Whether to enable observation normalization
            key: JAX random key
            **kwargs: Must contain "actor_kwargs" and "critic_kwargs"
        """
        self.actor_obs_dim = num_actor_obs
        self.critic_obs_dim = num_critic_obs
        self.num_actions = num_actions
        self.is_squashed = is_squashed
        self.is_recurrent = False

        # Distributional parameters
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

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

        # Initialize observation normalizers (stored in BaseActorCritic)
        if obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(shape=num_actor_obs)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=num_critic_obs)
        else:
            self.actor_obs_normalizer = None
            self.critic_obs_normalizer = None

        print("🎭 FastTD3 Actor-Critic: deterministic policy + distributional critics")
        print(f"🤖 Actor: {actor_class_name}")
        print(f"📊 C51: {num_atoms} atoms, support [{v_min}, {v_max}]")
        print(f"📏 Obs normalization: {obs_normalization}")

    def _build_networks(
        self,
        actor_class_name: str,
        kinematic_tree: "KinematicTree | None",
        key_actor: jax.Array,
        key_critic1: jax.Array,
        key_critic2: jax.Array,
        **kwargs,
    ):
        """Build deterministic actor and distributional twin critics."""
        actor_kwargs = kwargs["actor_kwargs"]
        critic_kwargs = kwargs["critic_kwargs"]

        # Build actor (deterministic)
        self.actor = self._build_actor_common(
            actor_class_name=actor_class_name,
            actor_obs_dim=self.actor_obs_dim,
            num_actions=self.num_actions,
            actor_kwargs=actor_kwargs,
            kinematic_tree=kinematic_tree,
            key=key_actor,
        )

        # Build distributional twin critics
        critic_hidden = critic_kwargs.get("hidden_dims", [1024, 512, 256])
        critic_activation = critic_kwargs.get("activation", "relu")
        ortho_init = critic_kwargs.get("ortho_init", True)
        output_gain = critic_kwargs.get("output_gain", 1.0)

        self.critic1 = DistributionalQNetwork(
            obs_dim=self.critic_obs_dim,
            action_dim=self.num_actions,
            hidden_dims=critic_hidden,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            activation=critic_activation,
            ortho_init=ortho_init,
            output_gain=output_gain,
            key=key_critic1,
        )

        self.critic2 = DistributionalQNetwork(
            obs_dim=self.critic_obs_dim,
            action_dim=self.num_actions,
            hidden_dims=critic_hidden,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
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
            key: JAX random key
            deterministic: Ignored (FastTD3 is always deterministic at policy level)

        Returns:
            Tuple of (actions, aux_dict)
        """
        normalized_obs = self._normalize_actor_obs(observations)

        if normalized_obs.ndim == 2:
            keys = jax.random.split(key, normalized_obs.shape[0])
            actions, aux = jax.vmap(self.actor)(normalized_obs, key=keys)
        else:
            actions, aux = self.actor(normalized_obs, key=key)

        if self.is_squashed:
            actions = jnp.tanh(actions)

        return actions, aux

    def critic1_forward(
        self,
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """
        Forward pass for first critic returning logits.

        Returns:
            Logits [batch, num_atoms]
        """
        normalized_obs = self._normalize_critic_obs(observations)
        return self.critic1(normalized_obs, actions)

    def critic2_forward(
        self,
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """
        Forward pass for second critic returning logits.

        Returns:
            Logits [batch, num_atoms]
        """
        normalized_obs = self._normalize_critic_obs(observations)
        return self.critic2(normalized_obs, actions)

    def critic1_q_value(
        self,
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """Get Q-value from critic1."""
        normalized_obs = self._normalize_critic_obs(observations)
        return self.critic1.get_q_value(normalized_obs, actions)

    def critic2_q_value(
        self,
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        """Get Q-value from critic2."""
        normalized_obs = self._normalize_critic_obs(observations)
        return self.critic2.get_q_value(normalized_obs, actions)

    def evaluate(
        self,
        actor_observations: jax.Array,
        critic_observations: jax.Array,
        *,
        key: jax.Array,
    ) -> jax.Array:
        """
        Evaluate state value using minimum of two critics.

        Returns:
            Minimum Q-value
        """
        actions, _ = self.act(actor_observations, key=key)
        q1 = self.critic1_q_value(critic_observations, actions)
        q2 = self.critic2_q_value(critic_observations, actions)
        return jnp.minimum(q1, q2)

    def act_inference(self, actor_obs: jax.Array, *, key: jax.Array) -> Tuple[jax.Array, dict]:
        """Get deterministic action for inference (no exploration noise)."""
        normalized_obs = self._normalize_actor_obs(actor_obs)

        if normalized_obs.ndim == 2:
            keys = jax.random.split(key, normalized_obs.shape[0])
            actions, aux = jax.vmap(self.actor)(normalized_obs, key=keys)
        else:
            actions, aux = self.actor(normalized_obs, key=key)

        if self.is_squashed:
            actions = jnp.tanh(actions)

        return actions, aux

    def evaluate_value(self, critic_obs: jax.Array) -> Tuple[jax.Array, dict]:
        """Placeholder for compatibility."""
        normalized_obs = self._normalize_critic_obs(critic_obs)
        if normalized_obs.ndim == 1:
            return jnp.zeros((1,)), {}
        else:
            return jnp.zeros((normalized_obs.shape[0], 1)), {}

    @property
    def support(self) -> jax.Array:
        """Value support atoms."""
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    @property
    def delta_z(self) -> float:
        """Atom spacing."""
        return (self.v_max - self.v_min) / (self.num_atoms - 1)

    @property
    def extra_to_log(self) -> dict:
        """Extra metrics to log."""
        extra = {}
        if hasattr(self.actor, "encoder") and hasattr(self.actor.encoder, "last_gate"):
            gate = self.actor.encoder.last_gate
            if gate is not None:
                gate_mean = gate.mean(axis=0)
                extra["gate/mean"] = float(gate.mean())
                extra["gate/std"] = float(gate.std())
                for i in range(gate_mean.shape[0]):
                    extra[f"gate/per_action/action_{i:02d}"] = float(gate_mean[i])
        return extra
