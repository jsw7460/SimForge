import math
from typing import Sequence

import equinox as eqx
import jax

from rlworld.rl.modules.architectures.base import BaseActor, BaseCritic
from rlworld.rl.modules.utils import MLP, orthogonal_init_mlp


class MLPActor(BaseActor):
    """
    Multi-layer perceptron actor.

    Returns only the mean action. Standard deviation is managed
    separately by the ActorCritic class.

    Equivalent to PyTorch MLPActor.
    """

    net: MLP
    num_obs: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layer_norm: bool = False,
        ortho_init: bool = True,
        output_gain: float = 1.0,
        *,
        key: jax.Array,
    ):
        """
        Args:
            num_obs: Observation dimension
            num_actions: Action dimension
            hidden_dims: Hidden layer dimensions [256, 256]
            activation: Activation function name
            use_layer_norm: Whether to use layer normalization
            ortho_init: Whether to use orthogonal initialization
            output_gain: Gain for the MLP (ignored if ortho_init=False)
            key: JAX random key
        """
        self.num_obs = num_obs
        self.num_actions = num_actions
        key1, key2 = jax.random.split(key)

        # Build MLP
        self.net = MLP(
            input_dim=num_obs,
            hidden_dims=hidden_dims,
            output_dim=num_actions,
            activation=activation,
            output_activation=None,
            use_layer_norm=use_layer_norm,
            key=key1,
        )

        # Apply orthogonal initialization if requested
        if ortho_init:
            # Calculate gain based on activation
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
                output_gain=output_gain,  # Standard for actor output
                key=key2,
            )

    def __call__(self, observations: jax.Array, key: jax.Array = None) -> tuple[jax.Array, dict]:
        """
        Forward pass.

        Args:
            observations: [batch_size, num_obs] or [num_obs]

        Returns:
            mean_actions: [batch_size, num_actions] or [num_actions]
        """
        return self.net(observations), {}


class MLPCritic(BaseCritic):
    """
    MLP Critic (value function).

    Returns scalar value estimate.
    """

    net: MLP
    num_obs: int = eqx.field(static=True)

    def __init__(
        self,
        num_obs: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
        use_layer_norm: bool = False,
        ortho_init: bool = True,
        *,
        key: jax.Array,
    ):
        """
        Args:
            num_obs: Observation dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function name
            use_layer_norm: Whether to use layer normalization
            ortho_init: Whether to use orthogonal initialization
            key: JAX random key
        """
        self.num_obs = num_obs

        key1, key2 = jax.random.split(key)

        self.net = MLP(
            input_dim=num_obs,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            output_activation=None,
            use_layer_norm=use_layer_norm,
            key=key1,
        )

        if ortho_init:
            gain_map = {
                "relu": math.sqrt(2),
                "elu": math.sqrt(2),
                "tanh": 1.0,
            }
            hidden_gain = gain_map.get(activation, math.sqrt(2))

            self.net = orthogonal_init_mlp(
                self.net,
                hidden_gain=hidden_gain,
                output_gain=1.0,  # Critic output gain
                key=key2,
            )

    def __call__(self, observations: jax.Array) -> tuple[jax.Array, dict]:
        """
        Forward pass.

        Args:
            observations: [batch_size, num_obs]

        Returns:
            values: [batch_size, 1] or [batch_size] depending on squeeze
        """
        return self.net(observations), {}
