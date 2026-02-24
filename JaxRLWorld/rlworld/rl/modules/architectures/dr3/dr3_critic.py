from typing import Sequence, Union

import equinox as eqx
import jax
import math

from rlworld.rl.modules.utils import get_activation, orthogonal_init_mlp

__all__ = ["DR3MLP", "DR3Critic"]


class DR3MLP(eqx.Module):
    """
    MLP that can return penultimate layer features.

    Same architecture as MLP, but with feature extraction capability.
    """
    linears: tuple
    layer_norms: tuple
    activation: callable = eqx.field(static=True)
    output_activation: Union[callable, None] = eqx.field(static=True)
    use_layer_norm: bool = eqx.field(static=True)
    num_hidden: int = eqx.field(static=True)
    feature_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "relu",
        output_activation: str | None = None,
        use_layer_norm: bool = False,
        *,
        key: jax.Array,
    ):
        self.activation = get_activation(activation)
        self.output_activation = get_activation(output_activation) if output_activation else None
        self.use_layer_norm = use_layer_norm
        self.num_hidden = len(hidden_dims)
        self.feature_dim = hidden_dims[-1] if hidden_dims else input_dim

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        keys = jax.random.split(key, len(dims) - 1)

        linears = []
        layer_norms = []

        for i, (in_d, out_d, k) in enumerate(zip(dims[:-1], dims[1:], keys)):
            linears.append(eqx.nn.Linear(in_d, out_d, key=k))

            is_hidden = i < len(hidden_dims)
            if use_layer_norm and is_hidden:
                layer_norms.append(eqx.nn.LayerNorm(out_d))
            else:
                layer_norms.append(None)

        self.linears = tuple(linears)
        self.layer_norms = tuple(layer_norms)

    def _forward_single(self, x: jax.Array) -> jax.Array:
        """Forward pass for a single sample."""
        for i, (linear, ln) in enumerate(zip(self.linears, self.layer_norms)):
            x = linear(x)

            is_hidden = i < self.num_hidden
            if is_hidden:
                if ln is not None:
                    x = ln(x)
                x = self.activation(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x

    def _forward_single_with_features(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass returning both output and penultimate features."""
        features = None

        for i, (linear, ln) in enumerate(zip(self.linears, self.layer_norms)):
            x = linear(x)

            is_hidden = i < self.num_hidden
            if is_hidden:
                if ln is not None:
                    x = ln(x)
                x = self.activation(x)

                # Save penultimate layer features (last hidden layer output)
                if i == self.num_hidden - 1:
                    features = x

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x, features

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim == 1:
            return self._forward_single(x)
        else:
            return jax.vmap(self._forward_single)(x)

    def forward_with_features(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Forward pass with feature extraction.

        Args:
            x: Input [input_dim] or [batch, input_dim]

        Returns:
            output: Network output [output_dim] or [batch, output_dim]
            features: Penultimate layer features [feature_dim] or [batch, feature_dim]
        """
        if x.ndim == 1:
            return self._forward_single_with_features(x)
        else:
            return jax.vmap(self._forward_single_with_features)(x)


class DR3Critic(eqx.Module):
    """
    Critic with DR3 feature extraction capability.

    Same as MLPCritic but can return penultimate layer features.
    """
    net: DR3MLP
    num_obs: int = eqx.field(static=True)
    feature_dim: int = eqx.field(static=True)

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
        self.num_obs = num_obs
        self.feature_dim = hidden_dims[-1] if hidden_dims else num_obs

        key1, key2 = jax.random.split(key)

        self.net = DR3MLP(
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
                output_gain=1.0,
                key=key2,
            )

    def __call__(self, observations: jax.Array) -> tuple[jax.Array, dict]:
        """Standard forward pass."""
        return self.net(observations), {}

    def forward_with_features(
        self, observations: jax.Array
    ) -> tuple[jax.Array, jax.Array, dict]:
        """
        Forward pass with feature extraction.

        Args:
            observations: [batch_size, num_obs]

        Returns:
            values: [batch_size, 1]
            features: [batch_size, feature_dim]
            aux: Empty dict for compatibility
        """
        values, features = self.net.forward_with_features(observations)
        return values, features, {}
