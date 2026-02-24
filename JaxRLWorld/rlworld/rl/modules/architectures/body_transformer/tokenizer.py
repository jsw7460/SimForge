from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import equinox as eqx

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = ["BodyTokenizer"]


class BodyTokenizer(eqx.Module):
    """
    Per-body tokenizer for Body Transformer.
    Processes unbatched input. Use jax.vmap for batched input.
    """
    projections: list
    num_bodies: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    hidden_dim: int | None = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        embed_dim: int,
        hidden_dim: int | None = None,
        *,
        key: jax.Array,
    ):
        self.num_bodies = kinematic_tree.num_bodies
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        key, init_key = jax.random.split(key)
        keys = jax.random.split(key, self.num_bodies)

        if hidden_dim is None:
            projections = [
                eqx.nn.Linear(obs_dim, embed_dim, key=keys[i])
                for i in range(self.num_bodies)
            ]
        else:
            projections = [
                self._make_mlp_projection(obs_dim, hidden_dim, embed_dim, keys[i])
                for i in range(self.num_bodies)
            ]

        self.projections = tuple(self._init_weights(projections, init_key))

    @staticmethod
    def _make_mlp_projection(
        obs_dim: int,
        hidden_dim: int,
        embed_dim: int,
        key: jax.Array,
    ) -> tuple:
        key1, key2 = jax.random.split(key)
        return (  # list → tuple
            eqx.nn.Linear(obs_dim, hidden_dim, key=key1),
            eqx.nn.Linear(hidden_dim, embed_dim, key=key2),
        )

    def _init_weights(self, projections, key: jax.Array) -> tuple:
        new_projections = []
        keys = jax.random.split(key, len(projections))

        for i, proj in enumerate(projections):
            if isinstance(proj, eqx.nn.Linear):
                new_projections.append(self._orthogonal_init_linear(proj, gain=1.0, key=keys[i]))
            else:
                layer_keys = jax.random.split(keys[i], len(proj))
                new_layers = tuple(
                    self._orthogonal_init_linear(layer, gain=1.0, key=layer_keys[j])
                    if isinstance(layer, eqx.nn.Linear) else layer
                    for j, layer in enumerate(proj)
                )
                new_projections.append(new_layers)

        return tuple(new_projections)

    @staticmethod
    def _orthogonal_init_linear(
        linear: eqx.nn.Linear,
        gain: float,
        key: jax.Array,
    ) -> eqx.nn.Linear:
        weight = linear.weight
        max_dim = max(weight.shape)
        q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(max_dim, max_dim)))
        new_weight = gain * q[:weight.shape[0], :weight.shape[1]]
        new_bias = jnp.zeros_like(linear.bias)

        linear = eqx.tree_at(lambda l: l.weight, linear, new_weight)
        linear = eqx.tree_at(lambda l: l.bias, linear, new_bias)
        return linear

    def __call__(self, observation: jax.Array) -> jax.Array:
        """
        Tokenize single observation.

        Args:
            observation: (obs_dim,) - unbatched

        Returns:
            tokens: (num_bodies, embed_dim)
        """
        tokens = []
        for proj in self.projections:
            if isinstance(proj, eqx.nn.Linear):
                tokens.append(proj(observation))
            else:
                x = proj[0](observation)
                x = jax.nn.relu(x)
                x = proj[1](x)
                tokens.append(x)

        return jnp.stack(tokens, axis=0)