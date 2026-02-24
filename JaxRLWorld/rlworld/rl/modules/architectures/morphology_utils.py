from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.utils import get_activation

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = ["ParentLinkToJointActionDecoder"]


class ActionHead(eqx.Module):
    """Single action head for one joint."""
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int,
        action_hidden_dim: int,
        activation: str,
        *,
        key: jax.Array,
    ):
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(hidden_dim, action_hidden_dim, key=key1)
        self.linear2 = eqx.nn.Linear(action_hidden_dim, 1, key=key2)
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (hidden_dim,) -> (1,)"""
        x = self.linear1(x)
        x = get_activation(self.activation)(x)
        x = self.linear2(x)
        return x


class ParentLinkToJointActionDecoder(eqx.Module):
    """
    Action decoder. Processes unbatched input.
    """
    action_heads: tuple
    parent_indices: jax.Array = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    action_hidden_dim: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        hidden_dim: int,
        activation: str,
        action_hidden_dim: int | None = None,
        ortho_init: bool = True,
        output_gain: float = 0.1,
        *,
        key: jax.Array,
    ):
        self.hidden_dim = hidden_dim

        if action_hidden_dim is None:
            action_hidden_dim = max(hidden_dim // 2, 32)
        self.action_hidden_dim = action_hidden_dim

        active_joints = kinematic_tree.get_active_joint_indices()
        self.num_actions = len(active_joints)

        parent_indices = []
        for joint_idx in active_joints:
            joint_info = kinematic_tree.joints[joint_idx]
            parent_indices.append(joint_info['parent_link'])
        self.parent_indices = jnp.array(parent_indices, dtype=jnp.int32)

        key, init_key = jax.random.split(key)
        head_keys = jax.random.split(key, self.num_actions)
        action_heads = [
            ActionHead(
                hidden_dim=hidden_dim,
                action_hidden_dim=action_hidden_dim,
                activation=activation,
                key=head_keys[i],
            )
            for i in range(self.num_actions)
        ]

        if ortho_init:
            action_heads = self._init_weights(action_heads, init_key, output_gain)

        self.action_heads = tuple(action_heads)

    def _init_weights(
        self,
        action_heads: list,
        key: jax.Array,
        output_gain: float,
    ) -> list:
        keys = jax.random.split(key, len(action_heads))
        new_heads = []

        for i, head in enumerate(action_heads):
            key1, key2 = jax.random.split(keys[i])

            new_linear1 = self._orthogonal_init_linear(head.linear1, gain=jnp.sqrt(2.0), key=key1)
            new_linear2 = self._orthogonal_init_linear(head.linear2, gain=output_gain, key=key2)

            head = eqx.tree_at(lambda h: h.linear1, head, new_linear1)
            head = eqx.tree_at(lambda h: h.linear2, head, new_linear2)
            new_heads.append(head)

        return new_heads

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

    def __call__(self, link_features: jax.Array) -> jax.Array:
        """
        Args:
            link_features: (num_bodies, hidden_dim) - unbatched

        Returns:
            actions: (num_actions,)
        """
        parent_features = link_features[self.parent_indices]

        actions = []
        for joint_idx, head in enumerate(self.action_heads):
            joint_feature = parent_features[joint_idx]
            action = head(joint_feature)
            actions.append(action)

        return jnp.concatenate(actions, axis=0)