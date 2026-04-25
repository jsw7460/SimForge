"""Per-joint action decoder.

The decoder maps per-body link features to per-joint mean actions by
looking up each active joint's parent body and running an
:class:`ActionHead` MLP over that body's feature vector.

The N action heads are stored as a *single* :class:`ActionHead` whose
leaves carry an extra leading dim of size ``num_actions`` — this is a
filter-vmap construction that compiles the forward pass to one batched
matmul instead of unrolling ``num_actions`` separate Python iterations
at trace time. The previous tuple-of-heads layout produced an XLA HLO
graph proportional to ``num_actions`` and made compilation slow + GPU
launch overhead heavy on every step. Equivalent params, much faster.
"""
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.utils import get_activation

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = ["ParentLinkToJointActionDecoder"]


def _orthogonal_init_linear(
    linear: eqx.nn.Linear,
    gain: float,
    key: jax.Array,
) -> eqx.nn.Linear:
    """Replace a Linear's weight with QR-orthogonal init scaled by ``gain``."""
    weight = linear.weight
    max_dim = max(weight.shape)
    q, _ = jnp.linalg.qr(jax.random.normal(key, shape=(max_dim, max_dim)))
    new_weight = gain * q[: weight.shape[0], : weight.shape[1]]
    new_bias = jnp.zeros_like(linear.bias)
    linear = eqx.tree_at(lambda l: l.weight, linear, new_weight)
    linear = eqx.tree_at(lambda l: l.bias, linear, new_bias)
    return linear


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
    """Stacked-head decoder. Processes unbatched input.

    Action ordering follows ``actuated_joint_names`` (the canonical order
    produced by :class:`ActionManagerBase`), NOT the kinematic tree's
    MJCF parse order. The two are not the same — for example T1's tree
    visits its leaves depth-first while the act manager interleaves
    left/right joints — so an action produced at index ``k`` must be
    routed to ``actuated_joint_names[k]``'s parent body to learn
    correctly. ``actuated_joint_names=None`` falls back to tree order
    (for tests / non-actuated-decode scenarios) but emits no warning;
    runtime callers should always provide the list to avoid silent
    mis-routing.
    """

    stacked_heads: ActionHead
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
        actuated_joint_names: "list[str] | tuple[str, ...] | None" = None,
        *,
        key: jax.Array,
    ):
        self.hidden_dim = hidden_dim

        if action_hidden_dim is None:
            action_hidden_dim = max(hidden_dim // 2, 32)
        self.action_hidden_dim = action_hidden_dim

        active_joints = kinematic_tree.get_active_joint_indices()

        if actuated_joint_names is None:
            # Fallback: tree order. Only correct when the env's action
            # manager happens to use the same order — verify before using.
            parent_indices = [
                kinematic_tree.joints[j]["parent_link"] for j in active_joints
            ]
        else:
            # Canonical actuator order: re-map so action[k] drives
            # actuated_joint_names[k]'s parent body.
            tree_idx_by_name = {
                kinematic_tree.joints[j]["name"]: j for j in active_joints
            }
            missing = [n for n in actuated_joint_names if n not in tree_idx_by_name]
            if missing:
                raise ValueError(
                    "actuated_joint_names contains joints not in the kinematic "
                    f"tree's active set: {missing}. Tree active joints: "
                    f"{sorted(tree_idx_by_name)}."
                )
            parent_indices = [
                kinematic_tree.joints[tree_idx_by_name[n]]["parent_link"]
                for n in actuated_joint_names
            ]

        self.num_actions = len(parent_indices)
        self.parent_indices = jnp.array(parent_indices, dtype=jnp.int32)

        gain = float(jnp.sqrt(2.0))

        @eqx.filter_vmap
        def make_head(k: jax.Array) -> ActionHead:
            k0, k1, k2 = jax.random.split(k, 3)
            head = ActionHead(
                hidden_dim=hidden_dim,
                action_hidden_dim=action_hidden_dim,
                activation=activation,
                key=k0,
            )
            if ortho_init:
                new_l1 = _orthogonal_init_linear(head.linear1, gain, k1)
                new_l2 = _orthogonal_init_linear(head.linear2, output_gain, k2)
                head = eqx.tree_at(lambda h: h.linear1, head, new_l1)
                head = eqx.tree_at(lambda h: h.linear2, head, new_l2)
            return head

        self.stacked_heads = make_head(jax.random.split(key, self.num_actions))

    def __call__(self, link_features: jax.Array) -> jax.Array:
        """
        Args:
            link_features: (num_bodies, hidden_dim) - unbatched.

        Returns:
            actions: (num_actions,)
        """
        parent_features = link_features[self.parent_indices]
        # Apply each stacked head to its joint's parent feature in one batched op.
        actions = jax.vmap(lambda head, x: head(x))(
            self.stacked_heads, parent_features,
        )
        return actions.squeeze(-1)
