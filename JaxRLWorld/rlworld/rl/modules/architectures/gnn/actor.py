from typing import TYPE_CHECKING

import equinox as eqx
import jax

from rlworld.rl.modules.architectures.base import BaseActor
from rlworld.rl.modules.architectures.gnn.encoder import GNNEncoder
from rlworld.rl.modules.architectures.morphology_utils import ParentLinkToJointActionDecoder

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = ["GNNActor"]


class GNNActor(BaseActor):
    """
    GNN-based actor.
    Processes unbatched input. Use jax.vmap for batched input.
    """
    encoder: GNNEncoder
    decoder: ParentLinkToJointActionDecoder

    num_obs: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        num_obs: int,
        num_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        action_hidden_dim: int | None = None,
        actuated_joint_names: "list[str] | None" = None,
        *,
        key: jax.Array,
        **kwargs
    ):
        self.num_obs = num_obs
        self.num_actions = num_actions

        key_enc, key_dec = jax.random.split(key)

        self.encoder = GNNEncoder(
            kinematic_tree=kinematic_tree,
            obs_dim=num_obs,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            key=key_enc,
            **kwargs
        )

        self.decoder = ParentLinkToJointActionDecoder(
            kinematic_tree=kinematic_tree,
            hidden_dim=hidden_dim,
            activation=activation,
            action_hidden_dim=action_hidden_dim,
            actuated_joint_names=actuated_joint_names,
            key=key_dec,
        )

    def __call__(self, observation: jax.Array, *, key: jax.Array | None = None) -> tuple[jax.Array, dict]:
        """
        Args:
            observation: (num_obs,) unbatched

        Returns:
            actions: (num_actions,)
            aux: {"gnn_feature_mean": ..., "gnn_feature_std": ...}
        """
        features = self.encoder(observation, key=key)  # (num_bodies, hidden_dim)
        actions = self.decoder(features)  # (num_actions,)

        aux = {
            "gnn_feature_mean": features.mean(),
            "gnn_feature_std": features.std(),
        }

        return actions, aux
