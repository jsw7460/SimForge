from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.architectures.base import BaseActor
from rlworld.rl.modules.architectures.morphology_utils import ParentLinkToJointActionDecoder
from .encoder import create_encoder, ABAEncoder

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


class ABAActor(BaseActor):
    """
    ABA-based actor.
    Processes unbatched input. Use jax.vmap for batched input.
    """
    encoder: ABAEncoder | eqx.Module
    decoder: ParentLinkToJointActionDecoder

    num_obs: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    has_auxiliary_loss: bool = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        num_obs: int,
        num_actions: int,
        encoder_type: str = "ABAEncoder",
        # Encoder params
        link_channels: int = 8,
        spatial_dim: int = 6,
        learnable_contribution_weight: bool = False,
        use_global_layer_norm: bool = False,
        use_positive_constraint: bool = True,
        use_auxiliary_loss: bool = True,
        # Decoder params
        activation: str = "elu",
        action_hidden_dim: int | None = None,
        actuated_joint_names: "list[str] | None" = None,
        *,
        key: jax.Array,
        **kwargs,
    ):
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.has_auxiliary_loss = use_auxiliary_loss

        key_enc, key_dec = jax.random.split(key)

        # Create encoder via factory
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            kinematic_tree=kinematic_tree,
            obs_dim=num_obs,
            link_channels=link_channels,
            spatial_dim=spatial_dim,
            learnable_contribution_weight=learnable_contribution_weight,
            use_global_layer_norm=use_global_layer_norm,
            use_positive_constraint=use_positive_constraint,
            key=key_enc,
            **kwargs,
        )

        # Create decoder
        hidden_dim = self.encoder.output_dim[1]
        self.decoder = ParentLinkToJointActionDecoder(
            kinematic_tree=kinematic_tree,
            hidden_dim=hidden_dim,
            activation=activation,
            action_hidden_dim=action_hidden_dim,
            actuated_joint_names=actuated_joint_names,
            key=key_dec,
        )

    def __call__(
        self,
        observation: jax.Array,
        *,
        key: jax.Array | None = None
    ) -> tuple[jax.Array, dict]:
        """
        Args:
            observation: (num_obs,) unbatched

        Returns:
            actions: (num_actions,)
            aux: auxiliary info dict
        """
        features = self.encoder(observation, key=key)  # (num_bodies, hidden_dim)
        actions = self.decoder(features)  # (num_actions,)

        aux = {
            "aba_feature_mean": features.mean(),
            "aba_feature_std": features.std(),
        }

        return actions, aux

    def compute_auxiliary_loss(self, observation: jax.Array) -> jax.Array:
        """
        Compute auxiliary loss (orthogonality regularization).

        Args:
            observation: (num_obs,) unbatched

        Returns:
            loss: scalar
        """
        if not self.has_auxiliary_loss:
            return jnp.array(0.0)
        return self.encoder.compute_auxiliary_loss(observation)