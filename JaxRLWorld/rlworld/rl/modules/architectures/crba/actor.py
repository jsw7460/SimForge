from typing import TYPE_CHECKING

import torch
from torch import nn

from rlworld.rl.modules.architectures.crba import encoder as crba_encoder

from .decoder import ActiveJointDecoder, SimpleJointDecoder

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree


class CRBAActor(nn.Module):
    spatial_dim: int = 6

    encoder: crba_encoder.CRBAAttentionBiasedEncoder
    has_auxiliary_loss = True

    def __init__(self, kinematic_tree: "KinematicTree", num_obs: int, num_actions: int, encoder_type: str, **kwargs):
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_actions
        encoder_class = getattr(crba_encoder, encoder_type)
        self.encoder = encoder_class(kinematic_tree=kinematic_tree, obs_dim=num_obs, act_dim=num_actions, **kwargs)

        self.use_auxiliary_loss = kwargs.get("use_auxiliary_loss", False)

        num_joints, hidden_dim = self.encoder.output_dim

        if num_joints == num_actions:
            self.decoder = SimpleJointDecoder(num_joints=num_actions, latent_dim=hidden_dim)

        else:
            self.decoder = ActiveJointDecoder(
                active_joint_indices=kinematic_tree.get_active_joint_indices(),
                latent_dim=hidden_dim,
            )

    def compute_auxiliary_loss(self, *args, **kwargs) -> torch.Tensor:
        return self.encoder.compute_auxiliary_loss(*args, **kwargs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Observation → Action

        Args:
            observations: (batch, obs_dim) - Raw observations

        Returns:
            actions: (batch, action_dim) - Predicted actions
        """
        # Encode observations
        features = self.encoder(observations)
        # Decode to actions
        actions = self.decoder(features)
        return actions

    def post_update_step(self, *args, **kwargs):
        pass

    @property
    def extra_to_log(self) -> dict:
        extra = {}
        if hasattr(self.encoder, "extra_to_log"):
            extra.update(**self.encoder.extra_to_log)
        return extra
