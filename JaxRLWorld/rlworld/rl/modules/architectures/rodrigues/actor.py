from typing import TYPE_CHECKING

import torch
from torch import nn

from rlworld.rl.modules.architectures.rodrigues import RodriguesDecoder, RodriguesEncoder
from rlworld.rl.utils.model_manager import print_detailed_parameters

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree


class RodriguesActor(nn.Module):
    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        num_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [128, 128],
        joint_channels: int = 4,
        link_channels: int = 8,
        num_blocks: int = 12,
        embed_dim: int = 256,
        num_heads: int = 4,
        use_global_token: bool = True,
        global_token_dim: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.kinematic_tree = kinematic_tree
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.joint_channels = joint_channels
        self.link_channels = link_channels
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_global_token = use_global_token
        self.global_token_dim = global_token_dim
        self.encoder = RodriguesEncoder(
            kinematic_tree=kinematic_tree,
            obs_dim=num_obs,
            joint_channels=joint_channels,
            link_channels=link_channels,
            num_blocks=num_blocks,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_global_token=use_global_token,
            global_token_dim=global_token_dim,
        )

        print_detailed_parameters(self.encoder, "Rodrigues Encoder")

        joint_features, link_features, global_token = self.encoder(torch.zeros((1, self.num_obs)))
        link_feature_dim = link_features.flatten(1).shape[-1]
        self.decoder = RodriguesDecoder(
            kinematic_tree=kinematic_tree,
            joint_channels=joint_channels,
            link_feature_flatten_dim=link_feature_dim,
            action_dim=num_actions,
            hidden_dims=actor_hidden_dims,
        )
        print_detailed_parameters(self.decoder, "Rodrigues Decoder")

        for name, param in self.encoder.named_parameters():
            if "layer_norm" in name.lower() or "norm" in name.lower():
                print(f"{name}: {param.shape}")

        for idx in kinematic_tree.get_active_joint_indices():
            joint = kinematic_tree.joints[idx]
            print(f"joint {idx}: parent={joint['parent_link']}, child={joint['child_link']}")

    def post_update_step(self, *args, **kwargs):
        pass

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Observation → Action

        Args:
            observations: (batch, obs_dim) - Raw observations

        Returns:
            actions: (batch, action_dim) - Predicted actions
        """
        # Encode observations
        joint_features, link_features, global_token = self.encoder(observations)
        # print("Encoder done")
        # import ipdb; ipdb.set_trace()

        # Decode to actions
        actions = self.decoder(joint_features, link_features, global_token)
        # print("Decoder done")
        # import ipdb; ipdb.set_trace()
        return actions
