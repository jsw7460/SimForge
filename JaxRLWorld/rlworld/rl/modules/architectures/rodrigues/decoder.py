import torch
from torch import nn

from rlworld.rl.modules.utils import orthogonal_init, scale_last_layer


class RodriguesDecoder(nn.Module):
    """
    Simple MLP Decoder: Converts encoded features to actions.

    Flattens all joint and link features, passes through MLP to predict actions.

    Args:
        kinematic_tree: KinematicTree defining robot structure
        joint_channels: Number of joint feature channels (C_J)
        link_channels: Number of link feature channels (C_L)
        action_dim: Output action dimension
        hidden_dims: List of hidden layer dimensions
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        joint_channels: int,
        link_feature_flatten_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [512, 256],
        **kwargs,
    ):
        super().__init__()
        self.tree = kinematic_tree

        # Calculate input dimension
        num_active_joints = len(kinematic_tree.get_active_joint_indices())

        input_dim = (
            num_active_joints * joint_channels  # Joint features
            + link_feature_flatten_dim
        )

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

        # orthogonal_init(self.mlp)
        # scale_last_layer(self.mlp, scale=0.01)

    def _init_weights(self):
        """Initialize MLP weights"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, joint_features: torch.Tensor, link_features: torch.Tensor, global_token: torch.Tensor = None):
        """
        Decode features to actions.

        Args:
            joint_features: (batch, max_joint_idx+1, C_J) - Encoded joint features
            link_features: (batch, num_bodies, C_L, d, d) - Encoded link features
            global_token: (batch, token_dim) - Optional

        Returns:
            actions: (batch, action_dim) - Predicted actions
        """
        # Gather active joint features
        active_joints = self.tree.get_active_joint_indices()
        active_joint_features = joint_features[:, active_joints]  # (batch, num_active, C_J)

        # Flatten all features
        joint_flat = active_joint_features.flatten(start_dim=1)  # (batch, num_active * C_J)
        link_flat = link_features.flatten(start_dim=1)  # (batch, num_bodies * C_L * 16)

        # Concatenate
        features = torch.cat([joint_flat, link_flat], dim=-1)

        # Pass through MLP
        actions = self.mlp(features)
        return actions


class FlattenDecoder(nn.Module):
    def __init__(self, feature_dim, num_joints=None, num_actions=None, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.projection = None
        if num_joints is not None and num_actions is not None:
            self.projection = nn.Sequential(nn.ReLU(), nn.Linear(num_joints, num_actions))

            orthogonal_init(self.projection)
            scale_last_layer(self.projection, scale=0.01)

        orthogonal_init(self.net)
        scale_last_layer(self.net, scale=0.01)

    def forward(self, body_features):
        """
        body_features: (batch, num_bodies, feature_dim)
        """
        # b = body_features.shape[0]
        # n = body_features.shape[1]
        # actuated = body_features.reshape(b, n, -1)
        actuated = body_features

        actions = self.net(actuated).squeeze(-1)  # (B, num_joints)

        if self.projection is not None:
            actions = self.projection(actions)

        return actions
