from typing import TYPE_CHECKING

import torch
from torch import nn

from rlworld.rl.modules.utils import get_activation

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


class MLPDecoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, body_features):
        actions = self.net(body_features).squeeze(-1)  # (B, num_joints)
        return actions


class GlobalAggregationDecoder(nn.Module):
    """
    Aggregate joint features and decode to actions.

    Uses attention-based pooling to aggregate joint-level features
    into global representation, then decode to actions.

    Args:
        num_joints: Number of joints in kinematic tree
        feature_dim: Hidden dimension per joint
        num_actions: Number of output actions
        aggregation_type: Type of aggregation ('attention', 'mean', 'max')
    """

    def __init__(
        self,
        num_joints: int,
        feature_dim: int,
        num_actions: int,
        activation: str,
        aggregation_type: str = "attention",
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.aggregation_type = aggregation_type

        if aggregation_type == "attention":
            # Attention-based pooling
            self.attention = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

        # MLP decoder: aggregated features → actions
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Small initialization for last layer
        # last_layer = self.mlp[-1]
        # nn.init.orthogonal_(last_layer.weight, gain=0.01)

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate joint features and decode to actions.

        Args:
            joint_features: (batch, num_joints, feature_dim)

        Returns:
            actions: (batch, num_actions)
        """
        batch_size = joint_features.shape[0]

        # Aggregate joint features
        if self.aggregation_type == "attention":
            # Attention pooling: (batch, num_joints, feature_dim) → (batch, feature_dim)
            attn_scores = self.attention(joint_features)  # (batch, num_joints, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, num_joints, 1)
            aggregated = (joint_features * attn_weights).sum(dim=1)  # (batch, feature_dim)

        elif self.aggregation_type == "mean":
            # Mean pooling
            aggregated = joint_features.mean(dim=1)  # (batch, feature_dim)

        elif self.aggregation_type == "max":
            # Max pooling
            aggregated = joint_features.max(dim=1)[0]  # (batch, feature_dim)

        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")

        # Decode to actions
        actions = self.mlp(aggregated)  # (batch, num_actions)

        return actions


class KinematicTreeActionDecoder(nn.Module):
    """
    Action decoder for kinematic tree-based encoders.

    Converts per-body features (num_bodies) to per-joint actions (num_actions)
    by combining parent and child link features for each actuated joint.

    Uses separate (but lightweight) action heads per joint, following Body Transformer.

    This naturally handles the case where num_bodies > num_actions due to:
    - Root body (no parent joint)
    - Leaf bodies (may have no child joint)
    - Unactuated links

    Args:
        kinematic_tree: Robot kinematic structure
        hidden_dim: Input feature dimension per body
        action_hidden_dim: Hidden dimension for per-joint action head (default: hidden_dim // 2)
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        hidden_dim: int,
        activation: str,
        action_hidden_dim: int = None,
        ortho_init: bool = True,
    ):
        super().__init__()
        self.tree = kinematic_tree
        self.hidden_dim = hidden_dim

        # Default: use smaller hidden dimension to keep parameters low
        if action_hidden_dim is None:
            action_hidden_dim = max(hidden_dim // 2, 32)
        self.action_hidden_dim = action_hidden_dim

        # Get actuated joints
        active_joints = kinematic_tree.get_active_joint_indices()
        # active_joints.remove(8)
        self.num_actions = len(active_joints)

        # Pre-compute parent-child link indices for each joint
        parent_indices = []
        child_indices = []

        for joint_idx in active_joints:
            joint_info = kinematic_tree.joints[joint_idx]
            parent_indices.append(joint_info["parent_link"])
            child_indices.append(joint_info["child_link"])

        # Register as buffers for device movement
        self.register_buffer("parent_indices", torch.tensor(parent_indices, dtype=torch.long))
        self.register_buffer("child_indices", torch.tensor(child_indices, dtype=torch.long))

        # Separate action head for each joint (Body Transformer style)
        # Input: 2 * hidden_dim (concatenated parent + child)
        # Output: 1 (single action value)
        self.action_heads = nn.ModuleList([self._make_action_head(activation) for _ in range(self.num_actions)])

        if ortho_init:
            self._init_weights()

    def _make_action_head(self, activation: str) -> nn.Module:
        """Create a single action head for one joint"""
        layers = [
            nn.Linear(2 * self.hidden_dim, self.action_hidden_dim),
            get_activation(activation),
            nn.Linear(self.action_hidden_dim, 1),
        ]
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Orthogonal initialization for action heads"""
        for action_head in self.action_heads:
            modules = [m for m in action_head if isinstance(m, nn.Linear)]

            # Hidden layers
            for module in modules[:-1]:
                nn.init.orthogonal_(module.weight, gain=torch.sqrt(torch.tensor(2)).item())
                nn.init.zeros_(module.bias)

            # Output layer (small gain)
            # nn.init.orthogonal_(modules[-1].weight, gain=0.01)
            # nn.init.zeros_(modules[-1].bias)

    def forward(self, link_features: torch.Tensor) -> torch.Tensor:
        """
        Decode per-body features to per-joint actions.

        Args:
            link_features: (batch, num_bodies, hidden_dim)

        Returns:
            actions: (batch, num_actions)
        """
        batch = link_features.shape[0]
        device = link_features.device

        # Move indices to device
        parent_idx = self.parent_indices.to(device)
        child_idx = self.child_indices.to(device)

        # Gather parent and child features for all joints
        # (batch, num_actions, hidden_dim)
        parent_features = link_features[:, parent_idx, :]
        child_features = link_features[:, child_idx, :]

        # Concatenate parent + child
        # (batch, num_actions, 2 * hidden_dim)
        combined_features = torch.cat([parent_features, child_features], dim=-1)

        # Generate actions using separate heads for each joint
        actions = []
        for joint_idx in range(self.num_actions):
            joint_feature = combined_features[:, joint_idx, :]  # (batch, 2*hidden_dim)
            joint_action = self.action_heads[joint_idx](joint_feature)  # (batch, 1)
            actions.append(joint_action)

        # Stack to (batch, num_actions)
        actions = torch.cat(actions, dim=-1)

        return actions


class ParentLinkToJointActionDecoder(nn.Module):
    """
    Action decoder for kinematic tree-based encoders.

    Converts per-body features (num_bodies) to per-joint actions (num_actions)
    by combining parent and child link features for each actuated joint.

    Uses separate (but lightweight) action heads per joint, following Body Transformer.

    This naturally handles the case where num_bodies > num_actions due to:
    - Root body (no parent joint)
    - Leaf bodies (may have no child joint)
    - Unactuated links

    Args:
        kinematic_tree: Robot kinematic structure
        hidden_dim: Input feature dimension per body
        action_hidden_dim: Hidden dimension for per-joint action head (default: hidden_dim // 2)
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        hidden_dim: int,
        activation: str,
        device: torch.device,
        action_hidden_dim: int = None,
        ortho_init: bool = True,
    ):
        super().__init__()
        self.tree = kinematic_tree
        self.hidden_dim = hidden_dim

        # Default: use smaller hidden dimension to keep parameters low
        if action_hidden_dim is None:
            action_hidden_dim = max(hidden_dim // 2, 32)
        self.action_hidden_dim = action_hidden_dim

        # Get actuated joints
        active_joints = kinematic_tree.get_active_joint_indices()
        # active_joints.remove(8)
        self.num_actions = len(active_joints)

        # Pre-compute parent-child link indices for each joint
        parent_indices = []
        child_indices = []

        for joint_idx in active_joints:
            joint_info = kinematic_tree.joints[joint_idx]
            parent_indices.append(joint_info["parent_link"])
            child_indices.append(joint_info["child_link"])

        # Register as buffers for device movement
        self.register_buffer("parent_indices", torch.tensor(parent_indices, dtype=torch.long, device=device))

        # Separate action head for each joint (Body Transformer style)
        # Input: 2 * hidden_dim (concatenated parent + child)
        # Output: 1 (single action value)
        self.action_heads = nn.ModuleList([self._make_action_head(activation) for _ in range(self.num_actions)])
        if ortho_init:
            self._init_weights()

    def _make_action_head(self, activation: str) -> nn.Module:
        """Create a single action head for one joint"""
        layers = [
            nn.Linear(self.hidden_dim, self.action_hidden_dim),
            get_activation(activation),
            nn.Linear(self.action_hidden_dim, 1),
        ]
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Orthogonal initialization for action heads"""
        for action_head in self.action_heads:
            modules = [m for m in action_head if isinstance(m, nn.Linear)]

            # Hidden layers (ReLU)
            for module in modules[:-1]:
                nn.init.orthogonal_(module.weight, gain=torch.sqrt(torch.tensor(2)).item())
                nn.init.zeros_(module.bias)

            # nn.init.orthogonal_(modules[-1].weight, gain=0.01)
            # nn.init.constant_(modules[-1].bias, 0.0)

    def forward(self, link_features: torch.Tensor) -> torch.Tensor:
        """
        Decode per-body features to per-joint actions.

        Args:
            link_features: (batch, num_bodies, hidden_dim)

        Returns:
            actions: (batch, num_actions)
        """

        # Move indices to device
        parent_idx = self.parent_indices

        # Gather parent and child features for all joints
        # (batch, num_actions, hidden_dim)
        parent_features = link_features[:, parent_idx, :]

        # Generate actions using separate heads for each joint
        actions = []
        for joint_idx in range(self.num_actions):
            joint_feature = parent_features[:, joint_idx, :]  # (batch, 2*hidden_dim)
            joint_action = self.action_heads[joint_idx](joint_feature)  # (batch, 1)
            actions.append(joint_action)

        # Stack to (batch, num_actions)
        actions = torch.cat(actions, dim=-1)
        return actions
