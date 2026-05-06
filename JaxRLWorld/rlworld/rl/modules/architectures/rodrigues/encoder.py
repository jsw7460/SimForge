import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree


class MultiChannelRodriguesOperator(nn.Module):
    """
    Multi-Channel Neural Rodrigues Operator
    """

    def __init__(
        self,
        joint_channels: int,  # C_J
        link_channels_in: int,  # C_L
        link_channels_out: int,  # C'_L
        spatial_dim: int,
        use_stable_init: bool = False,
    ):
        super().__init__()
        self.C_J = joint_channels
        self.C_L = link_channels_in
        self.C_L_out = link_channels_out
        self.spatial_dim = spatial_dim
        self.use_stable_init = use_stable_init

        # Learnable weights: (C_L, C'_L, C_J, 4, 4) - 5D tensor
        self.W_bias = nn.Parameter(torch.randn(link_channels_in, link_channels_out, spatial_dim, spatial_dim))
        self.W_cos = nn.Parameter(
            torch.randn(link_channels_in, link_channels_out, joint_channels, spatial_dim, spatial_dim)
        )
        self.W_sin = nn.Parameter(
            torch.randn(link_channels_in, link_channels_out, joint_channels, spatial_dim, spatial_dim)
        )

        # Conjugate weights
        self.W_bias_bar = nn.Parameter(torch.randn(link_channels_in, link_channels_out, spatial_dim, spatial_dim))
        self.W_cos_bar = nn.Parameter(
            torch.randn(link_channels_in, link_channels_out, joint_channels, spatial_dim, spatial_dim)
        )
        self.W_sin_bar = nn.Parameter(
            torch.randn(link_channels_in, link_channels_out, joint_channels, spatial_dim, spatial_dim)
        )

        self._init_weights()

    # def _init_weights(self):
    #     """Initialize weights properly"""
    #     if self.use_stable_init:
    #         # Stable initialization for 4D/5D tensors
    #         spatial_dim_sq = self.W_bias.shape[-1] * self.W_bias.shape[-2]  # spatial_dim^2
    #         fan_in = self.C_L * spatial_dim_sq
    #         fan_out = self.C_L_out * spatial_dim_sq
    #         std = math.sqrt(2.0 / (fan_in + fan_out)) * 0.1
    #
    #         for w in [self.W_bias, self.W_cos, self.W_sin,
    #                   self.W_bias_bar, self.W_cos_bar, self.W_sin_bar]:
    #             nn.init.normal_(w, mean=0.0, std=std)
    #     else:
    #         # Original: Xavier/Glorot initialization
    #         for w in [self.W_bias, self.W_cos, self.W_sin,
    #                   self.W_bias_bar, self.W_cos_bar, self.W_sin_bar]:
    #             nn.init.xavier_uniform_(w)

    def _init_weights(self):
        if self.use_stable_init:
            ################### Positive Init ####################
            spatial_dim_sq = self.spatial_dim * self.spatial_dim
            fan_in = self.C_L * spatial_dim_sq
            fan_out = self.C_L_out * spatial_dim_sq
            std = math.sqrt(2.0 / (fan_in + fan_out)) * 0.1

            for w in [self.W_cos, self.W_sin, self.W_cos_bar, self.W_sin_bar]:
                nn.init.normal_(w, mean=0.0, std=std)

            nn.init.normal_(self.W_bias, mean=0.001, std=std)
            nn.init.normal_(self.W_bias_bar, mean=0.001, std=std)
        else:
            for w in [self.W_bias, self.W_cos, self.W_sin, self.W_bias_bar, self.W_cos_bar, self.W_sin_bar]:
                nn.init.xavier_uniform_(w)

    # def _init_weights(self):
    #     if self.use_stable_init:
    #         ################### Low std init ####################
    #         std = 0.001
    #
    #         for w in [self.W_cos, self.W_sin, self.W_cos_bar, self.W_sin_bar]:
    #             nn.init.normal_(w, mean=0.0, std=std)
    #
    #         nn.init.zeros_(self.W_bias)
    #         nn.init.zeros_(self.W_bias_bar)
    #
    #         with torch.no_grad():
    #             for i in range(min(self.spatial_dim, self.spatial_dim)):
    #                 self.W_bias[..., i, i] = 0.1 / self.C_L
    #                 self.W_bias_bar[..., i, i] = 0.1 / self.C_L
    #     else:
    #         for w in [self.W_bias, self.W_cos, self.W_sin,
    #                   self.W_bias_bar, self.W_cos_bar, self.W_sin_bar]:
    #             nn.init.xavier_uniform_(w)

    def forward(self, f_parent: torch.Tensor, theta_joint: torch.Tensor):
        """
        Args:
            f_parent: (batch, C_L, 4, 4) - parent link features
            theta_joint: (batch, C_J) - joint features

        Returns:
            F_child: (batch, C'_L, 4, 4) - child link features
        """
        # Compute trigonometric functions
        cos_theta = torch.cos(theta_joint)  # (batch, C_J)
        sin_theta = torch.sin(theta_joint)  # (batch, C_J)

        # Equation 6: Compute transformation matrices U and Ū
        U = self.W_bias.unsqueeze(0)
        U = U + torch.einsum("bc,ijcde->bijde", cos_theta, self.W_cos)
        U = U + torch.einsum("bc,ijcde->bijde", sin_theta, self.W_sin)

        U_bar = self.W_bias_bar.unsqueeze(0)
        U_bar = U_bar + torch.einsum("bc,ijcde->bijde", cos_theta, self.W_cos_bar)
        U_bar = U_bar + torch.einsum("bc,ijcde->bijde", sin_theta, self.W_sin_bar)

        # Equation 8: Apply transformations and sum over input channels
        F_left = torch.einsum("bijde,bief->bjdf", U_bar, f_parent)
        F_right = torch.einsum("bide,bijef->bjdf", f_parent, U)

        return F_right + F_left


class RodriguesLayer(nn.Module):
    """
    Rodrigues Layer: Applies Rodrigues Operator along the kinematic tree.

    Uses sparse indexing matching entity's joint indices.
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        joint_channels: int,
        link_channels: int,
        spatial_dim: int,
        parent_contribution: float = 1.0,
        use_stable_init: bool = False,
        learnable_contribution_weight: bool = False,
        use_global_layer_norm: bool = False,
    ):
        super().__init__()
        self.tree = kinematic_tree
        self.joint_channels = joint_channels
        self.link_channels = link_channels
        self.learnable_contribution_weight = learnable_contribution_weight

        if learnable_contribution_weight:
            init_scale = 0.1
            num_joints = len(kinematic_tree.get_active_joint_indices())
            init_logit = math.log(init_scale / (1 - init_scale))
            self.contribution_weight = nn.Parameter(torch.full((num_joints,), init_logit))
        else:
            self.register_buffer("contribution_weight", torch.tensor(parent_contribution))

        # Sparse operator array matching entity joint indices
        # operators[entity_joint_idx] = operator for that joint
        self.operators = nn.ModuleList([None] * len(kinematic_tree.joints))

        for entity_joint_idx, joint_info in enumerate(kinematic_tree.joints):
            if joint_info is not None:
                self.operators[entity_joint_idx] = MultiChannelRodriguesOperator(
                    joint_channels=joint_channels,
                    link_channels_in=link_channels,
                    link_channels_out=link_channels,
                    spatial_dim=spatial_dim,
                    use_stable_init=use_stable_init,
                )

        # Layer normalization for each link
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm([link_channels, spatial_dim, spatial_dim]) for _ in range(kinematic_tree.num_bodies)]
        )

        self.use_global_layer_norm = use_global_layer_norm
        if self.use_global_layer_norm:
            # Normalizing over all link_features (excluding Batch/Body dimensions)
            global_norm_dim = link_channels * spatial_dim * spatial_dim
            self.global_layer_norm = nn.LayerNorm(global_norm_dim)

    def forward(self, joint_features: torch.Tensor, link_features: torch.Tensor):
        """
        Args:
            joint_features: (batch, max_joint_idx+1, C_J) - sparse, indexed by entity joint idx
            link_features: (batch, num_bodies, C_L, d, d)

        Returns:
            link_features_out: (batch, num_bodies, C_L, d, d)
        """
        link_features_in = link_features
        link_features_out = link_features.clone()

        # Process each active joint (Eq. 9-10) - Body-wise LayerNorm applied here
        for idx, entity_joint_idx in enumerate(self.tree.get_active_joint_indices()):
            joint_info = self.tree.joints[entity_joint_idx]
            parent_idx = joint_info["parent_link"]
            child_idx = joint_info["child_link"]

            F_parent = link_features_in[:, parent_idx]  # (batch, C_L, 4, 4)
            F_child_in = link_features_in[:, child_idx]  # (batch, C_L, 4, 4)
            Theta_j = joint_features[:, entity_joint_idx]  # (batch, C_J)

            # Equation 9: Transform parent feature
            F_trans = self.operators[entity_joint_idx](F_parent, Theta_j)

            # Equation 10: Residual connection + Body-wise LayerNorm (Kept as is)

            if self.learnable_contribution_weight:
                weight = torch.sigmoid(self.contribution_weight[idx])
            else:
                weight = self.contribution_weight

            F_child_out = self.layer_norms[child_idx](F_child_in + weight * F_trans)
            link_features_out[:, child_idx] = F_child_out

        # Root link: just normalize (Body-wise LayerNorm)
        root_idx = self.tree.root_idx
        link_features_out[:, root_idx] = self.layer_norms[root_idx](link_features_in[:, root_idx])

        if not self.use_global_layer_norm:
            return link_features_out

        # -------------------------------------------------------------
        # ⭐ MODIFICATION: Global Feature Stabilization ⭐
        # Apply LayerNorm to the entire set of link features to stabilize the std.
        # -------------------------------------------------------------

        B, N, C, D, D = link_features_out.shape

        # 1. Flatten the feature dimensions (C, D, D) -> (C*D*D)
        # Resulting shape: (B, N, C*D*D)
        flat_features = link_features_out.flatten(start_dim=2)

        # 2. Apply Global Layer Normalization
        # This normalizes over the C*D*D dimension, stabilizing the feature distribution.
        # This is the key step to prevent std oscillation.
        stable_features = self.global_layer_norm(flat_features)

        # 3. Restore original shape
        link_features_out_stable = stable_features.unflatten(dim=2, sizes=(C, D, D))

        return link_features_out_stable


class JointLayer(nn.Module):
    """
    Joint Layer: Updates joint features from link features.

    For each joint j, transforms its child link feature to update the joint feature
    using a joint-specific linear transformation (Eq. 11).

    This implementation vectorizes the operation across all active joints for efficiency.

    Args:
        kinematic_tree: KinematicTree defining the robot structure
        joint_channels: Number of joint feature channels (C_J)
        link_channels: Number of link feature channels (C_L)
        spatial_dim: Spatial dimension of link features (4 for 4×4, 6 for 6×6)
    """

    def __init__(self, kinematic_tree: "KinematicTree", joint_channels: int, link_channels: int, spatial_dim: int):
        super().__init__()
        self.tree = kinematic_tree

        # Pre-compute active joint indices and their child link mappings
        active = kinematic_tree.get_active_joint_indices()
        self.register_buffer("active_joints", torch.tensor(active, dtype=torch.long))
        self.register_buffer(
            "child_links", torch.tensor([kinematic_tree.joints[j]["child_link"] for j in active], dtype=torch.long)
        )

        # Batched linear transformation weights for all joints
        # Shape: (num_active_joints, C_J, C_L * 16)
        num_active = len(active)
        self.weight = nn.Parameter(torch.randn(num_active, joint_channels, link_channels * spatial_dim * spatial_dim))
        self.bias = nn.Parameter(torch.zeros(num_active, joint_channels))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, joint_features: torch.Tensor, link_features: torch.Tensor) -> torch.Tensor:
        """
        Update joint features based on child link features.

        Args:
            joint_features: (batch, max_joint_idx+1, C_J) - Input joint features {Θ_j^in}
            link_features: (batch, num_bodies, C_L, 4, 4) - Input link features {F_l^in}

        Returns:
            joint_features_out: (batch, max_joint_idx+1, C_J) - Updated joint features {Θ_j^out}
        """
        batch = link_features.shape[0]
        device = link_features.device

        # Move buffers to device if needed
        active_joints = self.active_joints.to(device)
        child_links = self.child_links.to(device)

        # Gather child link features for all active joints
        # Shape: (batch, num_active_joints, C_L, 4, 4)
        F_children = link_features[:, child_links]

        # Flatten link features
        # Shape: (batch, num_active_joints, C_L * 16)
        F_flat = F_children.flatten(start_dim=2)

        # Apply batched linear transformation for all joints
        # weight: (num_active, C_J, C_L*16)
        # F_flat: (batch, num_active, C_L*16)
        # Output: (batch, num_active, C_J)
        Theta_transformed = torch.einsum("ncd,bnd->bnc", self.weight, F_flat) + self.bias

        # Equation 11: Residual connection
        # Θ_j^out = Linear_j(Flatten(F_{cj}^in)) + Θ_j^in
        Theta_in = joint_features[:, active_joints]
        Theta_out = Theta_transformed + Theta_in

        # Scatter updated features back to sparse joint array
        result = joint_features.clone()
        result[:, active_joints] = Theta_out

        return result


class SelfAttentionLayer(nn.Module):
    """
    Self-Attention Layer: Enables global communication across all links.

    Projects link features to tokens, applies multi-head self-attention,
    then projects back with residual connection and normalization.

    Args:
        kinematic_tree: KinematicTree (for num_bodies)
        link_channels: Number of link feature channels (C_L)
        spatial_dim: Spatial dimension of link features (4 for 4×4, 6 for 6×6)
        embed_dim: Embedding dimension for attention
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        link_channels: int,
        spatial_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        self.tree = kinematic_tree
        self.link_channels = link_channels
        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Project link features to tokens
        self.to_token = nn.Linear(link_channels * spatial_dim * spatial_dim, embed_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Project tokens back to link features
        self.to_feature = nn.Linear(embed_dim, link_channels * spatial_dim * spatial_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(link_channels * spatial_dim * spatial_dim)

    def forward(self, link_features):
        """
        Args:
            link_features: (batch, num_bodies, C_L, spatial_dim, spatial_dim)

        Returns:
            link_features_out: (batch, num_bodies, C_L, spatial_dim, spatial_dim)
        """
        batch, num_bodies, C_L, d1, d2 = link_features.shape

        # Flatten link features to tokens
        link_flat = link_features.flatten(start_dim=2)  # (batch, num_bodies, C_L*16)
        tokens = self.to_token(link_flat)  # (batch, num_bodies, embed_dim)

        # Self-attention
        attended_tokens, _ = self.attention(
            tokens, tokens, tokens, need_weights=False
        )  # (batch, num_bodies, embed_dim)

        # Project back to link feature space
        attended_flat = self.to_feature(attended_tokens)  # (batch, num_bodies, C_L*16)

        # Residual + LayerNorm (on flattened features)
        out_flat = self.layer_norm(link_flat + attended_flat)

        # Reshape back
        link_features_out = out_flat.reshape(batch, num_bodies, C_L, d1, d2)

        return link_features_out


class RodriguesBlock(nn.Module):
    """
    Rodrigues Block: Combines RodriguesLayer, JointLayer, and SelfAttentionLayer.

    This is the basic building block of the Rodrigues Network.
    Multiple blocks are stacked to form the complete network.
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        joint_channels: int,
        link_channels: int,
        spatial_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        self.rodrigues_layer = RodriguesLayer(
            kinematic_tree=kinematic_tree,
            joint_channels=joint_channels,
            link_channels=link_channels,
            spatial_dim=spatial_dim,
        )

        self.joint_layer = JointLayer(
            kinematic_tree=kinematic_tree,
            joint_channels=joint_channels,
            link_channels=link_channels,
            spatial_dim=spatial_dim,
        )

        self.self_attention = SelfAttentionLayer(
            kinematic_tree=kinematic_tree,
            link_channels=link_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            spatial_dim=spatial_dim,
        )

    def forward(self, joint_features, link_features, global_token=None):
        """
        Args:
            joint_features: (batch, max_joint_idx+1, C_J)
            link_features: (batch, num_bodies, C_L, 4, 4)
            global_token: Optional, (batch, token_dim) - not used yet

        Returns:
            joint_features_out: (batch, max_joint_idx+1, C_J)
            link_features_out: (batch, num_bodies, C_L, 4, 4)
            global_token_out: Same as input (not processed yet)
        """
        # 1. Rodrigues Layer: Update link features
        link_features = self.rodrigues_layer(joint_features, link_features)

        # 2. Joint Layer: Update joint features
        joint_features = self.joint_layer(joint_features, link_features)

        # 3. Self-Attention: Global communication
        link_features = self.self_attention(link_features)

        return joint_features, link_features, global_token


class RodriguesFeatureExtractor(nn.Module):
    """
    Rodrigues Feature Extractor: Prepares input features for Rodrigues Network.

    Takes input vector (time step + observation + noisy action) and produces:
    - Joint features via DoF separate linear transformations
    - Link features via (1 + DoF) separate linear transformations
    - Optional global token via one additional linear transformation

    Based on Diffusion Policy framework where the network acts as denoising backbone.

    Args:
        kinematic_tree: KinematicTree defining robot structure
        obs_dim: Dimension of input vector
        joint_channels: Number of joint feature channels (C_J)
        link_channels: Number of link feature channels (C_L)
        global_token_dim: Dimension of global token (optional)
        use_global_token: Whether to use global token (e.g., for gripper)
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        joint_channels: int,
        link_channels: int,
        spatial_dim: int,
        global_token_dim: int = 128,
        use_global_token: bool = False,
    ):
        super().__init__()
        self.tree = kinematic_tree
        self.obs_dim = obs_dim
        self.joint_channels = joint_channels
        self.link_channels = link_channels
        self.spatial_dim = spatial_dim
        self.use_global_token = use_global_token

        # Pre-compute active joint indices
        active_joints = kinematic_tree.get_active_joint_indices()

        self.register_buffer("active_joint_indices", torch.tensor(active_joints, dtype=torch.long))
        num_active_joints = len(active_joints)
        num_bodies = kinematic_tree.num_bodies

        # Batched linear transformations for all joints
        # Instead of ModuleList, use single parameter tensor
        self.joint_weight = nn.Parameter(torch.randn(num_active_joints, joint_channels, obs_dim))
        self.joint_bias = nn.Parameter(torch.zeros(num_active_joints, joint_channels))

        # Batched linear transformations for all links
        self.link_weight = nn.Parameter(torch.randn(num_bodies, link_channels * spatial_dim * spatial_dim, obs_dim))
        self.link_bias = nn.Parameter(torch.zeros(num_bodies, link_channels * spatial_dim * spatial_dim))

        # Optional: Global token projection
        if use_global_token:
            self.global_projection = nn.Linear(obs_dim, global_token_dim)
        else:
            self.global_projection = None
        self._init_weights()

    def _init_weights(self):
        """Initialize all projection weights"""
        nn.init.xavier_uniform_(self.joint_weight)
        nn.init.zeros_(self.joint_bias)

        nn.init.xavier_uniform_(self.link_weight)
        nn.init.zeros_(self.link_bias)

        if self.global_projection is not None:
            nn.init.xavier_uniform_(self.global_projection.weight)
            nn.init.zeros_(self.global_projection.bias)

    def forward(self, input_vector: torch.Tensor):
        """
        Vectorized forward pass without loops.

        Args:
            input_vector: (batch, obs_dim)
                Concatenation of [time_embedding, observation, noisy_action]

        Returns:
            joint_features: (batch, max_joint_idx+1, C_J) - Sparse joint features
            link_features: (batch, num_bodies, C_L, 4, 4) - Link features
            global_token: (batch, token_dim) or None - Optional global token
        """
        batch = input_vector.shape[0]
        device = input_vector.device

        # ===== Joint Features (Vectorized) =====
        # Batched matrix multiplication for all joints at once
        # joint_weight: (num_active, C_J, obs_dim)
        # input_vector: (batch, obs_dim)
        # Output: (batch, num_active, C_J)
        joint_features_active = torch.einsum(
            "nco,bo->bnc", self.joint_weight, input_vector
        ) + self.joint_bias.unsqueeze(0)  # (batch, num_active, C_J)

        # Scatter to sparse array
        max_joint_idx = len(self.tree.joints)
        joint_features = torch.zeros(batch, max_joint_idx, self.joint_channels, device=device)

        active_indices = self.active_joint_indices.to(device)
        # Expand indices for scatter
        indices = active_indices.unsqueeze(0).unsqueeze(-1).expand(batch, -1, self.joint_channels)
        joint_features.scatter_(1, indices, joint_features_active)

        # ===== Link Features (Vectorized) =====
        # Batched matrix multiplication for all links at once
        # link_weight: (num_bodies, C_L*16, obs_dim)
        # input_vector: (batch, obs_dim)
        # Output: (batch, num_bodies, C_L*16)
        link_features_flat = torch.einsum("nlo,bo->bnl", self.link_weight, input_vector) + self.link_bias.unsqueeze(
            0
        )  # (batch, num_bodies, C_L*16)

        # Reshape to 4×4 matrices
        link_features = link_features_flat.reshape(
            batch, self.tree.num_bodies, self.link_channels, self.spatial_dim, self.spatial_dim
        )

        # ===== Global Token (Optional) =====
        global_token = None
        if self.use_global_token:
            global_token = self.global_projection(input_vector)  # (batch, token_dim)

        return joint_features, link_features, global_token


class RodriguesEncoder(nn.Module):
    """
    Rodrigues Encoder: Full encoding pipeline from observations to processed features.

    Architecture:
        Observation → FeatureExtractor → RodriguesBlocks (×12) → Encoded Features

    Takes raw observations and outputs processed joint/link features ready for decoder.

    Args:
        kinematic_tree: KinematicTree defining robot structure
        obs_dim: Observation dimension
        joint_channels: Number of joint feature channels (C_J)
        link_channels: Number of link feature channels (C_L)
        num_blocks: Number of Rodrigues Blocks to stack
        embed_dim: Embedding dimension for self-attention
        num_heads: Number of attention heads
        use_global_token: Whether to use global token
        global_token_dim: Dimension of global token
    """

    spatial_dim: int = 4

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        joint_channels: int = 4,
        link_channels: int = 8,
        num_blocks: int = 3,
        embed_dim: int = 256,
        num_heads: int = 4,
        use_global_token: bool = True,
        global_token_dim: int = 128,
    ):
        super().__init__()
        self.tree = kinematic_tree
        self.obs_dim = obs_dim
        self.joint_channels = joint_channels
        self.link_channels = link_channels
        self.num_blocks = num_blocks

        # Feature Extractor: Observation → Initial Features
        self.feature_extractor = RodriguesFeatureExtractor(
            kinematic_tree=kinematic_tree,
            obs_dim=obs_dim,
            joint_channels=joint_channels,
            link_channels=link_channels,
            global_token_dim=global_token_dim,
            use_global_token=use_global_token,
            spatial_dim=self.spatial_dim,
        )

        # Rodrigues Blocks: Process features through kinematic structure
        self.blocks = nn.ModuleList(
            [
                RodriguesBlock(
                    kinematic_tree=kinematic_tree,
                    joint_channels=joint_channels,
                    link_channels=link_channels,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    spatial_dim=self.spatial_dim,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, observations: torch.Tensor):
        """
        Encode observations to processed features.

        Args:
            observations: (batch, obs_dim) - Raw observations

        Returns:
            joint_features: (batch, max_joint_idx+1, C_J) - Processed joint features
            link_features: (batch, num_bodies, C_L, 4, 4) - Processed link features
            global_token: (batch, token_dim) or None - Processed global token
        """
        # Step 1: Extract initial features from observations
        joint_features, link_features, global_token = self.feature_extractor(observations)

        # Step 2: Process through Rodrigues Blocks
        for block in self.blocks:
            joint_features, link_features, global_token = block(joint_features, link_features, global_token)

        return joint_features, link_features, global_token
