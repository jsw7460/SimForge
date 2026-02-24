from typing import TYPE_CHECKING, Literal

import math
import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree

__all__ = [
    "CRBABottomUpLayer",
    "CRBAAttentionBiasedEncoder",
]


class CRBABottomUpLayer(nn.Module):
    """
    Computes composite rigid body inertia and joint-space inertia matrix H.

    Physics:
        - I_i^c = I_i + sum_{children} I_j^c  (composite inertia, bottom-up)
        - H_ij = S_i^T @ I_i^c @ S_j  (joint-space inertia)

    Each joint has its own observation projection (per-joint processing).

    Args:
        kinematic_tree: Robot kinematic structure
        obs_dim: Observation dimension
        num_heads: Number of attention heads (= number of channels)
        spatial_dim: Spatial dimension for inertia matrix (d x d)
        orth_loss_weight: Weight for orthogonality regularization
        orth_loss_min_weight: Minimum weight after annealing
        orth_loss_decay: Decay factor per step
        use_auxiliary_loss: Whether to compute auxiliary loss
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        num_heads: int = 4,
        spatial_dim: int = 6,
        orth_loss_weight: float = 1.0,
        orth_loss_min_weight: float = 0.05,
        orth_loss_decay: float = 1.0,
        use_auxiliary_loss: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.tree = kinematic_tree
        self.obs_dim = obs_dim
        self.num_heads = num_heads
        self.spatial_dim = spatial_dim
        self.num_bodies = kinematic_tree.num_bodies
        self.use_auxiliary_loss = use_auxiliary_loss

        # Get joint information
        self.num_joints = kinematic_tree.num_joints
        self.joint_to_child = self._build_joint_to_child_mapping(kinematic_tree)
        self.child_to_joint = {v: k for k, v in self.joint_to_child.items()}

        # Annealing parameters
        self.orth_loss_min_weight = orth_loss_min_weight
        self.orth_loss_decay = orth_loss_decay
        self.register_buffer('_current_orth_weight', torch.tensor(orth_loss_weight))

        # === Per-body inertia projection ===
        # Each body has its own projection: obs -> I_i (full matrix)
        inertia_output_dim = num_heads * spatial_dim * spatial_dim
        self.inertia_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, inertia_output_dim * 2),
                nn.SiLU(),
                nn.Linear(inertia_output_dim * 2, inertia_output_dim),
            )
            for _ in range(self.num_bodies)
        ])

        # Learnable base inertia per body (full matrix)
        self.base_inertia = nn.Parameter(
            torch.zeros(self.num_bodies, num_heads, spatial_dim, spatial_dim)
        )

        # Per-body layer norm for composite inertia
        self.inertia_norms = nn.ModuleList([
            nn.LayerNorm([num_heads, spatial_dim, spatial_dim])
            for _ in range(self.num_bodies)
        ])

        # === Per-joint motion subspace (dual) ===
        # S_left and S_right for H_ij = S_left_i^T @ I^c @ S_right_j
        self.motion_subspace_left = nn.Parameter(
            torch.randn(self.num_joints, num_heads, spatial_dim)
        )
        self.motion_subspace_right = nn.Parameter(
            torch.randn(self.num_joints, num_heads, spatial_dim)
        )

        # Identity buffer for auxiliary loss
        self.register_buffer('_identity', torch.eye(spatial_dim))

        # Precompute tree structure for efficient bottom-up pass
        self._precompute_tree_structure()
        self._init_weights()

    def _build_joint_to_child_mapping(self, kinematic_tree) -> dict[int, int]:
        """Build mapping from joint index to child body index."""
        mapping = {}
        for joint_idx in range(kinematic_tree.num_joints):
            child_link = kinematic_tree.joints[joint_idx]["child_link"]
            mapping[joint_idx] = child_link
        return mapping

    def _precompute_tree_structure(self):
        """Precompute tree traversal order and child indices for batched operations."""
        # Bottom-up traversal order
        self.traversal_order = list(self.tree.traverse_bottom_up())

        # Child indices for each body (for batched aggregation)
        max_children = max(
            len(self.tree.get_children(i)) for i in range(self.num_bodies)
        )
        max_children = max(max_children, 1)

        child_indices = torch.zeros((self.num_bodies, max_children), dtype=torch.long)
        child_mask = torch.zeros(self.num_bodies, max_children, dtype=torch.bool)

        for body_idx in range(self.num_bodies):
            children = self.tree.get_children(body_idx)
            for i, c in enumerate(children):
                child_indices[body_idx, i] = c
                child_mask[body_idx, i] = True

        self.register_buffer('child_indices', child_indices)
        self.register_buffer('child_mask', child_mask)

        # Joint to child body mapping as tensor
        joint_child_indices = torch.tensor(
            [self.joint_to_child[j] for j in range(self.num_joints)],
            dtype=torch.long
        )
        self.register_buffer('joint_child_indices', joint_child_indices)

    def _init_weights(self):
        """Initialize parameters."""
        # Per-body inertia projections
        for proj in self.inertia_projections:
            modules = [m for m in proj if isinstance(m, nn.Linear)]
            for module in modules[:-1]:
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
            # Output layer with small gain
            nn.init.orthogonal_(modules[-1].weight, gain=0.01)
            nn.init.zeros_(modules[-1].bias)

        # Base inertia: initialize as scaled identity
        # with torch.no_grad():
        #     for body_idx in range(self.num_bodies):
        #         for h in range(self.num_heads):
        #             self.base_inertia.data[body_idx, h] = 0.1 * torch.eye(self.spatial_dim)

        # Motion subspaces: orthogonal initialization
        with torch.no_grad():
            for j in range(self.num_joints):
                for h in range(self.num_heads):
                    vec = torch.randn(self.spatial_dim)
                    vec = vec / (vec.norm() + 1e-8)
                    self.motion_subspace_left.data[j, h] = vec

                    vec = torch.randn(self.spatial_dim)
                    vec = vec / (vec.norm() + 1e-8)
                    self.motion_subspace_right.data[j, h] = vec

    def __getstate__(self):
        state = self.__dict__.copy()
        state['tree'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def step_annealing(self):
        """Decay orthogonality loss weight."""
        new_weight = max(
            self._current_orth_weight.item() * self.orth_loss_decay,
            self.orth_loss_min_weight
        )
        self._current_orth_weight.fill_(new_weight)

    def _compute_per_body_inertia(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute per-body inertia matrices from observations.

        Args:
            observations: (B, obs_dim)

        Returns:
            inertia: (B, num_bodies, num_heads, d, d)
        """
        batch = observations.shape[0]

        inertia_list = []
        for body_idx in range(self.num_bodies):
            proj_out = self.inertia_projections[body_idx](observations)
            proj_out = proj_out.reshape(
                batch, self.num_heads, self.spatial_dim, self.spatial_dim
            )

            # Add learnable base
            base = self.base_inertia[body_idx].unsqueeze(0)
            inertia = proj_out + base

            inertia = torch.nn.functional.softplus(inertia) + 1e-6

            inertia_list.append(inertia)

        return torch.stack(inertia_list, dim=1)

    def _compute_composite_inertia(
        self,
        body_inertia: torch.Tensor
    ) -> torch.Tensor:

        # Use dict instead of inplace tensor modification
        composite_dict = {i: body_inertia[:, i].clone() for i in range(self.num_bodies)}

        for body_idx in self.traversal_order:
            children = self.tree.get_children(body_idx)

            if children:
                child_sum = torch.zeros_like(composite_dict[body_idx])
                for child_idx in children:
                    child_sum = child_sum + composite_dict[child_idx]

                composite_dict[body_idx] = composite_dict[body_idx] + child_sum

            composite_dict[body_idx] = self.inertia_norms[body_idx](composite_dict[body_idx])

        # Stack back to tensor
        composite = torch.stack([composite_dict[i] for i in range(self.num_bodies)], dim=1)
        return composite

    def _compute_H_matrix(
        self,
        composite_inertia: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute joint-space inertia matrix H.

        H_ij = S_left_i^T @ I_i^c @ S_right_j

        Args:
            composite_inertia: (B, num_bodies, num_heads, d, d)

        Returns:
            H: (B, num_heads, num_joints, num_joints)
        """
        # Get composite inertia at joint child bodies
        I_c = composite_inertia[:, self.joint_child_indices]  # (B, J, H, d, d)

        # Normalize motion subspaces
        S_left = F.normalize(self.motion_subspace_left, dim=-1)   # (J, H, d)
        S_right = F.normalize(self.motion_subspace_right, dim=-1)  # (J, H, d)

        # S_left^T @ I_c: (B, J, H, d)
        # einsum: 'jhd,bjhde->bjhe'
        S_I = torch.einsum('ihd,bihdj->bihj', S_left, I_c)  # (B, J, H, d)

        # S_I @ S_right^T: (B, J, J, H)
        # einsum: 'bihd,jhd->bijh'
        H = torch.einsum('bihd,jhd->bijh', S_I, S_right)  # (B, J, J, H)

        # Permute to (B, H, J, J) for attention
        H = H.permute(0, 3, 1, 2)

        return H

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute joint-space inertia matrix H.

        Args:
            observations: (B, obs_dim)

        Returns:
            H: (B, num_heads, num_joints, num_joints) - attention bias
            composite_inertia: (B, num_bodies, num_heads, d, d) - for auxiliary loss
        """
        # Step 1: Per-body inertia from observations
        body_inertia = self._compute_per_body_inertia(observations)

        # Step 2: Bottom-up composite inertia
        composite_inertia = self._compute_composite_inertia(body_inertia)

        # Step 3: Joint-space inertia matrix
        H = self._compute_H_matrix(composite_inertia)
        return H, composite_inertia

    def compute_auxiliary_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary loss for regularization.

        Loss components:
        1. Symmetry encouragement for H (optional)
        2. Motion subspace diversity

        Args:
            observations: (B, obs_dim)

        Returns:
            loss: scalar tensor
        """
        if not self.use_auxiliary_loss:
            return torch.tensor(0.0, device=observations.device)

        # Motion subspace orthogonality within each joint
        S_left = self.motion_subspace_left   # (J, H, d)
        S_right = self.motion_subspace_right  # (J, H, d)

        # Encourage diversity across heads: S @ S^T should be diagonal-ish
        # For each joint, compute gram matrix across heads
        loss = torch.tensor(0.0, device=observations.device)

        for j in range(self.num_joints):
            # Left subspace gram: (H, H)
            gram_left = S_left[j] @ S_left[j].T
            off_diag_left = gram_left - torch.diag(gram_left.diag())
            loss = loss + (off_diag_left ** 2).mean()

            # Right subspace gram: (H, H)
            gram_right = S_right[j] @ S_right[j].T
            off_diag_right = gram_right - torch.diag(gram_right.diag())
            loss = loss + (off_diag_right ** 2).mean()

        loss = loss / (2 * self.num_joints)
        return self._current_orth_weight * loss


class CRBATokenizer(nn.Module):
    """
    Tokenizer that creates joint-level tokens using per-joint projections.

    Args:
        kinematic_tree: Robot kinematic structure
        obs_dim: Observation dimension
        embed_dim: Output embedding dimension
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        embed_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.num_joints = kinematic_tree.num_joints
        self.embed_dim = embed_dim

        # Per-joint projection
        self.joint_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, embed_dim * 2),
                nn.SiLU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            for _ in range(self.num_joints)
        ])

        self._init_weights()

    def _init_weights(self):
        for proj in self.joint_projections:
            modules = [m for m in proj if isinstance(m, nn.Linear)]
            for module in modules[:-1]:
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
            nn.init.orthogonal_(modules[-1].weight, gain=0.01)
            nn.init.zeros_(modules[-1].bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (B, obs_dim)

        Returns:
            tokens: (B, num_joints, embed_dim)
        """
        tokens = [proj(observations) for proj in self.joint_projections]
        return torch.stack(tokens, dim=1)


class CRBABiasedAttentionLayer(nn.Module):
    """
    Transformer layer with CRBA-based attention bias.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d) + H) @ V

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Multi-head attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable temperature for H scaling
        self.h_scale = nn.Parameter(torch.ones(num_heads))

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        H_bias: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input tokens
            H_bias: (B, num_heads, N, N) attention bias from CRBA
            attn_mask: (N, N) optional attention mask

        Returns:
            output: (B, N, D)
        """
        B, N, D = x.shape

        # Pre-norm
        x_norm = self.norm1(x)

        # QKV projections
        Q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, H, N, d)

        # Attention scores with bias
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)

        # Add H bias with learnable scale
        h_scale = self.h_scale.view(1, self.num_heads, 1, 1)
        attn_scores = attn_scores + h_scale * H_bias

        # Apply mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and apply to values
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, H, N, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        attn_output = self.out_proj(attn_output)

        # Residual connection
        x = x + self.dropout(attn_output)

        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))

        return x


class JointPositionalEmbedding(nn.Module):
    """
    Positional embedding for joints based on kinematic tree structure.

    Uses depth in tree as positional signal.
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        embed_dim: int,
    ):
        super().__init__()
        self.num_joints = kinematic_tree.num_joints
        self.embed_dim = embed_dim

        # Compute joint depths
        depths = self._compute_joint_depths(kinematic_tree)
        self.register_buffer('depths', torch.tensor(depths, dtype=torch.long))

        max_depth = max(depths) + 1
        self.depth_embedding = nn.Embedding(max_depth, embed_dim)

        # Additional learned embedding per joint
        self.joint_embedding = nn.Embedding(self.num_joints, embed_dim)

    def _compute_joint_depths(self, kinematic_tree) -> list[int]:
        """Compute depth of each joint in the tree."""
        depths = []
        for joint_idx in range(kinematic_tree.num_joints):
            child_link = kinematic_tree.joints[joint_idx]["child_link"]
            depth = 0
            current = child_link
            while kinematic_tree.get_parent(current) is not None:
                depth += 1
                current = kinematic_tree.get_parent(current)
            depths.append(depth)
        return depths

    def forward(self) -> torch.Tensor:
        """
        Returns:
            pe: (num_joints, embed_dim)
        """
        depth_pe = self.depth_embedding(self.depths)
        joint_pe = self.joint_embedding(torch.arange(self.num_joints, device=self.depths.device))
        return depth_pe + joint_pe


class CRBAAttentionBiasedEncoder(nn.Module):
    """
    Encoder using CRBA-derived joint-space inertia H as attention bias.

    Architecture:
        1. Per-joint tokenizer: obs -> joint tokens
        2. CRBA layer: obs -> H matrix (attention bias)
        3. Transformer layers with H-biased attention
        4. Output: (B, num_joints, latent_dim)

    The joint-space inertia H_ij captures inertial coupling between joints,
    guiding attention to focus on dynamically coupled joints.

    Args:
        kinematic_tree: Robot kinematic structure
        obs_dim: Observation dimension
        latent_dim: Output latent dimension
        num_heads: Number of attention heads (= CRBA channels)
        spatial_dim: Spatial dimension for inertia matrices
        num_layers: Number of transformer layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        use_adjacency_mask: Whether to use kinematic adjacency mask
        interleave_mask: Alternate mask usage across layers
        orth_loss_weight: Initial orthogonality loss weight
        orth_loss_decay: Decay factor for orthogonality loss
        use_auxiliary_loss: Whether to compute auxiliary loss
    """

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        latent_dim: int = 128,
        num_heads: int = 4,
        spatial_dim: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        use_adjacency_mask: bool = False,
        interleave_mask: bool = True,
        orth_loss_weight: float = 1.0,
        orth_loss_decay: float = 1.0,
        use_auxiliary_loss: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.tree = kinematic_tree
        self.num_joints = kinematic_tree.num_joints
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.use_adjacency_mask = use_adjacency_mask
        self.interleave_mask = interleave_mask

        # === Tokenizer ===
        self.tokenizer = CRBATokenizer(
            kinematic_tree=kinematic_tree,
            obs_dim=obs_dim,
            embed_dim=latent_dim,
        )

        # === CRBA layer for H computation ===
        self.crba_layer = CRBABottomUpLayer(
            kinematic_tree=kinematic_tree,
            obs_dim=obs_dim,
            num_heads=num_heads,
            spatial_dim=spatial_dim,
            orth_loss_weight=orth_loss_weight,
            orth_loss_decay=orth_loss_decay,
            use_auxiliary_loss=use_auxiliary_loss,
        )

        # === Positional embedding ===
        self.pe = JointPositionalEmbedding(
            kinematic_tree=kinematic_tree,
            embed_dim=latent_dim,
        )

        # === Transformer layers ===
        self.layers = nn.ModuleList([
            CRBABiasedAttentionLayer(
                embed_dim=latent_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # === Adjacency mask ===
        if use_adjacency_mask:
            adj = self._build_joint_adjacency(kinematic_tree)
            adj = adj + torch.eye(self.num_joints)
            mask = (adj == 0)
            self.register_buffer('attn_mask', mask)
        else:
            self.attn_mask = None

        # Output dimension
        self.output_dim = (self.num_joints, latent_dim)

        # Logging
        self.last_H = None
        self.last_composite_inertia = None

    def _build_joint_adjacency(self, kinematic_tree) -> torch.Tensor:
        """Build adjacency matrix for joints based on kinematic chain."""
        adj = torch.zeros(self.num_joints, self.num_joints)

        for i in range(self.num_joints):
            child_i = kinematic_tree.joints[i]["child_link"]
            parent_i = kinematic_tree.joints[i]["parent_link"]

            for j in range(self.num_joints):
                if i == j:
                    continue
                child_j = kinematic_tree.joints[j]["child_link"]
                parent_j = kinematic_tree.joints[j]["parent_link"]

                # Adjacent if connected through parent-child relationship
                if child_i == parent_j or child_j == parent_i:
                    adj[i, j] = 1

        return adj

    def __getstate__(self):
        state = self.__dict__.copy()
        state['tree'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (B, obs_dim)

        Returns:
            output: (B, num_joints, latent_dim)
        """
        # Step 1: Create joint tokens
        tokens = self.tokenizer(observations)  # (B, J, D)

        # Step 2: Add positional embedding
        pe = self.pe()  # (J, D)
        x = tokens + pe.unsqueeze(0)

        # Step 3: Compute H matrix from CRBA
        H, composite_inertia = self.crba_layer(observations)  # (B, H, J, J)

        # Store for logging
        self.last_H = H.detach()
        self.last_composite_inertia = composite_inertia.detach()

        # Step 4: Transformer layers with H bias
        for layer_idx, layer in enumerate(self.layers):
            if self.use_adjacency_mask:
                if self.interleave_mask:
                    use_mask = (layer_idx % 2 == 0)
                else:
                    use_mask = True
                mask = self.attn_mask if use_mask else None
            else:
                mask = None

            x = layer(x, H, attn_mask=mask)
        return x

    def post_update_step(self, *args, **kwargs):
        """Called after each optimizer step for annealing."""
        self.crba_layer.step_annealing()

    def compute_auxiliary_loss(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss from CRBA layer."""
        return self.crba_layer.compute_auxiliary_loss(observations)

    @property
    def extra_to_log(self) -> dict:
        """Extra metrics for logging."""
        extra = {}

        if self.last_H is not None:
            H = self.last_H
            extra["crba/H_mean"] = H.mean().item()
            extra["crba/H_std"] = H.std().item()
            extra["crba/H_max"] = H.max().item()
            extra["crba/H_min"] = H.min().item()

            # Symmetry measure: ||H - H^T|| / ||H||
            H_sym_diff = (H - H.transpose(-1, -2)).abs().mean()
            H_norm = H.abs().mean() + 1e-8
            extra["crba/H_asymmetry"] = (H_sym_diff / H_norm).item()

        # Log attention scale parameters
        for i, layer in enumerate(self.layers):
            h_scale = layer.h_scale.detach()
            extra[f"crba/layer{i}_h_scale_mean"] = h_scale.mean().item()
            extra[f"crba/layer{i}_h_scale_std"] = h_scale.std().item()

        # Current orthogonality loss weight
        extra["crba/orth_weight"] = self.crba_layer._current_orth_weight.item()

        return extra