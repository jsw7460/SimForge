import math
import torch
from torch import nn


class ActiveJointDecoder(nn.Module):
    """
    Decoder that selects active joints and applies per-joint projection.

    For robots where num_joints > num_actions (has fixed joints).

    Args:
        active_joint_indices: List of active joint indices
        latent_dim: Input latent dimension per joint
    """

    def __init__(
        self,
        active_joint_indices: list[int],
        latent_dim: int,
    ):
        super().__init__()
        self.num_active_joints = len(active_joint_indices)
        self.latent_dim = latent_dim

        # Register as buffer for device handling
        self.register_buffer(
            'active_joint_indices',
            torch.tensor(active_joint_indices, dtype=torch.long)
        )

        # Per-joint projection: latent_dim -> 1
        self.joint_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.SiLU(),
                nn.Linear(latent_dim // 2, 1),
            )
            for _ in range(self.num_active_joints)
        ])

        self._init_weights()

    def _init_weights(self):
        for proj in self.joint_projections:
            modules = [m for m in proj if isinstance(m, nn.Linear)]
            for module in modules[:-1]:
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
            # Output layer with small gain
            nn.init.orthogonal_(modules[-1].weight, gain=0.01)
            nn.init.zeros_(modules[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_joints, latent_dim)

        Returns:
            actions: (B, num_active_joints)
        """
        # Select active joints
        x_active = x[:, self.active_joint_indices]  # (B, num_active_joints, latent_dim)

        # Per-joint projection
        outputs = []
        for i, proj in enumerate(self.joint_projections):
            out = proj(x_active[:, i])  # (B, 1)
            outputs.append(out)

        actions = torch.cat(outputs, dim=-1)  # (B, num_active_joints)
        return actions


class SimpleJointDecoder(nn.Module):
    """
    Decoder that applies per-joint projection to all joints.

    For robots where num_joints == num_actions (all joints are active).

    Args:
        num_joints: Number of joints
        latent_dim: Input latent dimension per joint
    """

    def __init__(
        self,
        num_joints: int,
        latent_dim: int,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.latent_dim = latent_dim

        # Per-joint projection: latent_dim -> 1
        self.joint_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.SiLU(),
                nn.Linear(latent_dim // 2, 1),
            )
            for _ in range(num_joints)
        ])

        self._init_weights()

    def _init_weights(self):
        for proj in self.joint_projections:
            modules = [m for m in proj if isinstance(m, nn.Linear)]
            for module in modules[:-1]:
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
            # Output layer with small gain
            nn.init.orthogonal_(modules[-1].weight, gain=0.01)
            nn.init.zeros_(modules[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_joints, latent_dim)

        Returns:
            actions: (B, num_joints)
        """
        # Per-joint projection
        outputs = []
        for i, proj in enumerate(self.joint_projections):
            out = proj(x[:, i])  # (B, 1)
            outputs.append(out)

        actions = torch.cat(outputs, dim=-1)  # (B, num_joints)
        return actions
