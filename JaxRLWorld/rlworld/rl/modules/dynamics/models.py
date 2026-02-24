from typing import List, Tuple

import torch
import torch.nn as nn

from .base import DynamicsModel


# ============================================================================
# Decoder
# ============================================================================

class DynamicsDecoder(nn.Module):
    """Decoder: features + action → delta"""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()

        layers = []
        in_dim = feature_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, feature_dim)
            action: (batch, action_dim)
        Returns:
            delta: (batch, output_dim)
        """
        x = torch.cat([features, action], dim=-1)
        delta = self.network(x)
        return delta


# ============================================================================
# Complete Dynamics Models
# ============================================================================


class HybridPhysicsInformedDynamics(DynamicsModel):
    """HybridABARodriguesEncoder + Decoder"""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
            action: (batch, action_dim)
        Returns:
            next_x_pred: (batch, output_dim)
        """
        # Encoder returns (joint_feat, link_feat, global_token)
        joint_feat = self.encoder(x)

        # Flatten and concatenate
        features = joint_feat.flatten(start_dim=1)
        # features = torch.cat([joint_flat, link_flat], dim=-1)

        # Decode
        delta = self.decoder(features, action)
        return x + delta


class MLPDynamics(DynamicsModel):
    """Standard MLP baseline"""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [512, 512, 256]
    ):
        super().__init__()

        layers = []
        in_dim = input_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.LayerNorm(hidden_dim)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
            action: (batch, action_dim)
        Returns:
            next_x_pred: (batch, output_dim)
        """
        inp = torch.cat([x, action], dim=-1)
        delta = self.network(inp)
        return x + delta


# ============================================================================
# Model Factory
# ============================================================================

def create_models(
    input_dim: int,
    action_dim: int,
    output_dim: int,
    kinematic_tree,
    encoder_type: str = 'simple',
    encoder_config: dict = None,
    decoder_config: dict = None,
    mlp_config: dict = None,
    device: str = 'cuda'
) -> Tuple[DynamicsModel, DynamicsModel]:
    """
    Create physics-informed and MLP models.

    Args:
        input_dim: Input dimension
        action_dim: Action dimension
        output_dim: Output dimension
        kinematic_tree: Robot kinematic tree (for hybrid encoder)
        encoder_type: 'simple' or 'hybrid'
        encoder_config: Config dict for encoder
        decoder_config: Config dict for decoder
        mlp_config: Config dict for MLP
        device: Device to run on

    Returns:
        (physics_model, mlp_model) - Both are DynamicsModel instances
    """
    encoder_config = encoder_config or {}
    decoder_config = decoder_config or {}
    mlp_config = mlp_config or {}

    # Create Encoder
    if encoder_type == 'hybrid':
        from rlworld.rl.modules.architectures.aba.encoder import ABAEncoder
        encoder = ABAEncoder(
            kinematic_tree=kinematic_tree,
            obs_dim=input_dim,
            **encoder_config
        ).to(device)

        # Compute feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, input_dim).to(device)
            joint_feat = encoder(dummy_input)
            joint_flat = joint_feat.flatten(start_dim=1)
            feature_dim = joint_flat.shape[1]

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    # Create Decoder
    decoder = DynamicsDecoder(
        feature_dim=feature_dim,
        action_dim=action_dim,
        output_dim=output_dim,
        hidden_dims=decoder_config.get('hidden_dims', [256, 256, 256])
    ).to(device)

    # Create Physics Model
    if encoder_type == 'hybrid':
        physics_model = HybridPhysicsInformedDynamics(encoder, decoder).to(device)

    # Create MLP Baseline
    mlp_model = MLPDynamics(
        input_dim=input_dim,
        action_dim=action_dim,
        output_dim=output_dim,
        hidden_dims=mlp_config.get('hidden_dims', [512, 512, 256])
    ).to(device)

    return physics_model, mlp_model


def print_model_summary(physics_model: DynamicsModel, mlp_model: DynamicsModel):
    """Print model parameter summary"""

    print("\n" + "=" * 70)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 70)

    # Physics-informed model
    if hasattr(physics_model, 'encoder'):
        encoder_params = sum(p.numel() for p in physics_model.encoder.parameters())
        decoder_params = sum(p.numel() for p in physics_model.decoder.parameters())
        physics_total = encoder_params + decoder_params

        print("\nPhysics-Informed Model:")
        print(f"  Encoder:       {encoder_params:>12,}")
        print(f"  Decoder:       {decoder_params:>12,}")
        print(f"  {'─' * 40}")
        print(f"  Total:         {physics_total:>12,}")
    else:
        physics_total = sum(p.numel() for p in physics_model.parameters())
        print("\nPhysics-Informed Model:")
        print(f"  Total:         {physics_total:>12,}")

    # MLP model
    mlp_total = sum(p.numel() for p in mlp_model.parameters())

    print("\nMLP Baseline:")
    print(f"  Total:         {mlp_total:>12,}")

    print("\nComparison:")
    print(f"  Parameter ratio: {physics_total / mlp_total:.2f}x")
