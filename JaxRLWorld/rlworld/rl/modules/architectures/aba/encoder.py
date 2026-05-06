from typing import TYPE_CHECKING

import equinox as eqx
import jax

from .layers import PerBodyABABottomUpLayer

if TYPE_CHECKING:
    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


class ABAEncoder(eqx.Module):
    """
    Encoder using ABA bottom-up pass.
    """

    aba_bottom_up: PerBodyABABottomUpLayer

    num_bodies: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)
    _output_dim: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        obs_dim: int,
        link_channels: int = 8,
        spatial_dim: int = 6,
        learnable_contribution_weight: bool = False,
        use_global_layer_norm: bool = False,
        use_positive_constraint: bool = True,
        *,
        key: jax.Array,
        **kwargs,
    ):
        self.num_bodies = kinematic_tree.num_bodies
        self.obs_dim = obs_dim

        self.aba_bottom_up = PerBodyABABottomUpLayer(
            kinematic_tree=kinematic_tree,
            obs_dim=obs_dim,
            link_channels=link_channels,
            spatial_dim=spatial_dim,
            learnable_contribution_weight=learnable_contribution_weight,
            use_global_layer_norm=use_global_layer_norm,
            use_positive_constraint=use_positive_constraint,
            key=key,
        )

        self._output_dim = (self.num_bodies, link_channels * spatial_dim)

    def __call__(self, observations: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        """
        Args:
            observations: (obs_dim,) unbatched

        Returns:
            features: (num_bodies, link_channels * spatial_dim)
        """
        aba_features = self.aba_bottom_up(observations)  # (N, C, d)
        return aba_features.reshape(self.num_bodies, -1)  # (N, C*d)

    def compute_auxiliary_loss(self, observations: jax.Array) -> jax.Array:
        return self.aba_bottom_up.compute_orthogonality_loss(observations)

    @property
    def output_dim(self) -> tuple[int, int]:
        return self._output_dim


def create_encoder(
    encoder_type: str, kinematic_tree: "KinematicTree", obs_dim: int, *, key: jax.Array, **kwargs
) -> eqx.Module:
    """
    Factory function for creating encoders.

    Args:
        encoder_type: "ABAEncoder", "MLPEncoder", etc.
        kinematic_tree: Robot kinematic structure
        obs_dim: Observation dimension
        key: JAX random key
        **kwargs: Encoder-specific arguments

    Returns:
        Encoder instance
    """
    encoder_map = {
        "ABAEncoder": ABAEncoder,
    }

    if encoder_type not in encoder_map:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: {list(encoder_map.keys())}")

    encoder_class = encoder_map[encoder_type]
    return encoder_class(kinematic_tree=kinematic_tree, obs_dim=obs_dim, key=key, **kwargs)
