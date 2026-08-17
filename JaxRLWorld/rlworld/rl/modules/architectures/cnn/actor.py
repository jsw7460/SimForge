"""Actor and critic that read image groups alongside the state vector.

The structure is ``rsl_rl.models.CNNModel``'s: every image group goes
through its own convolutional encoder, the resulting latents are
concatenated onto the (already normalised) state vector, and the whole
thing is fed to an ordinary MLP. Normalisation covers the state vector
only — an image is already in [0, 1] and running statistics over pixels
would drift with whatever the camera happens to be pointing at.
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.configs.common_config_classes import CNNEncoderCfg
from rlworld.rl.modules.architectures.base import BaseActor, BaseCritic
from rlworld.rl.modules.architectures.cnn.encoder import CNNEncoder

__all__ = ["VisionActor", "VisionCritic", "build_encoders"]


def build_encoders(
    obs_shapes: Mapping[str, tuple[int, ...]],
    image_groups: tuple[str, ...],
    cnn_cfg: CNNEncoderCfg,
    *,
    key: jax.Array,
) -> tuple[dict[str, CNNEncoder], int]:
    """One encoder per image group, plus the total latent width."""
    keys = jax.random.split(key, len(image_groups))
    encoders: dict[str, CNNEncoder] = {}
    latent_dim = 0
    for group_key, group in zip(keys, image_groups, strict=True):
        shape = obs_shapes[group]
        if len(shape) != 3:
            raise ValueError(f"Image group {group!r} must be (C, H, W) per env, got {shape}.")
        channels, height, width = shape
        encoder = CNNEncoder(
            input_hw=(height, width),
            input_channels=channels,
            output_channels=cnn_cfg.output_channels,
            kernel_size=cnn_cfg.kernel_size,
            stride=cnn_cfg.stride,
            dilation=cnn_cfg.dilation,
            padding=cnn_cfg.padding,
            activation=cnn_cfg.activation,
            spatial_softmax=cnn_cfg.spatial_softmax,
            spatial_softmax_temperature=cnn_cfg.spatial_softmax_temperature,
            key=group_key,
        )
        encoders[group] = encoder
        latent_dim += encoder.output_dim
    return encoders, latent_dim


def _encode(encoder: CNNEncoder, image: jax.Array) -> jax.Array:
    """Run one encoder over ``(C, H, W)`` or a batch of them.

    The MLP actors in this package batch themselves, and the critic is
    called with a batch directly, so the encoder has to do the same
    rather than depend on the caller having vmapped.
    """
    if image.ndim == 4:
        return jax.vmap(encoder)(image)
    return encoder(image)


def _latent(
    obs: Mapping[str, jax.Array],
    vector_group: str,
    image_groups: tuple[str, ...],
    encoders: dict[str, CNNEncoder],
) -> jax.Array:
    parts = [obs[vector_group]]
    parts.extend(_encode(encoders[group], obs[group]) for group in image_groups)
    return jnp.concatenate(parts, axis=-1)


class VisionActor(BaseActor):
    """MLP actor whose input is the state vector plus encoded images."""

    encoders: dict[str, CNNEncoder]
    trunk: BaseActor
    vector_group: str = eqx.field(static=True)
    image_groups: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        encoders: dict[str, CNNEncoder],
        trunk: BaseActor,
        vector_group: str,
        image_groups: tuple[str, ...],
    ):
        self.encoders = encoders
        self.trunk = trunk
        self.vector_group = vector_group
        self.image_groups = image_groups
        self.num_obs = trunk.num_obs
        self.num_actions = trunk.num_actions

    def __call__(self, obs: Mapping[str, jax.Array], key: jax.Array = None) -> tuple[jax.Array, dict]:
        return self.trunk(_latent(obs, self.vector_group, self.image_groups, self.encoders), key=key)


class VisionCritic(BaseCritic):
    """MLP critic whose input is the state vector plus encoded images."""

    encoders: dict[str, CNNEncoder]
    trunk: BaseCritic
    vector_group: str = eqx.field(static=True)
    image_groups: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        encoders: dict[str, CNNEncoder],
        trunk: BaseCritic,
        vector_group: str,
        image_groups: tuple[str, ...],
    ):
        self.encoders = encoders
        self.trunk = trunk
        self.vector_group = vector_group
        self.image_groups = image_groups
        self.num_obs = trunk.num_obs

    def __call__(self, obs: Mapping[str, jax.Array]) -> tuple[jax.Array, dict]:
        return self.trunk(_latent(obs, self.vector_group, self.image_groups, self.encoders))
