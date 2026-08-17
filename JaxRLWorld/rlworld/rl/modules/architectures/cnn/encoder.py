"""Convolutional encoder for image observation groups.

A port of ``rsl_rl.modules.CNN`` and mjlab's ``SpatialSoftmaxCNN``, kept
faithful down to the padding arithmetic and the coordinate grid, so a
policy trained here and one trained there see the same architecture.

The encoder is UNBATCHED, like every other actor in this package: it
takes ``(C, H, W)`` and the caller vmaps it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.modules.utils import get_activation

__all__ = ["SpatialSoftmax", "CNNEncoder", "compute_padding", "compute_output_dim"]


def compute_padding(input_hw: tuple[int, int], kernel: int, stride: int, dilation: int) -> tuple[int, int]:
    """Padding that keeps the output at ``ceil(input / stride)``.

    Verbatim from ``rsl_rl.modules.cnn._compute_padding``.
    """
    h = math.ceil((stride * math.floor(input_hw[0] / stride) - input_hw[0] - stride + dilation * (kernel - 1) + 1) / 2)
    w = math.ceil((stride * math.floor(input_hw[1] / stride) - input_hw[1] - stride + dilation * (kernel - 1) + 1) / 2)
    return (h, w)


def compute_output_dim(
    input_hw: tuple[int, int],
    kernel: int,
    stride: int,
    dilation: int,
    padding: tuple[int, int],
) -> tuple[int, int]:
    """Output height and width of one convolution.

    Verbatim from ``rsl_rl.modules.cnn._compute_output_dim`` (the
    max-pool branch is omitted along with max-pool support).
    """
    h = math.floor((input_hw[0] + 2 * padding[0] - dilation * (kernel - 1) - 1) / stride + 1)
    w = math.floor((input_hw[1] + 2 * padding[1] - dilation * (kernel - 1) - 1) / stride + 1)
    return (h, w)


def _param(value: int | Sequence[int], idx: int) -> int:
    """One layer's value from either a per-layer sequence or a shared scalar."""
    if isinstance(value, int):
        return value
    return value[idx]


class SpatialSoftmax(eqx.Module):
    """Spatial soft-argmax over feature maps.

    Turns ``(C, H, W)`` activations into ``(C * 2,)`` coordinates: each
    channel's map is softmaxed over its pixels and reduced to the
    expected position, in [-1, 1]. The policy therefore receives WHERE
    each feature fired rather than how strongly — the representation
    that makes a camera policy about geometry, and the reason mjlab uses
    it in place of flattening or global pooling.
    """

    pos_h: jax.Array
    pos_w: jax.Array
    temperature: float = eqx.field(static=True)

    def __init__(self, height: int, width: int, temperature: float = 1.0):
        pos_h, pos_w = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, height),
            jnp.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.pos_h = pos_h.reshape(1, -1)
        self.pos_w = pos_w.reshape(1, -1)
        self.temperature = temperature

    def __call__(self, x: jax.Array) -> jax.Array:
        """``(C, H, W)`` -> ``(C * 2,)``, interleaved as (h, w) per channel."""
        channels = x.shape[0]
        features = x.reshape(channels, -1)
        weights = jax.nn.softmax(features / self.temperature, axis=-1)
        expected_h = (weights * self.pos_h).sum(axis=-1)
        expected_w = (weights * self.pos_w).sum(axis=-1)
        return jnp.stack([expected_h, expected_w], axis=-1).reshape(channels * 2)


class CNNEncoder(eqx.Module):
    """Convolution stack that turns one image group into a flat latent.

    With ``spatial_softmax`` the latent is ``2`` numbers per output
    channel; without it the final feature map is flattened, which is
    ``rsl_rl.modules.CNN``'s own ``flatten=True`` behaviour.
    """

    convs: tuple[eqx.nn.Conv2d, ...]
    spatial_softmax: SpatialSoftmax | None
    activation_name: str = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_hw: tuple[int, int],
        input_channels: int,
        output_channels: Sequence[int],
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        dilation: int | Sequence[int] = 1,
        padding: str = "zeros",
        activation: str = "elu",
        spatial_softmax: bool = True,
        spatial_softmax_temperature: float = 1.0,
        *,
        key: jax.Array,
    ):
        if padding not in ("zeros", "none"):
            raise ValueError(f"padding must be 'zeros' or 'none', got {padding!r}.")

        keys = jax.random.split(key, len(output_channels))
        convs = []
        last_channels = input_channels
        last_hw = input_hw
        for idx, channels in enumerate(output_channels):
            kernel = _param(kernel_size, idx)
            stride_i = _param(stride, idx)
            dilation_i = _param(dilation, idx)
            pad = compute_padding(last_hw, kernel, stride_i, dilation_i) if padding == "zeros" else (0, 0)
            convs.append(
                eqx.nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    stride=stride_i,
                    padding=pad,
                    dilation=dilation_i,
                    key=keys[idx],
                )
            )
            last_channels = channels
            last_hw = compute_output_dim(last_hw, kernel, stride_i, dilation_i, pad)

        if min(last_hw) < 1:
            raise ValueError(
                f"The convolution stack reduces a {input_hw} image to {last_hw}, which has no pixels left. "
                "Use fewer layers, a smaller stride, or a larger image."
            )

        self.convs = tuple(convs)
        self.activation_name = activation
        if spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(last_hw[0], last_hw[1], spatial_softmax_temperature)
            self.output_dim = last_channels * 2
        else:
            self.spatial_softmax = None
            self.output_dim = last_channels * last_hw[0] * last_hw[1]

    def __call__(self, image: jax.Array) -> jax.Array:
        """``(C, H, W)`` -> ``(output_dim,)``."""
        activation = get_activation(self.activation_name)
        x = image
        for conv in self.convs:
            x = activation(conv(x))
        if self.spatial_softmax is not None:
            return self.spatial_softmax(x)
        return x.reshape(-1)
