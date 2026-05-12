"""Procedural, *tileable* ground texture for the Viser eval scene.

Pure numpy + PIL.  Run as a script to (re)bake ``assets/ground_texture.png``
(the bundled asset ``ViserScene`` uses by default); tweak the colors / octave
weights here and re-run if you want a different look.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image


def _tileable_fbm(size: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """A periodic (wrap-around) fractal-noise field in [0, 1].

    White noise → FFT → 1/f^beta radial spectrum → IFFT; the result is
    naturally tileable.  Larger ``beta`` → smoother / larger features.
    """
    spectrum = np.fft.fft2(rng.standard_normal((size, size)))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    shaped = spectrum / (radius ** (beta / 2.0))
    shaped[0, 0] = 0.0  # zero mean
    field = np.fft.ifft2(shaped).real
    field -= field.min()
    field /= max(field.max(), 1e-9)
    return field


def generate_ground_texture(size: int = 512, seed: int = 7) -> Image.Image:
    """Earthy ground (dark soil → dry tan) with grain, grit and faint cracks — tileable."""
    rng = np.random.default_rng(seed)

    # Multi-octave height field: a few big patches + medium + fine grain.
    h = (
        0.62 * _tileable_fbm(size, beta=3.8, rng=rng)
        + 0.28 * _tileable_fbm(size, beta=2.6, rng=rng)
        + 0.16 * _tileable_fbm(size, beta=1.6, rng=rng)
        + 0.08 * _tileable_fbm(size, beta=0.8, rng=rng)
    )
    h -= h.min()
    h /= max(h.max(), 1e-9)
    h = np.clip((h - 0.5) * 1.35 + 0.5, 0.0, 1.0)  # punch up the contrast

    dark = np.array([74, 65, 53], dtype=np.float64)  # damp soil
    light = np.array([176, 160, 127], dtype=np.float64)  # dry tan
    rgb = dark[None, None, :] * (1.0 - h[..., None]) + light[None, None, :] * h[..., None]

    # Coarse "pebbles/clods": a medium-freq field pushed to high contrast → scattered
    # lighter/darker blobs that survive being viewed from far away.
    clods = _tileable_fbm(size, beta=1.3, rng=rng)
    clods = np.clip((clods - 0.5) * 4.5, -1.0, 1.0)
    rgb += clods[..., None] * 26.0

    # Per-pixel grit (i.i.d. → tileable) — fine soil grain up close.
    rgb += (rng.random((size, size, 1)) - 0.5) * 32.0

    # A network of darker cracks: ridges near the 0.5 level set of a mid-freq field.
    cracks = _tileable_fbm(size, beta=2.1, rng=rng)
    crack_mask = np.exp(-((cracks - 0.5) ** 2) / (2 * 0.022**2))
    rgb -= crack_mask[..., None] * 42.0

    # A whisper of mossy green in the lowest, dampest spots.
    moss = np.clip(0.45 - h, 0.0, 0.45) / 0.45
    rgb[..., 1] += moss * 10.0
    rgb[..., 0] -= moss * 4.0

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ground_texture.png")


def default_texture_path() -> str:
    """Path to the bundled ``ground_texture.png``."""
    return _DEFAULT_PATH


if __name__ == "__main__":
    os.makedirs(os.path.dirname(_DEFAULT_PATH), exist_ok=True)
    generate_ground_texture().save(_DEFAULT_PATH)
    print(f"wrote {_DEFAULT_PATH}")
