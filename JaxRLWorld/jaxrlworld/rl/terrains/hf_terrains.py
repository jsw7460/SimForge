"""Height-field sub-terrains (backend-agnostic).

Each function returns a 2D height grid **in metres**; the generator tiles
them. Ported from IsaacLab's ``hf_terrains`` but returning metre grids
(not int16 vertical-scale units) and using numpy bilinear upsampling
instead of a scipy spline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .sub_terrain_cfg import SubTerrainCfg
from .utils import bilinear_upsample


def random_uniform_terrain(difficulty: float, cfg: HfRandomUniformTerrainCfg) -> np.ndarray:
    """Heights sampled uniformly from ``noise_range`` (metres).

    Coarse heights are sampled on a ``downsampled_scale`` grid then
    bilinearly upsampled to the full ``horizontal_scale`` resolution.
    ``difficulty`` is ignored (amplitude is fixed by ``noise_range``).
    """
    downsampled_scale = cfg.downsampled_scale if cfg.downsampled_scale is not None else cfg.horizontal_scale
    if downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            f"downsampled_scale ({downsampled_scale}) must be >= horizontal_scale ({cfg.horizontal_scale})."
        )

    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    width_ds = max(2, int(cfg.size[0] / downsampled_scale))
    length_ds = max(2, int(cfg.size[1] / downsampled_scale))

    # Discrete height levels in metres.
    levels = np.arange(cfg.noise_range[0], cfg.noise_range[1] + cfg.noise_step, cfg.noise_step)

    rng = np.random.default_rng(cfg.seed)
    coarse = rng.choice(levels, size=(width_ds, length_ds))
    return bilinear_upsample(coarse, (width_px, length_px)).astype(np.float32)


@dataclass
class HfRandomUniformTerrainCfg(SubTerrainCfg):
    function: Callable = random_uniform_terrain
    noise_range: tuple[float, float] = (-0.04, 0.04)
    noise_step: float = 0.02
    downsampled_scale: float | None = 0.2
