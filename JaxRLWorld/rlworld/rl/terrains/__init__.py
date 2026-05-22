"""Backend-agnostic height-field terrain generation.

A single terrain definition (:class:`TerrainGeneratorCfg`) is turned into
one canonical :class:`TerrainData` — a height grid in metres + spawn
origins. Per-backend terrain importers feed that grid to each simulator's
native heightfield API, so the terrain is identical (and collides
correctly, non-convexly) across Genesis / Newton / MuJoCo.

A heightfield — not a triangle mesh — is the cross-sim canonical form on
purpose: MuJoCo collides meshes as convex hulls (a rough mesh would
collapse to a flat lid), whereas all three simulators support native
non-convex heightfield collision.

Layering:  SubTerrainCfg.function → TerrainGenerator → TerrainData(heights, origins) → <backend importer>
"""

from __future__ import annotations

from .hf_terrains import HfRandomUniformTerrainCfg, random_uniform_terrain
from .sub_terrain_cfg import SubTerrainCfg
from .terrain_generator import TerrainData, TerrainGenerator, TerrainGeneratorCfg

# ── Preset terrain configs ───────────────────────────────────────────

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    border_width=0.0,
    seed=0,
    sub_terrains={
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.04, 0.04),
            noise_step=0.02,
            downsampled_scale=0.2,
        ),
    },
)
"""Low-amplitude random-uniform rough ground — the blind-walking baseline."""


__all__ = [
    "SubTerrainCfg",
    "HfRandomUniformTerrainCfg",
    "random_uniform_terrain",
    "TerrainGenerator",
    "TerrainGeneratorCfg",
    "TerrainData",
    "ROUGH_TERRAINS_CFG",
]
