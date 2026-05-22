"""Genesis terrain importer.

Turns a sim-agnostic :class:`~rlworld.rl.configs.scene.terrain_config.TerrainCfg`
into a Genesis scene entity. The canonical terrain is a height grid, fed
to Genesis' native ``gs.morphs.Terrain`` (heightfield) — Genesis collides
a heightfield non-convexly, unlike a single mesh.

Genesis stores the height field in RAW units (value × vertical_scale =
metres), so the canonical metre grid is divided by ``vertical_scale``. The
patch is placed so it is centred on the world origin (matching the Newton
and MuJoCo backends — same canonical grid → identical terrain).
"""

from __future__ import annotations

import genesis as gs

from rlworld.rl.configs.scene.terrain_config import TerrainCfg
from rlworld.rl.terrains import TerrainGenerator


def import_terrain_genesis(scene: gs.Scene, cfg: TerrainCfg):
    """Add the terrain to ``scene``. Returns ``(entity, terrain_data)``.

    ``terrain_data`` is ``None`` for a flat plane.
    """
    if cfg.terrain_type == "plane":
        morph = gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        return scene.add_entity(morph=morph), None

    if cfg.terrain_type == "generator":
        if cfg.terrain_generator is None:
            raise ValueError("TerrainCfg(terrain_type='generator') requires a terrain_generator.")
        data = TerrainGenerator(cfg.terrain_generator).data
        lx, ly = data.size_xy
        # No material arg → Genesis defaults to gs.materials.Rigid(), which
        # is accepted for Terrain (same as the flat-plane URDF path).
        morph = gs.morphs.Terrain(
            height_field=data.heights_m / data.vertical_scale,  # metres → raw units
            horizontal_scale=data.horizontal_scale,
            vertical_scale=data.vertical_scale,
            pos=(-lx / 2.0, -ly / 2.0, 0.0),  # centre the patch on the origin
        )
        return scene.add_entity(morph=morph), data

    raise ValueError(f"Unknown terrain_type: {cfg.terrain_type!r}")
