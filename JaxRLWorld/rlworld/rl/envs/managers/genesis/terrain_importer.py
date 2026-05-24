"""Genesis :class:`TerrainImporter` subclass.

Adds either a flat ground URDF or a ``gs.morphs.Terrain`` (heightfield)
to the Genesis scene. Genesis stores the height field in RAW units
(``value * vertical_scale = metres``), so the canonical metre grid is
divided by ``vertical_scale``. The patch is placed so it is centred on
the world origin, matching the other backends.
"""

from __future__ import annotations

import genesis as gs

from rlworld.rl.terrains import TerrainImporter


class GenesisTerrainImporter(TerrainImporter):
    """TerrainImporter that adds a Genesis terrain / plane entity."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entity = None
        """Genesis ``RigidEntity`` for the ground (set by :meth:`add_to_scene`).

        Exposed so the Genesis contact sensor can resolve the
        ``entity="terrain"`` sentinel against its link range without the
        ground living in ``scene_manager.entities`` (which is reserved
        for robot/articulated entities)."""

    def add_to_scene(self, scene: gs.Scene):
        """Add the terrain to ``scene``; stash the entity for later lookup."""
        if self.cfg.terrain_type == "plane":
            morph = gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
            self.entity = scene.add_entity(morph=morph)
            return self.entity

        if self.cfg.terrain_type == "generator":
            data = self._run_generator()
            lx, ly = data.size_xy
            # No material arg → Genesis defaults to gs.materials.Rigid().
            morph = gs.morphs.Terrain(
                height_field=data.heights_m / data.vertical_scale,  # metres → raw units
                horizontal_scale=data.horizontal_scale,
                vertical_scale=data.vertical_scale,
                pos=(-lx / 2.0, -ly / 2.0, 0.0),  # centre the patch on the origin
            )
            self.entity = scene.add_entity(morph=morph)
            self.configure_env_origins(origins=data.origins)
            return self.entity

        raise ValueError(f"Unknown terrain_type: {self.cfg.terrain_type!r}")
