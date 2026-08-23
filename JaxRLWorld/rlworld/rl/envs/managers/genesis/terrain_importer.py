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
        # Genesis combines pair friction as max(mu_a, mu_b) and has no
        # geom-priority concept (the MJCF ``priority`` attribute is
        # ignored by its parser). Our robot assets rely on MuJoCo's
        # priority rule — the foot geom's (possibly DR'd) friction
        # applies exclusively against the ground — so a default-friction
        # (1.0) ground masks any robot-side friction below 1.0 and
        # silently disables foot-friction DR. Anchoring the ground at the
        # minimum allowed friction makes max() always resolve to the
        # robot side, matching the priority semantics of the other
        # backends.
        # Torsional and rolling are anchored for the same reason and by
        # the same max() rule (`collider/contact.py:399`). They default to
        # 0.005 and 0.0001 on a Genesis material, and MuJoCo's ground
        # contributes NOTHING to a foot contact -- the foot's priority
        # wins the whole friction triple -- so a non-zero ground would
        # hand spin and roll resistance to geoms MuJoCo leaves free,
        # including every condim=1 body geom. Inert until a scene turns
        # the coefficients on via RigidOptions, so this changes nothing
        # for a preset that does not.
        ground_material = gs.materials.Rigid(friction=0.01, friction_torsional=0.0, friction_rolling=0.0)

        if self.cfg.terrain_type == "plane":
            # Genesis primitive morphs default to contype/conaffinity
            # 0xFFFF, which makes the plane collide with EVERY robot geom
            # regardless of the geom's own mask. MuJoCo grounds carry
            # contype=1/conaffinity=1, so robot geoms whose masks exclude
            # bit 0 (e.g. foot boxes reserved for foot-to-foot collision)
            # never touch the ground there; mirror that mask so Genesis
            # agrees. Terrain morphs need no override — Genesis hardcodes
            # terrain geoms to contype=1/conaffinity=1.
            morph = gs.morphs.Plane(contype=1, conaffinity=1)
            self.entity = scene.add_entity(morph=morph, material=ground_material)
            return self.entity

        if self.cfg.terrain_type == "generator":
            data = self._run_generator()
            lx, ly = data.size_xy
            morph = gs.morphs.Terrain(
                height_field=data.heights_m / data.vertical_scale,  # metres → raw units
                horizontal_scale=data.horizontal_scale,
                vertical_scale=data.vertical_scale,
                pos=(-lx / 2.0, -ly / 2.0, 0.0),  # centre the patch on the origin
            )
            self.entity = scene.add_entity(morph=morph, material=ground_material)
            self.configure_env_origins(origins=data.origins)
            return self.entity

        raise ValueError(f"Unknown terrain_type: {self.cfg.terrain_type!r}")
