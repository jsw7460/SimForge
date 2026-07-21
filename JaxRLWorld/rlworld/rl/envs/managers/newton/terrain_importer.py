"""Newton :class:`TerrainImporter` subclass.

Injects a flat plane or a generated heightfield into a Newton
``ModelBuilder`` via the simulator's native API. The shape is labelled
``"ground_plane"`` so ground contact sensors keep matching regardless of
which terrain type is selected.

The heightfield works under both contact backends: Newton's native
triangle extraction and the ``SolverMuJoCo`` hfield conversion
(``use_mujoco_contacts=True``) read the grid with the same MuJoCo
convention (rows span Y, columns span X), so the canonical (x, y)-indexed
grid is transposed once here for both.
"""

from __future__ import annotations

import newton
import warp as wp

from rlworld.rl.configs.scene.terrain_config import TerrainCfg
from rlworld.rl.terrains import TerrainImporter


def _ground_shape_cfg(cfg: TerrainCfg, margin: float = 0.0) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        ke=cfg.contact_stiffness,
        kd=cfg.contact_damping,
        mu=cfg.friction,
        kf=cfg.ground_kf,
        mu_rolling=cfg.ground_mu_rolling,
        mu_torsional=cfg.ground_mu_torsional,
        margin=margin,
    )


class NewtonTerrainImporter(TerrainImporter):
    """TerrainImporter that emits Newton collision shapes."""

    def import_into_builder(self, builder: newton.ModelBuilder) -> None:
        """Add the terrain to ``builder`` and configure env origins."""
        if self.cfg.terrain_type == "plane":
            builder.add_ground_plane(cfg=_ground_shape_cfg(self.cfg))
            # No sub-terrain grid → env_origins stays at all-zeros (default).
            return

        if self.cfg.terrain_type == "generator":
            data = self._run_generator()
            hx, hy = data.half_extent
            heights = data.heights_m
            z_min = float(heights.min())
            z_range = max(float(heights.max()) - z_min, 1e-4)
            # Axis conventions: the canonical grid is heights[ix, iy]
            # (dim0 = X), while Newton/MuJoCo hfields read data[iy, ix]
            # (rows span Y, columns span X) and define nrow as the number
            # of Y samples. Transposing maps one onto the other, which is
            # also why nrow/ncol below come from the opposite canonical
            # dimensions.
            num_x, num_y = data.nrow, data.ncol
            #
            # The vertical anchor must live in the shape transform, with the
            # heightfield itself anchored at min_z=0: SolverMuJoCo re-syncs
            # the hfield geom position from ``shape_transform`` whenever
            # shape properties change, which drops any min_z offset that the
            # model conversion baked into the spec-time geom position. With
            # min_z=0 the spec-time position, the re-synced position, and
            # Newton's native path all resolve to the same world surface.
            hfield = newton.Heightfield(
                data=heights.T,
                nrow=num_y,
                ncol=num_x,
                hx=hx,
                hy=hy,
                min_z=0.0,
                max_z=z_range,
            )
            builder.add_shape_heightfield(
                xform=wp.transform(wp.vec3(0.0, 0.0, z_min), wp.quat_identity()),
                heightfield=hfield,
                cfg=_ground_shape_cfg(self.cfg, margin=self.cfg.contact_margin),
                label="ground_plane",
            )
            # IsaacLab-style env_origins from the sub-terrain grid (no-op
            # for the v1 single-cell grid: every env still lands at the
            # one origin (0, 0, surface_z)).
            self.configure_env_origins(origins=data.origins)
            return

        raise ValueError(f"Unknown terrain_type: {self.cfg.terrain_type!r}")
