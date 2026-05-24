"""Newton :class:`TerrainImporter` subclass.

Injects a flat plane or a generated heightfield into a Newton
``ModelBuilder`` via the simulator's native API. The shape is labelled
``"ground_plane"`` so ground contact sensors keep matching regardless of
which terrain type is selected.

Newton mesh / heightfield terrain requires the solver to run with
``use_mujoco_contacts=False`` (mjwarp self-collision over an hfield
overflows its hardcoded EPA horizon and silently zeroes the contact
margin); the preset enforces that. Contacts come from ``model.collide()``
via Newton's MPR pipeline, which the scene manager's step loop already
feeds to ``solver.step()``.
"""

from __future__ import annotations

import newton

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
            # heights_m are metres; auto-derived min_z / max_z make world z
            # equal heights_m exactly.
            hfield = newton.Heightfield(
                data=data.heights_m,
                nrow=data.nrow,
                ncol=data.ncol,
                hx=hx,
                hy=hy,
            )
            builder.add_shape_heightfield(
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
