"""Newton terrain importer.

Turns a sim-agnostic :class:`~rlworld.rl.configs.scene.terrain_config.TerrainCfg`
into Newton collision geometry. This is the per-backend "import" step.

A generated terrain is added as a single static (``body=-1``) heightfield
— the canonical terrain is a height grid, which Newton supports natively
via :class:`newton.Heightfield`. The shape is labelled ``"ground_plane"``
so ground contact sensors (which match the geom ``"ground_plane"``) work
whether the ground is a flat plane or a heightfield.

Newton terrain runs the solver with ``use_mujoco_contacts=False`` (set in
the preset): contacts come from ``model.collide()`` (MPR — stable for
deep penetration, honours the contact margin), avoiding mjwarp's EPA
horizon issues on heightfield contacts.
"""

from __future__ import annotations

import newton

from rlworld.rl.configs.scene.terrain_config import TerrainCfg
from rlworld.rl.terrains import TerrainData, TerrainGenerator


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


def import_terrain_newton(builder: newton.ModelBuilder, cfg: TerrainCfg) -> TerrainData | None:
    """Add the terrain to ``builder``. Returns generated terrain data, or None for a flat plane."""
    if cfg.terrain_type == "plane":
        builder.add_ground_plane(cfg=_ground_shape_cfg(cfg))
        return None

    if cfg.terrain_type == "generator":
        if cfg.terrain_generator is None:
            raise ValueError("TerrainCfg(terrain_type='generator') requires a terrain_generator.")
        data = TerrainGenerator(cfg.terrain_generator).data
        hx, hy = data.half_extent
        # heights_m is in metres, so Newton's auto-derived min_z/max_z make
        # world-space z == heights_m exactly.
        heightfield = newton.Heightfield(
            data=data.heights_m,
            nrow=data.nrow,
            ncol=data.ncol,
            hx=hx,
            hy=hy,
        )
        builder.add_shape_heightfield(
            heightfield=heightfield,
            cfg=_ground_shape_cfg(cfg, margin=cfg.contact_margin),
            label="ground_plane",
        )
        return data

    raise ValueError(f"Unknown terrain_type: {cfg.terrain_type!r}")
