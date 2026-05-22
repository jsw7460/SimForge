"""MuJoCo (mjlab) terrain importer.

Turns a sim-agnostic :class:`~rlworld.rl.configs.scene.terrain_config.TerrainCfg`
into MuJoCo collision geometry. The canonical terrain is a height grid,
injected as a MuJoCo ``<hfield>`` — MuJoCo collides heightfields
non-convexly (unlike meshes, which it collapses to a convex hull).

The hfield + a body named ``"terrain"`` are added to the ``MjSpec`` via a
``spec_fn`` hook (mjlab runs it before compile). With no mjlab terrain
entity, ``Scene.env_origins`` falls back to all-zeros, so every env spawns
at the origin on this single patch (matching the Newton backend); an
out-of-bounds termination resets robots before they reach the edge.

The ``"terrain"`` body name matches the ground contact sensors.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from rlworld.rl.configs.scene.terrain_config import TerrainCfg
from rlworld.rl.terrains import TerrainData, TerrainGenerator


def build_mujoco_terrain(terrain_cfg: TerrainCfg) -> tuple[Callable[[mujoco.MjSpec], None], TerrainData]:
    """Build a ``spec_fn`` that injects the generated terrain as an hfield.

    Returns ``(spec_fn, terrain_data)``. Only valid for
    ``terrain_type="generator"`` (the caller handles ``"plane"`` via
    mjlab's built-in plane).
    """
    if terrain_cfg.terrain_generator is None:
        raise ValueError("TerrainCfg(terrain_type='generator') requires a terrain_generator.")

    data = TerrainGenerator(terrain_cfg.terrain_generator).data
    heights = np.asarray(data.heights_m, dtype=np.float64)
    z_min = float(heights.min())
    z_max = float(heights.max())
    elevation = max(z_max - z_min, 1e-4)
    # MuJoCo hfield userdata is normalised to [0, 1]; world z = pos_z +
    # userdata * elevation. Placing the geom at z = z_min makes world z
    # equal the canonical heights exactly.
    userdata = ((heights - z_min) / elevation).flatten().tolist()
    hx, hy = data.half_extent
    nrow, ncol = data.nrow, data.ncol
    friction = float(terrain_cfg.friction)

    def _spec_fn(spec: mujoco.MjSpec) -> None:
        spec.add_hfield(
            name="terrain_hf",
            size=[hx, hy, elevation, 0.5],  # [radius_x, radius_y, elevation, base_thickness]
            nrow=nrow,
            ncol=ncol,
            userdata=userdata,
        )
        body = spec.worldbody.add_body(name="terrain")
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname="terrain_hf",
            pos=[0.0, 0.0, z_min],
            friction=[friction, 0.005, 0.0001],
        )

    return _spec_fn, data
