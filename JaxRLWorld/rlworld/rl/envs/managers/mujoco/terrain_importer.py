"""MuJoCo (mjlab) :class:`TerrainImporter` subclass.

Builds a ``spec_fn`` closure that injects the generated terrain as a
``<hfield>`` geom in a ``"terrain"`` body. For ``terrain_type="plane"``
the importer is a no-op — the scene manager falls back to mjlab's own
``TerrainEntityCfg("plane")`` for the flat ground.

The canonical height grid is normalised to ``[0, 1]`` for MuJoCo's
``userdata``; the geom is offset by ``z_min`` so the rendered surface and
the physics surface both span the canonical ``heights_m`` exactly.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from rlworld.rl.terrains import TerrainImporter


class MujocoTerrainImporter(TerrainImporter):
    """TerrainImporter that hands mjlab a spec_fn injecting an hfield."""

    def build_spec_fn(self) -> Callable[[mujoco.MjSpec], None] | None:
        """Return a spec_fn for mjlab, or ``None`` to use mjlab's flat plane."""
        if self.cfg.terrain_type == "plane":
            # Scene manager will use TerrainEntityCfg("plane") instead.
            return None

        if self.cfg.terrain_type == "generator":
            data = self._run_generator()
            self.configure_env_origins(origins=data.origins)

            heights = np.asarray(data.heights_m, dtype=np.float64)
            z_min = float(heights.min())
            z_max = float(heights.max())
            elevation = max(z_max - z_min, 1e-4)
            userdata = ((heights - z_min) / elevation).flatten().tolist()
            hx, hy = data.half_extent
            nrow, ncol = data.nrow, data.ncol
            friction = float(self.cfg.friction)

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

            return _spec_fn

        raise ValueError(f"Unknown terrain_type: {self.cfg.terrain_type!r}")
