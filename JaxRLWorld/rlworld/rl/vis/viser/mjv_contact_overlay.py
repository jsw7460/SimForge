"""Engine-level contact visualization for the mjwarp-backed viewers.

Sensor-free: instead of reading a ContactSensorCfg group, this copies
the displayed env's ``qpos / qvel / ctrl`` into a private CPU
``MjData``, runs ``mj_forward`` there, and asks MuJoCo's own
visualization pipeline (``mjv_updateScene`` with the CONTACTPOINT /
CONTACTFORCE flags) to generate the contact discs and force arrows —
the exact mechanism mjviser uses. Every contact in the scene shows up
(robot-ground, robot-object, object-object), at the true contact
points, with MuJoCo's force decomposition.

Works on both mjwarp cells: the mujoco backend's scene ``mj_model`` and
Newton's ``SolverMuJoCo.mj_model`` are each the CPU template of the
same physics the GPU steps. Genesis has no MuJoCo template; the overlay
simply does not appear there (the sensor-based ``ContactForceOverlay``
still does).

The GPU simulation is never written to — the CPU MjData is a private
copy, so training dynamics are untouched. The forces shown are the CPU
solver's re-interpretation of the displayed state: visually faithful,
not bitwise equal to the GPU solver's.

Rendering: batched unit meshes (disc / arrow shaft / arrow head)
updated in place per frame, mirroring mjviser's decor renderer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import viser
import viser.transforms as vtf

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World

_MAXGEOM = 2000
_SHAFT_RATIO = 0.8  # mjviser / mjlab convention: 80% shaft, 20% head


def _unit_meshes() -> dict[str, trimesh.Trimesh]:
    disc = trimesh.creation.cylinder(radius=1.0, height=1.0)  # centered at origin
    shaft = trimesh.creation.cylinder(radius=1.0, height=1.0)
    shaft.apply_translation(np.array([0.0, 0.0, 0.5]))  # base-anchored, along +Z
    head = trimesh.creation.cone(radius=1.0, height=1.0)  # base at z=0
    return {"disc": disc, "shaft": shaft, "head": head}


class MjvContactOverlay:
    """MuJoCo-native contact decor (points + force arrows) in viser."""

    def __init__(self, server: viser.ViserServer, env: World):
        self._server = server
        self._env = env
        self._available = env.sim_type in ("mujoco", "newton")
        self._handles: dict[str, object] = {}
        if not self._available:
            return

        # Backend-specific imports and model access live here on purpose:
        # this module is imported by the viewers on every backend, and a
        # genesis process must not touch mujoco/warp state it never uses.
        import copy

        import mujoco

        if env.sim_type == "mujoco":
            sm = env.scene_manager
            base_model = sm.mj_model
            self._wp_data = sm.sim.wp_data
        else:
            solver = env.scene_manager.solver
            base_model = solver.mj_model
            self._wp_data = solver.mjw_data

        # Private copy: the overlay tweaks visual-map fields and zeroes
        # margins, and must never touch the backend's shared template.
        self._mj_model = copy.deepcopy(base_model)
        # Newton bakes its contact-DETECTION distance (default 0.1 m)
        # into the exported geom margin/gap. MuJoCo creates an inactive
        # zero-force contact slot for every pair inside that distance,
        # and mjv draws each one — dots all over a standing robot whose
        # links sit within 10 cm of each other. Zeroing margin/gap in
        # the private copy leaves only true penetrating contacts, the
        # same semantics the Genesis overlay shows.
        self._mj_model.geom_margin[:] = 0.0
        self._mj_model.geom_gap[:] = 0.0
        if self._mj_model.npair:
            self._mj_model.pair_margin[:] = 0.0
            self._mj_model.pair_gap[:] = 0.0

        self._mujoco = mujoco
        self._mj_data = mujoco.MjData(self._mj_model)
        self._scn = mujoco.MjvScene(self._mj_model, maxgeom=_MAXGEOM)
        self._opt = mujoco.MjvOption()
        self._cam = mujoco.MjvCamera()
        self._meshes = _unit_meshes()

        with server.gui.add_folder("Engine contacts", expand_by_default=False):
            self._show_points = server.gui.add_checkbox("Contact points", initial_value=False)
            self._show_forces = server.gui.add_checkbox("Contact forces", initial_value=False)
            # ``vis.map.force`` only feeds the decor generator; the model's
            # physics side was already consumed by put_model, so writing a
            # visual-map field cannot reach the GPU simulation.
            self._force_scale = server.gui.add_slider(
                "Force scale", min=0.001, max=0.1, step=0.001, initial_value=float(self._mj_model.vis.map.force)
            )

    # ── per-frame update ─────────────────────────────────────────────

    def update(self, env_idx: int, scene_offset: np.ndarray) -> None:
        if not self._available:
            return
        if not (self._show_points.value or self._show_forces.value):
            self._hide_all()
            return

        import warp as wp

        mujoco = self._mujoco
        m, d = self._mj_model, self._mj_data

        d.qpos[:] = wp.to_torch(self._wp_data.qpos)[env_idx].detach().cpu().numpy()
        d.qvel[:] = wp.to_torch(self._wp_data.qvel)[env_idx].detach().cpu().numpy()
        if d.ctrl.size:
            d.ctrl[:] = wp.to_torch(self._wp_data.ctrl)[env_idx].detach().cpu().numpy()
        m.vis.map.force = float(self._force_scale.value)
        mujoco.mj_forward(m, d)

        self._opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._show_points.value
        self._opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._show_forces.value
        mujoco.mjv_updateScene(m, d, self._opt, None, self._cam, int(mujoco.mjtCatBit.mjCAT_DECOR), self._scn)

        cylinders: list = []
        arrows: list = []
        arrow_types = {
            int(mujoco.mjtGeom.mjGEOM_ARROW),
            int(mujoco.mjtGeom.mjGEOM_ARROW1),
            int(mujoco.mjtGeom.mjGEOM_ARROW2),
        }
        for i in range(self._scn.ngeom):
            g = self._scn.geoms[i]
            gtype = int(g.type)
            if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                cylinders.append(g)
            elif gtype in arrow_types:
                arrows.append(g)

        offset = scene_offset.astype(np.float32)

        # Contact discs.
        if cylinders:
            n = len(cylinders)
            pos = np.empty((n, 3), np.float32)
            quat = np.empty((n, 4), np.float32)
            scale = np.empty((n, 3), np.float32)
            color = np.empty((n, 3), np.uint8)
            for j, g in enumerate(cylinders):
                pos[j] = np.asarray(g.pos) + offset
                quat[j] = vtf.SO3.from_matrix(np.asarray(g.mat).reshape(3, 3)).wxyz
                size = np.asarray(g.size)
                scale[j] = [size[0], size[0], max(size[2] * 2.0, 1e-4)]
                color[j] = (np.clip(np.asarray(g.rgba)[:3], 0, 1) * 255).astype(np.uint8)
            self._draw("disc", "disc", pos, quat, scale, color)
        else:
            self._hide("disc")

        # Force arrows: shaft + head, split mjviser-style.
        if arrows:
            n = len(arrows)
            s_pos = np.empty((n, 3), np.float32)
            quat = np.empty((n, 4), np.float32)
            s_scale = np.empty((n, 3), np.float32)
            h_pos = np.empty((n, 3), np.float32)
            h_scale = np.empty((n, 3), np.float32)
            color = np.empty((n, 3), np.uint8)
            for j, g in enumerate(arrows):
                mat = np.asarray(g.mat).reshape(3, 3)
                size = np.asarray(g.size)  # [shaft_radius, head_radius, total_length]
                total = size[2]
                shaft_len = total * _SHAFT_RATIO
                s_pos[j] = np.asarray(g.pos) + offset
                quat[j] = vtf.SO3.from_matrix(mat).wxyz
                s_scale[j] = [size[0], size[0], max(shaft_len, 1e-4)]
                h_pos[j] = s_pos[j] + mat @ np.array([0.0, 0.0, shaft_len])
                h_scale[j] = [size[1], size[1], max(total - shaft_len, 1e-4)]
                color[j] = (np.clip(np.asarray(g.rgba)[:3], 0, 1) * 255).astype(np.uint8)
            self._draw("arrow_shaft", "shaft", s_pos, quat, s_scale, color)
            self._draw("arrow_head", "head", h_pos, quat, h_scale, color)
        else:
            self._hide("arrow_shaft")
            self._hide("arrow_head")

    # ── batched handle plumbing ──────────────────────────────────────

    def _draw(self, key: str, mesh_key: str, pos, quat, scale, color) -> None:
        handle = self._handles.get(key)
        if handle is not None:
            handle.visible = True
            handle.batched_positions = pos
            handle.batched_wxyzs = quat
            handle.batched_scales = scale
            handle.batched_colors = color
            return
        mesh = self._meshes[mesh_key]
        self._handles[key] = self._server.scene.add_batched_meshes_simple(
            f"/overlay/engine_contacts/{key}",
            mesh.vertices,
            mesh.faces,
            batched_wxyzs=quat,
            batched_positions=pos,
            batched_scales=scale,
            batched_colors=color,
            lod="off",
            cast_shadow=False,
            receive_shadow=False,
        )

    def _hide(self, key: str) -> None:
        handle = self._handles.get(key)
        if handle is not None:
            handle.visible = False

    def _hide_all(self) -> None:
        for key in self._handles:
            self._hide(key)
