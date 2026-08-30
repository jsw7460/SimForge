"""Engine-level contact visualization for the mjwarp-backed viewers.

Sensor-free: instead of reading a ContactSensorCfg group, this copies
the displayed env's ``qpos / qvel / ctrl`` into a private CPU
``MjData``, runs ``mj_forward`` there, and asks MuJoCo's own
visualization pipeline (``mjv_updateScene`` with CONTACTPOINT) to
generate the contact discs — the mechanism mjviser uses. Every contact
in the scene shows up (robot-ground, robot-object, object-object) at
the true contact points.

The force arrows are computed here via ``mj_contactForce`` instead of
mjv's arrow decor, for two reasons measured on the real scenes: mjv's
arrow length is normalized by per-model visual statistics
(``vis.map.force`` / ``stat``), so the same ~350 N ground reaction
rendered huge on the Newton template and nearly invisible on the mjlab
scene; and mjv's arrow direction follows the contact's geom ordering,
which differs between the two templates (force into the ground on one,
out of it on the other). Drawing them ourselves gives every backend
the same metres-per-newton slider and the same convention: the force
acting on the NON-STATIC geom, so a ground reaction always points out
of the ground.

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
_SHAFT_RADIUS = 0.006
_HEAD_RADIUS = 0.015
_MAX_LENGTH_M = 1.0
_FORCE_COLOR = (220, 55, 45)


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
            self._force_scale = server.gui.add_slider(
                "Force scale (m/N)", min=0.0005, max=0.02, step=0.0005, initial_value=0.002
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
        mujoco.mj_forward(m, d)

        self._opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._show_points.value
        mujoco.mjv_updateScene(m, d, self._opt, None, self._cam, int(mujoco.mjtCatBit.mjCAT_DECOR), self._scn)

        cylinders: list = []
        for i in range(self._scn.ngeom):
            g = self._scn.geoms[i]
            if int(g.type) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                cylinders.append(g)

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

        # Force arrows from mj_contactForce, oriented onto the non-static
        # geom (a ground reaction points out of the ground on every model,
        # regardless of the contact's geom ordering).
        if self._show_forces.value and d.ncon:
            n = int(d.ncon)
            pos = np.empty((n, 3), np.float32)
            vec = np.empty((n, 3), np.float32)
            f6 = np.zeros(6, dtype=np.float64)
            geom_bodyid = m.geom_bodyid
            for i in range(n):
                c = d.contact[i]
                mujoco.mj_contactForce(m, d, i, f6)
                # Contact-frame rows are the world-frame axes; f6[:3] is the
                # force on geom2 in that frame (normal component >= 0).
                f_world = np.asarray(c.frame).reshape(3, 3).T @ f6[:3]
                body1 = int(geom_bodyid[int(c.geom[0])])
                body2 = int(geom_bodyid[int(c.geom[1])])
                if body2 == 0 and body1 != 0:
                    f_world = -f_world  # show the force on geom1 instead
                pos[i] = np.asarray(c.pos) + offset
                vec[i] = f_world.astype(np.float32)

            mags = np.linalg.norm(vec, axis=1)
            dirs = np.where(mags[:, None] > 1e-9, vec / np.clip(mags[:, None], 1e-9, None), [[0.0, 0.0, 1.0]])
            quats = np.empty((n, 4), np.float32)
            z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            dots = dirs @ z
            crosses = np.cross(np.broadcast_to(z, dirs.shape), dirs)
            quats[:, 0] = 1.0 + dots
            quats[:, 1:] = crosses
            flipped = quats[:, 0] < 1e-8
            quats[flipped] = np.array([0.0, 1.0, 0.0, 0.0])
            quats /= np.linalg.norm(quats, axis=1, keepdims=True)

            lengths = np.clip(mags * float(self._force_scale.value), 0.0, _MAX_LENGTH_M)
            shaft_len = lengths * _SHAFT_RATIO
            s_scale = np.stack(
                [np.full(n, _SHAFT_RADIUS, np.float32), np.full(n, _SHAFT_RADIUS, np.float32), shaft_len], axis=1
            )
            h_scale = np.stack(
                [np.full(n, _HEAD_RADIUS, np.float32), np.full(n, _HEAD_RADIUS, np.float32), lengths - shaft_len],
                axis=1,
            )
            color = np.tile(np.array(_FORCE_COLOR, dtype=np.uint8), (n, 1))
            self._draw("arrow_shaft", "shaft", pos, quats, s_scale, color)
            self._draw("arrow_head", "head", pos + dirs * shaft_len[:, None], quats, h_scale, color)
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
