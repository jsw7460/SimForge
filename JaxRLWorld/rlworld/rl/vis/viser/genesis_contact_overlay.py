"""Engine-level contact visualization for the Genesis viewer.

The Genesis counterpart of :class:`MjvContactOverlay`: sensor-free,
reading the collider's own contact buffer
(``scene.rigid_solver.collider.get_contacts``) — every contact in the
scene, at the true contact points, with the solver's forces. Genesis
has no MuJoCo decor pipeline, so the markers are drawn directly:
a small sphere per contact point and a batched arrow along the contact
force (the raw ``force`` entry — the force applied to geom B).

``is_padded=True`` keeps the contact axis at a fixed capacity with a
per-env live count, so the batched viser handles update in place
instead of being recreated as the live contact count changes.

Read-only on the simulation state; inert on the other backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import viser

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World

_POINT_COLOR = (255, 190, 60)
_FORCE_COLOR = (220, 55, 45)
_POINT_RADIUS = 0.012
_SHAFT_RADIUS = 0.006
_HEAD_RADIUS = 0.015
_SHAFT_RATIO = 0.8
_MAX_LENGTH_M = 1.0


def _quats_from_z(dirs: np.ndarray) -> np.ndarray:
    """Batched wxyz quaternions rotating +Z onto each row of ``dirs`` (unit)."""
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dots = dirs @ z
    crosses = np.cross(np.broadcast_to(z, dirs.shape), dirs)
    quats = np.concatenate([(1.0 + dots)[:, None], crosses], axis=1)
    flipped = quats[:, 0] < 1e-8
    quats[flipped] = np.array([0.0, 1.0, 0.0, 0.0])
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


class GenesisContactOverlay:
    """Genesis collider contacts (points + force arrows) in viser."""

    def __init__(self, server: viser.ViserServer, env: World):
        self._server = server
        self._env = env
        self._available = env.sim_type == "genesis"
        self._handles: dict[str, object] = {}
        if not self._available:
            return

        self._collider = env.scene_manager.scene.rigid_solver.collider
        robot = env.scene_manager[env.robot_entity_name]
        self._robot_geom_range = (int(robot.geom_start), int(robot.geom_end))

        with server.gui.add_folder("Engine contacts", expand_by_default=False):
            self._show_points = server.gui.add_checkbox("Contact points", initial_value=False)
            self._show_forces = server.gui.add_checkbox("Contact forces", initial_value=False)
            self._force_scale = server.gui.add_slider(
                "Force scale (m/N)", min=0.0005, max=0.02, step=0.0005, initial_value=0.002
            )

    def update(self, env_idx: int, scene_offset: np.ndarray) -> None:
        if not self._available:
            return
        if not (self._show_points.value or self._show_forces.value):
            self._hide_all()
            return

        # The raw collider dict carries "force" (the force on geom B; the
        # entity-level API derives force_a/force_b as -/+ of it) and, when
        # padded, the per-env live count "n_contacts" instead of a mask.
        data = self._collider.get_contacts(as_tensor=True, to_torch=True, is_padded=True)
        pos_t, force_t = data["position"], data["force"]
        if pos_t.ndim == 3:  # parallelized scene: (n_envs, capacity, 3)
            pos = pos_t[env_idx].detach().cpu().numpy().astype(np.float32)
            force = force_t[env_idx].detach().cpu().numpy().astype(np.float32)
            geom_a = data["geom_a"][env_idx].detach().cpu().numpy()
            geom_b = data["geom_b"][env_idx].detach().cpu().numpy()
            n_live = int(data["n_contacts"][env_idx].item())
        else:  # single non-parallel scene: (capacity, 3)
            pos = pos_t.detach().cpu().numpy().astype(np.float32)
            force = force_t.detach().cpu().numpy().astype(np.float32)
            geom_a = data["geom_a"].detach().cpu().numpy()
            geom_b = data["geom_b"].detach().cpu().numpy()
            n_live = int(np.asarray(data["n_contacts"]).reshape(-1)[0])
        valid = np.arange(pos.shape[0]) < n_live

        # Direction convention shared with the mjwarp overlay: show the
        # force acting on the ROBOT geom. The raw "force" is the force on
        # geom B (the entity API derives force_a as its negation), so flip
        # the rows where the robot is geom A — a ground reaction then
        # points out of the ground on every backend.
        lo, hi = self._robot_geom_range
        robot_a = (geom_a >= lo) & (geom_a < hi)
        robot_b = (geom_b >= lo) & (geom_b < hi)
        force = np.where((robot_a & ~robot_b)[:, None], -force, force)

        n = pos.shape[0]
        if n == 0:
            self._hide_all()
            return
        pos = pos + scene_offset.astype(np.float32)

        mags = np.linalg.norm(force, axis=1)
        active = valid & (mags > 1e-6)
        dirs = np.where(mags[:, None] > 1e-9, force / np.clip(mags[:, None], 1e-9, None), [[0.0, 0.0, 1.0]])
        quats = _quats_from_z(dirs)
        lengths = np.clip(mags * float(self._force_scale.value), 0.0, _MAX_LENGTH_M)
        lengths[~active] = 0.0

        if self._show_points.value:
            point_scales = np.full((n, 3), _POINT_RADIUS, np.float32)
            point_scales[~valid] = 0.0
            self._draw("points", "sphere", pos, np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)), point_scales, _POINT_COLOR, n)
        else:
            self._hide("points")

        if self._show_forces.value:
            shaft_len = lengths * _SHAFT_RATIO
            shaft_scales = np.stack(
                [np.full(n, _SHAFT_RADIUS, np.float32), np.full(n, _SHAFT_RADIUS, np.float32), shaft_len], axis=1
            )
            head_scales = np.stack(
                [np.full(n, _HEAD_RADIUS, np.float32), np.full(n, _HEAD_RADIUS, np.float32), lengths - shaft_len],
                axis=1,
            )
            shaft_scales[~active] = 0.0
            head_scales[~active] = 0.0
            head_pos = pos + dirs * shaft_len[:, None]
            self._draw("shafts", "shaft", pos, quats, shaft_scales, _FORCE_COLOR, n)
            self._draw("heads", "head", head_pos, quats, head_scales, _FORCE_COLOR, n)
        else:
            self._hide("shafts")
            self._hide("heads")

    # ── batched handle plumbing ──────────────────────────────────────

    _mesh_builders = {
        "sphere": lambda: trimesh.creation.icosphere(subdivisions=1, radius=1.0),
        "shaft": lambda: trimesh.creation.cylinder(radius=1.0, height=1.0).apply_translation([0.0, 0.0, 0.5]),
        "head": lambda: trimesh.creation.cone(radius=1.0, height=1.0),
    }

    def _draw(self, key: str, mesh_key: str, pos, quats, scales, color, n: int) -> None:
        colors = np.tile(np.array(color, dtype=np.uint8), (n, 1))
        handle = self._handles.get(key)
        if handle is not None:
            handle.visible = True
            handle.batched_positions = np.asarray(pos, np.float32)
            handle.batched_wxyzs = np.asarray(quats, np.float32)
            handle.batched_scales = np.asarray(scales, np.float32)
            handle.batched_colors = colors
            return
        mesh = self._mesh_builders[mesh_key]()
        self._handles[key] = self._server.scene.add_batched_meshes_simple(
            f"/overlay/engine_contacts/{key}",
            mesh.vertices,
            mesh.faces,
            batched_wxyzs=np.asarray(quats, np.float32),
            batched_positions=np.asarray(pos, np.float32),
            batched_scales=np.asarray(scales, np.float32),
            batched_colors=colors,
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
