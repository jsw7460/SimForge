"""Translucent "ghost" overlay of a motion-tracking reference robot pose.

Renders the per-body reference pose of the env's active
:class:`MotionCommand` as a semi-transparent robot silhouette next to
the live robot in the viser scene — so the user can eyeball tracking
error (anchor drift, joint deviation) at a glance.

Cross-sim implementation: we read the robot meshes directly from the
preset's MJCF (``cfg.robot.mjcf_path``) via MuJoCo, build one merged
``trimesh.Trimesh`` per tracked body in body-local frame (geom-local
``pos``/``quat`` baked in), and add each as a single viser
``add_mesh_simple`` handle. Per viewer tick we pull the body's world
transform from ``MotionCommand.body_pos_w`` / ``body_quat_w`` and update
the handle's ``position``/``wxyz``. The env's actual sim (newton /
genesis / mujoco) never enters this path — the MJCF is the single
source of truth for visuals, so the ghost matches across all backends.

The Mjlab equivalent is ``mjlab.tasks.tracking.mdp.commands._debug_vis_impl``
+ ``DebugVisualizer.add_ghost_mesh`` — same intent, but Mjlab deep-copies
the running ``mj_model`` and zeros alpha on collision geoms, which is
only available on the MuJoCo backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import viser

if TYPE_CHECKING:
    from rlworld.rl.envs.mdp.commands.motion import MotionCommand
    from rlworld.rl.envs.world import World


# Pale-cyan ghost, fairly translucent. Tuned to read clearly against both
# the live robot and the default viser background.
_GHOST_COLOR_RGB = (120, 200, 255)
_GHOST_OPACITY = 0.35


class MotionGhost:
    """Owns the per-body ghost meshes + per-tick transform updates."""

    def __init__(
        self,
        server: viser.ViserServer,
        env: World,
        color: tuple[int, int, int] = _GHOST_COLOR_RGB,
        opacity: float = _GHOST_OPACITY,
    ) -> None:
        self._server = server
        self._env = env
        self._handles: dict[str, viser.MeshHandle] = {}
        self._color = color
        self._opacity = opacity

        cmd = self._motion_command()
        if cmd is None:
            # Non-tracking preset — nothing to draw.
            return

        mjcf_path = self._resolve_mjcf_path()
        body_names = tuple(cmd.cfg.body_names)
        meshes = _build_per_body_meshes(mjcf_path, body_names)

        for name, mesh in meshes.items():
            if mesh is None or len(mesh.vertices) == 0:
                continue
            handle = server.scene.add_mesh_simple(
                f"/motion_ghost/{name}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.int32),
                color=self._color,
                opacity=self._opacity,
                cast_shadow=False,
                receive_shadow=False,
            )
            self._handles[name] = handle

    # ── Public API ──────────────────────────────────────────────────

    def update(self, env_idx: int) -> None:
        """Pull current reference poses from the MotionCommand and write
        them onto the viser handles. Cheap — only sets position/wxyz."""
        cmd = self._motion_command()
        if cmd is None or not self._handles:
            return
        body_pos = cmd.body_pos_w[env_idx].detach().cpu().numpy()
        body_quat = cmd.body_quat_w[env_idx].detach().cpu().numpy()
        for i, name in enumerate(cmd.cfg.body_names):
            handle = self._handles.get(name)
            if handle is None:
                continue
            handle.position = (
                float(body_pos[i, 0]),
                float(body_pos[i, 1]),
                float(body_pos[i, 2]),
            )
            handle.wxyz = (
                float(body_quat[i, 0]),
                float(body_quat[i, 1]),
                float(body_quat[i, 2]),
                float(body_quat[i, 3]),
            )

    def set_visible(self, visible: bool) -> None:
        for handle in self._handles.values():
            handle.visible = visible

    def set_opacity(self, opacity: float) -> None:
        self._opacity = float(opacity)
        for handle in self._handles.values():
            handle.opacity = self._opacity

    @property
    def is_active(self) -> bool:
        return bool(self._handles)

    # ── Internal ────────────────────────────────────────────────────

    def _motion_command(self) -> MotionCommand | None:
        cm = self._env.command_manager
        terms = getattr(cm, "_terms", None)
        if not terms or "motion" not in terms:
            return None
        return cm.get_term("motion")

    def _resolve_mjcf_path(self) -> str:
        """Pull the robot MJCF path the env was built with."""
        entities = self._env.scene_manager.config.entities
        robot_entity = entities["robot"] if "robot" in entities else next(iter(entities.values()))
        return robot_entity.mjcf_path


# ─────────────────────────────────────────────────────────────────────
# MJCF parsing → per-body trimesh.Trimesh
# ─────────────────────────────────────────────────────────────────────


def _build_per_body_meshes(
    mjcf_path: str,
    body_names: tuple[str, ...],
) -> dict[str, trimesh.Trimesh | None]:
    """Parse the MJCF and return ``{body_name → merged trimesh in
    body-local frame}``. Skips collision-only geoms (``contype != 0`` or
    ``conaffinity != 0``) so only the visual silhouette remains."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    out: dict[str, trimesh.Trimesh | None] = {}
    for bname in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if body_id < 0:
            out[bname] = None
            continue

        parts: list[trimesh.Trimesh] = []
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != body_id:
                continue
            # Skip collision geoms (visual-only filter, matches Mjlab).
            if int(model.geom_contype[gid]) != 0 or int(model.geom_conaffinity[gid]) != 0:
                continue
            mesh = _geom_to_trimesh(model, gid)
            if mesh is None:
                continue
            parts.append(mesh)

        if not parts:
            out[bname] = None
        elif len(parts) == 1:
            out[bname] = parts[0]
        else:
            out[bname] = trimesh.util.concatenate(parts)
    return out


def _geom_to_trimesh(model, gid: int) -> trimesh.Trimesh | None:
    """One MuJoCo geom → trimesh in body-local frame.

    Bakes the geom's local ``pos``/``quat`` into the vertex coordinates so
    the caller only needs to push the *body's* world transform to viser
    per tick. Returns ``None`` for unsupported geom types (we cover the
    common visual shapes: mesh, sphere, box, capsule, cylinder)."""
    import mujoco

    geom_type = int(model.geom_type[gid])
    size = np.array(model.geom_size[gid], dtype=np.float32)
    local_pos = np.array(model.geom_pos[gid], dtype=np.float32)
    local_quat = np.array(model.geom_quat[gid], dtype=np.float32)  # wxyz

    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        v_adr = int(model.mesh_vertadr[mid])
        v_num = int(model.mesh_vertnum[mid])
        f_adr = int(model.mesh_faceadr[mid])
        f_num = int(model.mesh_facenum[mid])
        verts = np.asarray(model.mesh_vert[v_adr : v_adr + v_num], dtype=np.float32).copy()
        faces = np.asarray(model.mesh_face[f_adr : f_adr + f_num], dtype=np.int32).copy()
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        mesh = trimesh.creation.box(extents=(2.0 * size[:3]).tolist())
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
    else:
        return None

    # Do the transform in fp64 — MuJoCo's ``mesh_vert`` table can include
    # primitive-derived vertex coords with large magnitudes that overflow
    # an fp32 matmul. Promote to fp64 for the rotate-and-translate, cast
    # back to fp32 for viser.
    rot = _quat_wxyz_to_mat(local_quat)
    verts_64 = np.asarray(mesh.vertices, dtype=np.float64)
    mesh.vertices = ((rot @ verts_64.T).T + local_pos.astype(np.float64)).astype(np.float32)
    return mesh


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n > 0.0:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
