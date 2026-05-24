"""Translucent "ghost" overlay of a motion-tracking reference robot pose.

Draws the per-body reference pose of the env's active
:class:`MotionCommand` as a semi-transparent robot silhouette next to
the live robot in the viser scene — so the user can eyeball tracking
error (anchor drift, joint deviation) at a glance.

Cross-sim, single-robot ``mujoco.MjModel`` is the visual source of
truth. We pick the right access point per backend:

* **mjlab (mujoco sim)** — ``scene_manager.scene.compile()`` returns a
  fresh single-robot model from mjlab's spec (entity-prefixed body
  names, e.g. ``robot/pelvis``).
* **Newton** — ``solver.mj_model`` is num_envs-replicated and unusable
  for single-robot visuals, so we re-parse the configured
  ``entities["robot"].mjcf_path`` (bare body names).
* Anything else (e.g. Genesis without an mjcf_path) → ``RuntimeError``.

Per tracked body we merge that body's visual geoms (collision-only
geoms — those with nonzero ``contype``/``conaffinity`` — are dropped,
matching Mjlab) into one ``trimesh.Trimesh`` in body-local frame and
register it as a viser ``add_mesh_simple`` handle. Each viewer tick we
pull the body's world transform from ``MotionCommand.body_pos_w`` /
``body_quat_w`` and write it to the handle. The Mjlab analogue is
``mjlab.tasks.tracking.mdp.commands._debug_vis_impl`` +
``DebugVisualizer.add_ghost_mesh``; ours is sim-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import mujoco
import numpy as np
import trimesh
import viser

if TYPE_CHECKING:
    from rlworld.rl.envs.mdp.commands.motion import MotionCommand
    from rlworld.rl.envs.world import World


# Pale-cyan, fairly translucent — reads cleanly against both the live
# robot and the default viser background.
_GHOST_COLOR_RGB = (120, 200, 255)
_GHOST_OPACITY = 0.35


class MotionGhost:
    """Per-body ghost meshes + per-tick transform updates."""

    def __init__(
        self,
        server: viser.ViserServer,
        env: World,
        color: tuple[int, int, int] = _GHOST_COLOR_RGB,
        opacity: float = _GHOST_OPACITY,
    ) -> None:
        self._server = server
        self._env = env
        self._color = color
        self._opacity = opacity
        self._handles: dict[str, viser.MeshHandle] = {}

        cmd = self._motion_command()
        if cmd is None:
            return  # non-tracking preset — nothing to draw

        model = _get_robot_mj_model(env)
        meshes = _build_per_body_meshes(model, tuple(cmd.cfg.body_names))
        for name, mesh in meshes.items():
            if mesh is None or len(mesh.vertices) == 0:
                continue
            self._handles[name] = server.scene.add_mesh_simple(
                f"/motion_ghost/{name}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.int32),
                color=color,
                opacity=opacity,
                cast_shadow=False,
                receive_shadow=False,
            )

    # ── Public API ──────────────────────────────────────────────────

    def update(self, env_idx: int) -> None:
        """Pull the reference body transforms for ``env_idx`` from the
        MotionCommand and write them onto the viser handles. Cheap —
        only sets ``position`` / ``wxyz`` per handle."""
        cmd = self._motion_command()
        if cmd is None or not self._handles:
            return
        body_pos = cmd.body_pos_w[env_idx].detach().cpu().numpy()
        body_quat = cmd.body_quat_w[env_idx].detach().cpu().numpy()
        for i, name in enumerate(cmd.cfg.body_names):
            handle = self._handles.get(name)
            if handle is None:
                continue
            handle.position = tuple(body_pos[i].tolist())
            handle.wxyz = tuple(body_quat[i].tolist())

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


# ─────────────────────────────────────────────────────────────────────
# Model resolution + body / geom helpers
# ─────────────────────────────────────────────────────────────────────


def _get_robot_mj_model(env: World):
    """Return a single-robot ``mujoco.MjModel`` for ghost-mesh extraction.

    Branches by sim — see module docstring. Raises ``RuntimeError`` when
    neither path resolves; no silent fallback.
    """
    sm = env.scene_manager

    # mjlab: Scene → fresh compiled, single-robot MjModel.
    scene = getattr(sm, "scene", None)
    if scene is not None and hasattr(scene, "compile"):
        return scene.compile()

    # Newton / file-based: parse the configured MJCF.
    config = getattr(sm, "config", None)
    entities = getattr(config, "entities", None)
    if entities is not None and "robot" in entities:
        mjcf_path = getattr(entities["robot"], "mjcf_path", None)
        if mjcf_path:
            return mujoco.MjModel.from_xml_path(mjcf_path)

    sim_type = getattr(env, "sim_type", "<unknown>")
    raise RuntimeError(
        f"MotionGhost: cannot resolve a single-robot mujoco.MjModel "
        f"(sim_type={sim_type!r}). Tried `scene_manager.scene.compile()` "
        f"(mjlab) and `scene_manager.config.entities['robot'].mjcf_path` "
        f"(file-based); both unavailable. Backends without either "
        f"accessor need a dedicated single-robot extraction path."
    )


def _find_body_id_scoped(model, bare_name: str) -> int:
    """Resolve a tracked body's id by bare name, falling back to mjlab's
    scoped suffix convention (``<entity>/<bare>``, e.g. ``robot/pelvis``).

    Newton's file-parsed MJCF keeps bare names → first match wins;
    mjlab's compiled spec prepends an entity prefix → suffix match
    catches it. Raises ``RuntimeError`` when neither resolves, with a
    sample of actual body names so the mismatch is immediately legible.
    No silent skip.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bare_name)
    if bid >= 0:
        return bid
    target_suffix = "/" + bare_name
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and (name == bare_name or name.endswith(target_suffix)):
            return i
    sample = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(min(model.nbody, 8))]
    raise RuntimeError(
        f"MotionGhost: tracked body {bare_name!r} not found in mj_model "
        f"(tried exact + scoped suffix '/{bare_name}'). First "
        f"{len(sample)} body names: {sample}. MotionCommand.body_names "
        f"and the model disagree, or the model uses an unexpected scheme."
    )


def _build_per_body_meshes(
    model,
    body_names: tuple[str, ...],
) -> dict[str, trimesh.Trimesh | None]:
    """Merge each body's visual geoms into a single body-local
    ``trimesh.Trimesh``. Collision-only geoms (``contype != 0`` or
    ``conaffinity != 0``) are dropped, matching Mjlab's ghost filter.

    Returns ``None`` for a body when it has no visual geoms at all
    (legitimate — e.g. frame-only bodies); a missing body name raises
    via :func:`_find_body_id_scoped`.
    """
    out: dict[str, trimesh.Trimesh | None] = {}
    for bname in body_names:
        body_id = _find_body_id_scoped(model, bname)
        parts = [
            _geom_to_trimesh(model, gid)
            for gid in range(model.ngeom)
            if int(model.geom_bodyid[gid]) == body_id and _is_visual_geom(model, gid)
        ]
        parts = [p for p in parts if p is not None]
        if not parts:
            out[bname] = None
        elif len(parts) == 1:
            out[bname] = parts[0]
        else:
            out[bname] = trimesh.util.concatenate(parts)
    return out


def _is_visual_geom(model, gid: int) -> bool:
    """A geom is visual-only iff it doesn't participate in collisions."""
    return int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0


# Primitive geom → ``trimesh`` constructor (mesh handled separately).
_PRIMITIVE_BUILDERS: dict[int, Callable[[np.ndarray], trimesh.Trimesh]] = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): lambda size: trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2),
    int(mujoco.mjtGeom.mjGEOM_BOX): lambda size: trimesh.creation.box(extents=(2.0 * size[:3]).tolist()),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): lambda size: trimesh.creation.capsule(
        radius=float(size[0]), height=2.0 * float(size[1])
    ),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): lambda size: trimesh.creation.cylinder(
        radius=float(size[0]), height=2.0 * float(size[1])
    ),
}


def _geom_to_trimesh(model, gid: int) -> trimesh.Trimesh | None:
    """One MuJoCo geom → ``trimesh.Trimesh`` in *body-local* frame.

    The geom's local ``pos`` / ``quat`` are baked into the vertices so
    the caller only has to push the body's world transform to viser per
    tick. Returns ``None`` for geom types we don't draw (e.g. plane,
    hfield) so the merge step can skip them cleanly.
    """
    geom_type = int(model.geom_type[gid])
    size = np.asarray(model.geom_size[gid], dtype=np.float32)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = _extract_mesh_geom(model, gid)
    else:
        builder = _PRIMITIVE_BUILDERS.get(geom_type)
        if builder is None:
            return None
        mesh = builder(size)

    return _apply_local_transform(
        mesh,
        local_pos=np.asarray(model.geom_pos[gid], dtype=np.float32),
        local_quat=np.asarray(model.geom_quat[gid], dtype=np.float32),  # wxyz
    )


def _extract_mesh_geom(model, gid: int) -> trimesh.Trimesh:
    """Build a ``trimesh.Trimesh`` from the model's mesh table entry
    referenced by ``geom[gid]``."""
    mid = int(model.geom_dataid[gid])
    v0, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    f0, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    return trimesh.Trimesh(
        vertices=np.asarray(model.mesh_vert[v0 : v0 + vn], dtype=np.float32).copy(),
        faces=np.asarray(model.mesh_face[f0 : f0 + fn], dtype=np.int32).copy(),
        process=False,
    )


def _apply_local_transform(
    mesh: trimesh.Trimesh,
    local_pos: np.ndarray,
    local_quat: np.ndarray,
) -> trimesh.Trimesh:
    """Bake ``rotate(local_quat) ∘ translate(local_pos)`` into the mesh
    vertices. Done in fp64 because MuJoCo's mesh_vert table can include
    primitive-derived vert coords with large magnitudes that overflow an
    fp32 matmul; result is cast back to fp32 for viser."""
    rot = _quat_wxyz_to_mat(local_quat)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    mesh.vertices = ((rot @ verts.T).T + local_pos.astype(np.float64)).astype(np.float32)
    return mesh


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(c) for c in q)
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
