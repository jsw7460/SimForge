"""Per-body visual mesh extraction from a ``mujoco.MjModel`` — shared
by the mjlab and Newton ``SceneManager.get_visual_meshes`` paths (both
have an MjModel to hand off: mjlab via ``Scene.compile()``, Newton via
re-parsing the configured ``mjcf_path``).

Genesis does **not** use this module — it builds meshes directly from
its native ``RigidVisGeom`` data, so it has no mujoco dependency.
Pure trimesh transform math lives in :mod:`visual_mesh_transform` and
is used by both this module and the Genesis path.

Crash-early: a tracked body name that doesn't resolve raises
:class:`RuntimeError`. A body with zero visual geoms returns ``None``
(legitimate — e.g. frame-only bodies have no mesh to draw).
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np
import trimesh

from rlworld.rl.envs.managers.common.visual_mesh_transform import apply_local_transform

__all__ = ["extract_visual_meshes_from_mj_model"]


def extract_visual_meshes_from_mj_model(
    model,
    body_names: tuple[str, ...],
) -> dict[str, trimesh.Trimesh | None]:
    """Build per-body merged ``trimesh.Trimesh`` (body-local frame) for
    each name in ``body_names``. Collision-only geoms (``contype`` or
    ``conaffinity`` nonzero) are dropped.

    Body lookup tries bare ``mj_name2id`` first then falls back to
    mjlab's scoped suffix (``<entity>/<bare>``, e.g. ``robot/pelvis``).
    Raises ``RuntimeError`` when neither resolves.
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


# ─────────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────────


def _find_body_id_scoped(model, bare_name: str) -> int:
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
        f"visual_mesh: tracked body {bare_name!r} not found in mj_model "
        f"(tried exact + scoped suffix '/{bare_name}'). First "
        f"{len(sample)} body names: {sample}. body_names and the model "
        f"disagree, or the model uses an unexpected scheme."
    )


def _is_visual_geom(model, gid: int) -> bool:
    return int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0


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
    geom_type = int(model.geom_type[gid])
    size = np.asarray(model.geom_size[gid], dtype=np.float32)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = _extract_mesh_geom(model, gid)
    else:
        builder = _PRIMITIVE_BUILDERS.get(geom_type)
        if builder is None:
            return None
        mesh = builder(size)

    return apply_local_transform(
        mesh,
        local_pos=np.asarray(model.geom_pos[gid], dtype=np.float32),
        local_quat_wxyz=np.asarray(model.geom_quat[gid], dtype=np.float32),
    )


def _extract_mesh_geom(model, gid: int) -> trimesh.Trimesh:
    mid = int(model.geom_dataid[gid])
    v0, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    f0, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    return trimesh.Trimesh(
        vertices=np.asarray(model.mesh_vert[v0 : v0 + vn], dtype=np.float32).copy(),
        faces=np.asarray(model.mesh_face[f0 : f0 + fn], dtype=np.int32).copy(),
        process=False,
    )
