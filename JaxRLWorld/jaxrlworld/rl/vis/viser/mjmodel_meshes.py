"""Visual meshes of a compiled ``mujoco.MjModel``, per body.

Shared by the viser bridges that render a robot from its MJCF (MuJoCo, and
backends that simulate a USD converted from the same MJCF). Kept out of
the ``bridges`` package, whose import pulls in every simulator's bridge.
"""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import trimesh.visual
from scipy.spatial.transform import Rotation

from .bridge import BodyMeshGroup

_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

# MuJoCo convention: geomgroup 3 is collision geometry — skip it for rendering.
_COLLISION_GROUP = 3


def _geom_to_trimesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    """Build a unit-frame trimesh for one MjModel geom (None → skip)."""
    gtype = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    G = mujoco.mjtGeom
    if gtype == G.mjGEOM_PLANE:
        return None  # ViserScene draws its own ground
    if gtype == G.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        va, nv = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, nf = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        verts = np.asarray(model.mesh_vert[va : va + nv], dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(model.mesh_face[fa : fa + nf], dtype=np.int64).reshape(-1, 3)
        if len(verts) == 0 or len(faces) == 0:
            return None
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if gtype == G.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if gtype == G.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size[:3])
    if gtype == G.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]))
    if gtype == G.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]))
    if gtype == G.mjGEOM_ELLIPSOID:
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        m.apply_scale(size[:3])
        return m
    return None  # HFIELD / SDF / other — skip


def mjmodel_body_mesh_groups(m: mujoco.MjModel) -> list[BodyMeshGroup]:
    """Per-body visual meshes of a compiled ``MjModel``, one group per MuJoCo body id.

    Visible, non-collision geoms of every non-world body, each geom's local
    pose (``geom_pos`` / ``geom_quat``, wxyz) baked into its vertices so a
    body's own pose is all a frame update needs.
    """
    groups: dict[int, BodyMeshGroup] = {}
    for gid in range(m.ngeom):
        if float(m.geom_rgba[gid, 3]) == 0.0:
            continue  # invisible
        if int(m.geom_group[gid]) == _COLLISION_GROUP:
            continue  # collision geometry
        bid = int(m.geom_bodyid[gid])
        if bid == 0:
            continue  # world-body decoration (ground/skybox) — ViserScene owns the ground
        mesh = _geom_to_trimesh(m, gid)
        if mesh is None:
            continue
        mesh = mesh.copy()
        rgba8 = np.clip(np.asarray(m.geom_rgba[gid]) * 255.0, 0, 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=np.tile(rgba8, (len(mesh.faces), 1)))
        # Bake the geom's local pose into vertices so the body's frame
        # is the only thing that needs per-frame updating.
        local_pos = np.asarray(m.geom_pos[gid], dtype=np.float64)
        local_quat_wxyz = np.asarray(m.geom_quat[gid], dtype=np.float64)
        if not (np.allclose(local_pos, 0.0) and np.allclose(local_quat_wxyz, _IDENTITY_QUAT_WXYZ)):
            w, x, y, z = local_quat_wxyz
            rot = Rotation.from_quat([x, y, z, w]).as_matrix()
            mesh.vertices = (rot @ np.asarray(mesh.vertices).T).T + local_pos
        grp = groups.get(bid)
        if grp is None:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
            grp = BodyMeshGroup(
                body_id=bid,
                body_name=name,
                is_fixed=False,
                meshes=[],
            )
            groups[bid] = grp
        grp.meshes.append(mesh)

    return list(groups.values())
