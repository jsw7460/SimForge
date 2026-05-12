"""MuJoCo (mjlab) simulator bridge for ViserScene.

Extracts visual geometry from the compiled ``mujoco.MjModel`` and per-frame
body transforms from mjlab's batched ``data`` (``xpos`` / ``xquat``), so
MuJoCo eval renders through the same unified ``ViserScene`` path as Genesis
and Newton — i.e. the configurable ground + robot material from
``ViserSceneConfig`` apply to MuJoCo too.  (mjlab's own ``MjlabViserScene``
batched-mesh path renders all envs; this one renders the selected env, the
same as the Genesis/Newton bridges.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import trimesh
import trimesh.visual
from scipy.spatial.transform import Rotation

from ..bridge import BodyMeshGroup, SimulatorGeometry

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.mujoco.scene import MujocoSceneManager


# MuJoCo convention: geomgroup 3 is collision geometry — skip it for rendering.
_COLLISION_GROUP = 3


def _to_np(arr) -> np.ndarray:
    """``data.*`` arrays may be torch tensors or warp arrays — get numpy either way."""
    return arr.cpu().numpy() if hasattr(arr, "cpu") else arr.numpy()


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


class MujocoBridge:
    """Bridge between mjlab's MuJoCo backend and ViserScene."""

    def __init__(self, scene_manager: MujocoSceneManager):
        self._scene_manager = scene_manager
        self._model: mujoco.MjModel = scene_manager.mj_model
        self._num_envs = int(_to_np(scene_manager.data.xpos).shape[0])
        self._tracked_body_id = self._find_tracked_body()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def _find_tracked_body(self) -> int | None:
        m = self._model
        # The body that owns a free joint = the floating base.
        for b in range(1, m.nbody):
            adr, n = int(m.body_jntadr[b]), int(m.body_jntnum[b])
            for j in range(adr, adr + n):
                if int(m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
                    return b
        return 1 if m.nbody > 1 else 0

    def extract_geometry(self) -> SimulatorGeometry:
        m = self._model
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
            grp = groups.get(bid)
            if grp is None:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
                grp = BodyMeshGroup(
                    body_id=bid,
                    body_name=name,
                    is_fixed=False,
                    meshes=[],
                    local_positions=[],
                    local_quaternions=[],
                )
                groups[bid] = grp
            grp.meshes.append(mesh)
            grp.local_positions.append(np.asarray(m.geom_pos[gid], dtype=np.float32))
            grp.local_quaternions.append(np.asarray(m.geom_quat[gid], dtype=np.float32))  # wxyz

        return SimulatorGeometry(
            mesh_groups=list(groups.values()),
            num_bodies=m.nbody,
            tracked_body_id=self._tracked_body_id,
            tracked_body_name="base",
        )

    def get_body_transforms(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
        data = self._scene_manager.data
        positions = _to_np(data.xpos[env_idx]).astype(np.float32)  # (nbody, 3)
        quaternions = _to_np(data.xquat[env_idx]).astype(np.float32)  # (nbody, 4) wxyz
        return positions, quaternions

    def get_body_positions(self, env_idx: int) -> np.ndarray:
        return self.get_body_transforms(env_idx)[0]

    def get_body_quaternions(self, env_idx: int) -> np.ndarray:
        return self.get_body_transforms(env_idx)[1]

    def get_tracked_position(self, env_idx: int) -> np.ndarray:
        positions = self.get_body_transforms(env_idx)[0]
        if self._tracked_body_id is not None:
            return positions[self._tracked_body_id]
        return positions[0]

    def get_body_velocity(self, env_idx: int) -> np.ndarray | None:
        tracked = self._tracked_body_id
        if tracked is None:
            return None
        data = self._scene_manager.data
        if not hasattr(data, "cvel"):
            return None
        cvel = _to_np(data.cvel[env_idx])  # (nbody, 6) — [ang(3), lin(3)] in world frame at CoM
        world_vel = cvel[tracked, 3:6]
        w, x, y, z = _to_np(data.xquat[env_idx])[tracked]
        body_vel = Rotation.from_quat([x, y, z, w]).inv().apply(world_vel)
        return body_vel[:2].astype(np.float32)
