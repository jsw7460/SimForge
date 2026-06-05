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

from ..bridge import BodyMeshGroup, SimulatorBridge, SimulatorGeometry, terrain_data_to_trimesh

_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

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


class MujocoBridge(SimulatorBridge):
    """Bridge between mjlab's MuJoCo backend and ViserScene."""

    def __init__(self, scene_manager: MujocoSceneManager):
        super().__init__()
        self._scene_manager = scene_manager
        self._model: mujoco.MjModel = scene_manager.mj_model
        self._num_envs = int(_to_np(scene_manager.data.xpos).shape[0])
        self._tracked_body_id = self._find_tracked_body()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def tracked_body_id(self) -> int | None:
        return self._tracked_body_id

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
        """Extract per-body visual meshes from the compiled ``MjModel``.

        Each geom's local transform (``geom_pos`` / ``geom_quat``, wxyz)
        is baked into the mesh vertices so the per-frame
        :meth:`ViserScene.update` only needs the body's own pose —
        matching the Genesis / Newton bridges and letting ViserScene
        take the cheap ``handle.position = body_pos`` branch.
        """
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

        mesh_groups = list(groups.values())

        # Generated terrain is an ``<hfield>`` geom, which ``_geom_to_trimesh``
        # skips (MuJoCo hfields aren't a renderable vertex mesh). Render the
        # canonical height grid instead (matches what was injected) as a
        # fixed body, and suppress the viewer's cosmetic flat ground.
        terrain_data = self._scene_manager.terrain.data
        has_ground_mesh = terrain_data is not None
        if terrain_data is not None:
            mesh_groups.append(
                BodyMeshGroup(
                    body_id=-1,
                    body_name="terrain",
                    is_fixed=True,
                    meshes=[terrain_data_to_trimesh(terrain_data)],
                )
            )

        return SimulatorGeometry(
            mesh_groups=mesh_groups,
            num_bodies=m.nbody,
            tracked_body_id=self._tracked_body_id,
            tracked_body_name="base",
            has_ground_mesh=has_ground_mesh,
        )

    def _fetch_body_transforms(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Slice ``data.xpos`` / ``data.xquat`` — mjwarp keeps these as
        batched arrays sized over all envs and (typically) host-side,
        so this is effectively free.
        """
        data = self._scene_manager.data
        positions = _to_np(data.xpos[env_idx]).astype(np.float32)  # (nbody, 3)
        quaternions = _to_np(data.xquat[env_idx]).astype(np.float32)  # (nbody, 4) wxyz
        return positions, quaternions

    def _fetch_tracked_world_velocity(self, env_idx: int) -> np.ndarray | None:
        """World-frame linear velocity at the tracked body from ``data.cvel``."""
        data = self._scene_manager.data
        if not hasattr(data, "cvel"):
            return None
        cvel = _to_np(data.cvel[env_idx])  # (nbody, 6) — [ang(3), lin(3)] in world frame at CoM
        return cvel[self._tracked_body_id, 3:6].astype(np.float32)
