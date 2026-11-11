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

from ..bridge import BodyMeshGroup, SimulatorBridge, SimulatorGeometry, terrain_data_to_trimesh
from ..mjmodel_meshes import mjmodel_body_mesh_groups

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.managers.mujoco.scene import MujocoSceneManager


def _to_np(arr) -> np.ndarray:
    """``data.*`` arrays may be torch tensors or warp arrays — get numpy either way."""
    return arr.cpu().numpy() if hasattr(arr, "cpu") else arr.numpy()


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
        mesh_groups = mjmodel_body_mesh_groups(m)

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
