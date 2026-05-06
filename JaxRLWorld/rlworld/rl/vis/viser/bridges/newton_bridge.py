"""Newton simulator bridge for ViserScene.

Extracts mesh geometry from newton.Model and per-frame body transforms
from newton.State.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import trimesh.visual
from newton import ShapeFlags

from ..bridge import BodyMeshGroup, SimulatorGeometry

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.newton.scene import NewtonSceneManager


class NewtonBridge:
    """Bridge between Newton simulator and ViserScene."""

    def __init__(self, scene_manager: NewtonSceneManager):
        self._scene_manager = scene_manager
        self._model = scene_manager.model
        self._num_envs = scene_manager.config.num_worlds

        # Compute body count per world.
        # Newton replicates all bodies across worlds.
        total_bodies = self._model.body_count
        self._bodies_per_world = total_bodies // max(1, self._num_envs)

        # Compute shape-to-body mapping for one world.
        self._shapes_per_world = self._model.shape_count // max(1, self._num_envs)

        # Find the tracked body (first non-root body, typically the robot base).
        self._tracked_body_local = self._find_tracked_body()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def extract_geometry(self) -> SimulatorGeometry:
        """Extract visual meshes from Newton's model for world 0."""
        mesh_groups: list[BodyMeshGroup] = []
        body_meshes: dict[int, list[trimesh.Trimesh]] = {}
        body_names: dict[int, str] = {}

        model = self._model

        # Only extract shapes for world 0.
        shape_flags = model.shape_flags.numpy()
        for shape_idx in range(self._shapes_per_world):
            # Skip non-visible shapes (collision-only geometry).
            if not (shape_flags[shape_idx] & ShapeFlags.VISIBLE):
                continue

            geo_src = model.shape_source[shape_idx]
            if geo_src is None:
                continue

            body_idx = int(model.shape_body.numpy()[shape_idx])
            # Convert to local body index (within world 0).
            local_body_idx = body_idx  # Already world-0 for first N shapes.

            if local_body_idx not in body_meshes:
                body_meshes[local_body_idx] = []
                label = model.body_label[body_idx] if body_idx < len(model.body_label) else f"body_{local_body_idx}"
                body_names[local_body_idx] = label

            # Build trimesh from Newton mesh data.
            mesh = self._newton_mesh_to_trimesh(geo_src, shape_idx)
            if mesh is not None:
                body_meshes[local_body_idx].append(mesh)

        # Build BodyMeshGroups.
        for body_id, meshes in body_meshes.items():
            is_fixed = self._is_ground_body(body_id)
            mesh_groups.append(
                BodyMeshGroup(
                    body_id=body_id,
                    body_name=body_names.get(body_id, f"body_{body_id}"),
                    is_fixed=is_fixed,
                    meshes=meshes,
                )
            )

        return SimulatorGeometry(
            mesh_groups=mesh_groups,
            num_bodies=self._bodies_per_world,
            tracked_body_id=self._tracked_body_local,
            tracked_body_name="base",
        )

    def get_body_positions(self, env_idx: int) -> np.ndarray:
        """Get body positions for one environment. Returns (num_bodies, 3)."""
        state = self._scene_manager.state_0
        body_q = state.body_q.numpy()  # (total_bodies, 7)

        start = env_idx * self._bodies_per_world
        end = start + self._bodies_per_world
        return body_q[start:end, :3].copy()

    def get_body_quaternions(self, env_idx: int) -> np.ndarray:
        """Get body orientations for one environment.

        Returns (num_bodies, 4) in wxyz format.
        Newton stores quaternions as [qx, qy, qz, qw], we convert to [qw, qx, qy, qz].
        """
        state = self._scene_manager.state_0
        body_q = state.body_q.numpy()  # (total_bodies, 7)

        start = env_idx * self._bodies_per_world
        end = start + self._bodies_per_world
        quats_xyzw = body_q[start:end, 3:7]  # (N, 4) xyzw

        # Convert xyzw -> wxyz.
        quats_wxyz = np.empty_like(quats_xyzw)
        quats_wxyz[:, 0] = quats_xyzw[:, 3]  # w
        quats_wxyz[:, 1:] = quats_xyzw[:, :3]  # xyz
        return quats_wxyz

    def get_tracked_position(self, env_idx: int) -> np.ndarray:
        """Get tracked body position. Returns (3,)."""
        positions = self.get_body_positions(env_idx)
        if self._tracked_body_local is not None:
            return positions[self._tracked_body_local]
        return positions[0]

    def get_body_velocity(self, env_idx: int) -> np.ndarray | None:
        """Body-frame linear velocity of the tracked body. Returns (2,) [vx, vy]."""
        tracked = self._tracked_body_local
        if tracked is None:
            return None
        state = self._scene_manager.state_0
        body_idx = env_idx * self._bodies_per_world + tracked
        body_qd = state.body_qd.numpy()[body_idx]  # (6,) [vx,vy,vz,wx,wy,wz]
        world_vel = body_qd[:3]
        body_q = state.body_q.numpy()[body_idx]  # (7,) [x,y,z,qx,qy,qz,qw]
        qx, qy, qz, qw = body_q[3:7]
        from scipy.spatial.transform import Rotation

        body_vel = Rotation.from_quat([qx, qy, qz, qw]).inv().apply(world_vel)
        return body_vel[:2].astype(np.float32)

    def _is_ground_body(self, body_id: int) -> bool:
        """Check if a body is a ground/world plane by its label.

        Only the *leaf* segment of the label is tested so that MJCF
        hierarchical labels like ``T1/worldbody/Trunk`` are NOT
        misclassified as ground bodies (the ``worldbody`` prefix is
        just an XPath segment, not the body's own name).
        """
        if body_id >= len(self._model.body_label):
            return False
        # Use the last path segment as the body's own name.
        leaf = self._model.body_label[body_id].rsplit("/", 1)[-1].lower()
        return "ground" in leaf or "plane" in leaf

    def _find_tracked_body(self) -> int | None:
        """Find the robot base body index (first non-ground body)."""
        model = self._model
        for i in range(min(self._bodies_per_world, len(model.body_label))):
            label = model.body_label[i]
            if "base" in label.lower() or "pelvis" in label.lower():
                return i
        # Default to body 1 (skip ground at 0).
        return min(1, self._bodies_per_world - 1) if self._bodies_per_world > 1 else 0

    def _newton_mesh_to_trimesh(
        self,
        geo_src,
        shape_idx: int,
    ) -> trimesh.Trimesh | None:
        """Convert a Newton mesh to trimesh.Trimesh."""
        vertices = geo_src.vertices  # (N, 3) float32
        indices = geo_src.indices  # (M,) int32

        if vertices is None or len(vertices) == 0:
            return None

        # Apply shape scale.
        scale = self._model.shape_scale.numpy()[shape_idx]  # (3,)
        scaled_verts = vertices * scale

        # Apply shape local transform.
        shape_xform = self._model.shape_transform.numpy()[shape_idx]  # (7,)
        local_pos = shape_xform[:3]
        local_quat_xyzw = shape_xform[3:7]
        # Convert to rotation matrix.
        qx, qy, qz, qw = local_quat_xyzw
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        transformed_verts = (rot @ scaled_verts.T).T + local_pos

        faces = indices.reshape(-1, 3)

        # Color.
        color = getattr(geo_src, "color", None)
        if color is not None:
            rgba = [int(c * 255) for c in color[:3]] + [255]
            visual = trimesh.visual.ColorVisuals(face_colors=np.tile(rgba, (len(faces), 1)))
        else:
            visual = trimesh.visual.ColorVisuals(face_colors=np.tile([180, 180, 180, 255], (len(faces), 1)))

        mesh = trimesh.Trimesh(
            vertices=transformed_verts,
            faces=faces,
            visual=visual,
            process=False,
        )
        return mesh
