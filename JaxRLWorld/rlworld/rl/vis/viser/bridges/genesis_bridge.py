"""Genesis simulator bridge for ViserScene.

Extracts mesh geometry from Genesis entities (via vgeom.vmesh.trimesh)
and per-frame link transforms from entity.links.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import trimesh.visual

from ..bridge import SimulatorBridge, SimulatorGeometry, BodyMeshGroup

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.genesis.scene import SceneManager


class GenesisBridge:
    """Bridge between Genesis simulator and ViserScene."""

    def __init__(self, scene_manager: SceneManager):
        self._scene_manager = scene_manager
        self._num_envs = scene_manager.scene.n_envs

        # Cache link ordering for consistent body_id mapping.
        # We enumerate links across all entities to assign body_ids.
        self._link_map: list[tuple] = []  # [(entity, link, global_body_id), ...]
        self._tracked_body_id: int | None = None

        self._build_link_map()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def _is_ground_entity(self, entity) -> bool:
        """Check if an entity is a ground plane (skip for Viser rendering)."""
        # Check by entity name in scene_manager.
        for name, ent in self._scene_manager.entities.items():
            if ent is entity:
                name_lower = name.lower()
                if "plane" in name_lower or "ground" in name_lower or "terrain" in name_lower:
                    return True
                break

        # Check morph type.
        morph = getattr(entity, "morph", None)
        if morph is not None:
            morph_cls = type(morph).__name__.lower()
            if "plane" in morph_cls or "terrain" in morph_cls:
                return True

        # Check if morph file references a plane.
        if morph is not None:
            morph_file = getattr(morph, "file", "") or ""
            if "plane" in morph_file.lower():
                return True

        return False

    def _build_link_map(self) -> None:
        """Build a flat list of (entity, link, body_id) for all entities."""
        body_id = 0
        for entity in self._scene_manager.scene.entities:
            if not hasattr(entity, "links"):
                continue
            # Skip ground plane — ViserScene adds its own checkerboard.
            if self._is_ground_entity(entity):
                continue
            for link in entity.links:
                self._link_map.append((entity, link, body_id))
                # Track robot base.
                name = getattr(link, "name", "")
                if self._tracked_body_id is None and (
                    "base" in name.lower() or "pelvis" in name.lower()
                ):
                    self._tracked_body_id = body_id
                body_id += 1

        # Default to first link if no base found.
        if self._tracked_body_id is None and self._link_map:
            self._tracked_body_id = 0

    def extract_geometry(self) -> SimulatorGeometry:
        """Extract visual meshes from Genesis entities."""
        mesh_groups: list[BodyMeshGroup] = []

        for entity, link, body_id in self._link_map:
            vgeoms = getattr(link, "vgeoms", [])
            if not vgeoms:
                continue

            meshes = []
            local_positions = []
            local_quaternions = []

            for vgeom in vgeoms:
                mesh = self._extract_vgeom_mesh(vgeom)
                if mesh is not None:
                    meshes.append(mesh)
                    # Local offset of vgeom relative to link frame.
                    init_pos = getattr(vgeom, "init_pos", np.zeros(3))
                    init_quat = getattr(vgeom, "init_quat", np.array([1, 0, 0, 0]))
                    local_positions.append(np.asarray(init_pos, dtype=np.float32))
                    # Genesis quaternion is wxyz.
                    local_quaternions.append(np.asarray(init_quat, dtype=np.float32))

            if meshes:
                is_fixed = getattr(link, "is_fixed", False)
                link_name = getattr(link, "name", f"link_{body_id}")
                mesh_groups.append(BodyMeshGroup(
                    body_id=body_id,
                    body_name=link_name,
                    is_fixed=is_fixed,
                    meshes=meshes,
                    local_positions=local_positions,
                    local_quaternions=local_quaternions,
                ))

        return SimulatorGeometry(
            mesh_groups=mesh_groups,
            num_bodies=len(self._link_map),
            tracked_body_id=self._tracked_body_id,
            tracked_body_name="base",
        )

    def get_body_positions(self, env_idx: int) -> np.ndarray:
        """Get link positions for one environment. Returns (num_bodies, 3)."""
        positions = np.zeros((len(self._link_map), 3), dtype=np.float32)

        for entity, link, body_id in self._link_map:
            pos = link.get_pos()  # (n_envs, 3) torch tensor
            positions[body_id] = pos[env_idx].cpu().numpy()

        return positions

    def get_body_quaternions(self, env_idx: int) -> np.ndarray:
        """Get link orientations for one environment.

        Returns (num_bodies, 4) in wxyz format.
        Genesis stores quaternions as wxyz.
        """
        quaternions = np.zeros((len(self._link_map), 4), dtype=np.float32)
        quaternions[:, 0] = 1.0  # Identity quaternion default.

        for entity, link, body_id in self._link_map:
            quat = link.get_quat()  # (n_envs, 4) wxyz torch tensor
            quaternions[body_id] = quat[env_idx].cpu().numpy()

        return quaternions

    def get_tracked_position(self, env_idx: int) -> np.ndarray:
        """Get tracked body position. Returns (3,)."""
        positions = self.get_body_positions(env_idx)
        if self._tracked_body_id is not None:
            return positions[self._tracked_body_id]
        return positions[0]

    def get_body_velocity(self, env_idx: int) -> np.ndarray | None:
        """Body-frame linear velocity of the robot base. Returns (2,) [vx, vy]."""
        robot = self._scene_manager.entities.get("robot")
        if robot is None or not hasattr(robot, "get_vel"):
            return None
        world_vel = robot.get_vel()[env_idx].cpu().numpy()  # (3,)
        quat_wxyz = robot.get_quat()[env_idx].cpu().numpy()  # (4,) wxyz
        w, x, y, z = quat_wxyz
        from scipy.spatial.transform import Rotation
        body_vel = Rotation.from_quat([x, y, z, w]).inv().apply(world_vel)
        return body_vel[:2].astype(np.float32)

    @staticmethod
    def _extract_vgeom_mesh(vgeom) -> trimesh.Trimesh | None:
        """Extract trimesh from a Genesis RigidVisGeom."""
        # Genesis wraps trimesh internally.
        vmesh = getattr(vgeom, "vmesh", None)
        if vmesh is not None and hasattr(vmesh, "trimesh"):
            mesh = vmesh.trimesh
            if isinstance(mesh, trimesh.Trimesh):
                return mesh.copy()

        # Fallback: build from raw vertex/face data.
        verts = getattr(vgeom, "init_vverts", None)
        faces = getattr(vgeom, "init_vfaces", None)
        if verts is not None and faces is not None and len(verts) > 0:
            return trimesh.Trimesh(
                vertices=np.asarray(verts),
                faces=np.asarray(faces),
                process=False,
            )

        return None
