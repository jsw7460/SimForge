"""Genesis simulator bridge for ViserScene.

Extracts mesh geometry from Genesis entities (via vgeom.vmesh.trimesh)
and per-frame link transforms from entity.links.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import trimesh.visual
from scipy.spatial.transform import Rotation

from ..bridge import BodyMeshGroup, SimulatorBridge, SimulatorGeometry, terrain_data_to_trimesh

_IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.managers.genesis.scene import SceneManager


class GenesisBridge(SimulatorBridge):
    """Bridge between Genesis simulator and ViserScene."""

    def __init__(self, scene_manager: SceneManager):
        super().__init__()
        self._scene_manager = scene_manager
        self._num_envs = scene_manager.scene.n_envs

        # Cache link ordering for consistent body_id mapping.
        # We enumerate links across all entities to assign body_ids.
        self._link_map: list[tuple] = []  # [(entity, link, global_body_id), ...]
        # Per-entity contiguous body_id ranges, so the per-frame read is one
        # ``entity.get_links_pos()`` / ``get_links_quat()`` per entity (not per link).
        self._entity_ranges: list[tuple] = []  # [(entity, start_body_id, n_links), ...]
        self._tracked_body_id: int | None = None
        self._body_names: dict[int, str] = {}

        self._build_link_map()

        # Pre-allocated per-frame buffers (filled in place by _fetch_body_transforms).
        n = len(self._link_map)
        self._pos_buf = np.zeros((n, 3), dtype=np.float32)
        self._quat_buf = np.zeros((n, 4), dtype=np.float32)
        self._quat_buf[:, 0] = 1.0  # identity default

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def tracked_body_id(self) -> int | None:
        return self._tracked_body_id

    def _is_ground_entity(self, entity) -> bool:
        """Check if an entity is a ground plane (skip for Viser rendering)."""
        # Check by entity name in scene_manager.
        for name, ent in self._scene_manager.entities.items():
            if ent is entity:
                name_lower = name.lower()
                if "plane" in name_lower or "ground" in name_lower or "terrain" in name_lower:
                    return True
                break

        # Check morph type. A ``Terrain`` entity is skipped here and instead
        # rendered from the canonical ``TerrainImporter.data`` in ``extract_geometry``
        # (Genesis terrain mesh vertices are in the entity-local [0,L] frame
        # and the base-pos offset isn't applied to fixed bodies, so rendering
        # the entity directly would misplace it; the canonical grid is in
        # world coords and lines up with the robot).
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

    def _scene_name_of(self, entity) -> str | None:
        """The name this entity is registered under in the scene config."""
        for registry in (self._scene_manager.entities, self._scene_manager.rigid_objects):
            for name, ent in registry.items():
                if ent is entity:
                    return name
        return None

    def _build_link_map(self) -> None:
        """Build a flat list of (entity, link, body_id) for all entities.

        Body names are qualified with the entity they belong to once the
        scene holds more than one, because Genesis link names are bare:
        two copies of a robot both call a link ``link_1``, and anything
        keyed on the name — a picker, a log line — then cannot tell the
        two machines apart. Newton namespaces its labels the same way,
        and for the same reason. A single-entity scene keeps bare names.
        """
        body_id = 0
        entities = [
            e for e in self._scene_manager.scene.entities if hasattr(e, "links") and not self._is_ground_entity(e)
        ]
        qualify = len(entities) > 1
        for entity in self._scene_manager.scene.entities:
            if not hasattr(entity, "links"):
                continue
            # Skip ground plane — ViserScene adds its own checkerboard.
            if self._is_ground_entity(entity):
                continue
            scene_name = self._scene_name_of(entity)
            prefix = f"{scene_name}/" if (qualify and scene_name) else ""
            start = body_id
            for link in entity.links:
                self._link_map.append((entity, link, body_id))
                self._body_names[body_id] = f"{prefix}{getattr(link, 'name', f'link_{body_id}')}"
                # Track robot base.
                name = getattr(link, "name", "")
                if self._tracked_body_id is None and ("base" in name.lower() or "pelvis" in name.lower()):
                    self._tracked_body_id = body_id
                body_id += 1
            if body_id > start:
                # ``entity.get_links_pos()`` returns links in this same order
                # (``idx_local``), so body_ids [start, body_id) line up with it.
                self._entity_ranges.append((entity, start, body_id - start))

        # Default to first link if no base found.
        if self._tracked_body_id is None and self._link_map:
            self._tracked_body_id = 0

    def extract_geometry(self) -> SimulatorGeometry:
        """Extract visual meshes from Genesis entities.

        Each vgeom's local transform (``init_pos`` / ``init_quat``,
        wxyz) is baked into the mesh vertices so the per-frame
        :meth:`ViserScene.update` only needs the body's own pose —
        no per-mesh ``quaternion_multiply`` + rotation-matrix multiply.
        Mirrors how the Newton bridge already pre-bakes shape transforms
        (see :meth:`NewtonBridge._newton_mesh_to_trimesh`).
        """
        mesh_groups: list[BodyMeshGroup] = []

        for entity, link, body_id in self._link_map:
            vgeoms = getattr(link, "vgeoms", [])
            if not vgeoms:
                continue

            meshes = []
            for vgeom in vgeoms:
                mesh = self._extract_vgeom_mesh(vgeom)
                if mesh is None:
                    continue
                init_pos = np.asarray(
                    getattr(vgeom, "init_pos", np.zeros(3)),
                    dtype=np.float64,
                )
                init_quat_wxyz = np.asarray(
                    getattr(vgeom, "init_quat", _IDENTITY_QUAT_WXYZ),
                    dtype=np.float64,
                )
                if not (np.allclose(init_pos, 0.0) and np.allclose(init_quat_wxyz, _IDENTITY_QUAT_WXYZ)):
                    w, x, y, z = init_quat_wxyz
                    rot = Rotation.from_quat([x, y, z, w]).as_matrix()
                    mesh.vertices = (rot @ np.asarray(mesh.vertices).T).T + init_pos
                meshes.append(mesh)

            if meshes:
                # NOT ``link.is_fixed``. ``is_fixed`` tells the viewer the
                # group's vertices are already in world coordinates and need
                # no pose — which is true of a terrain mesh and of nothing
                # else. A welded link's vertices are in its own frame and it
                # has a pose like any other body; marking it fixed parks it
                # at the origin, so a bench-mounted arm and the bench itself
                # sink into the ground while everything bolted to them
                # appears to float. Only the terrain group below is fixed.
                link_name = self._body_names[body_id]
                mesh_groups.append(
                    BodyMeshGroup(
                        body_id=body_id,
                        body_name=link_name,
                        is_fixed=False,
                        meshes=meshes,
                        # local_positions / local_quaternions intentionally
                        # omitted — baked into mesh.vertices above so
                        # scene.update() can take the cheap branch.
                    )
                )

        # Generated terrain: render the canonical height grid in world
        # coordinates (the Terrain entity itself is skipped — see
        # _is_ground_entity) as a fixed body, and suppress the cosmetic ground.
        terrain_data = self._scene_manager.terrain.data
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
            num_bodies=len(self._link_map),
            tracked_body_id=self._tracked_body_id,
            tracked_body_name="base",
            has_ground_mesh=terrain_data is not None,
        )

    def _fetch_body_transforms(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """One ``get_links_pos`` + ``get_links_quat`` per Genesis entity.

        Genesis exposes link poses only at the entity level (no
        scene-wide accessor), so we issue one call per entity per data
        kind.  The base-class cache makes sure this runs at most once
        per ``(env_idx, frame)`` regardless of how many ``get_*``
        helpers ask for transforms during a tick.  The returned arrays
        are reused across frames (filled in place); the base-class
        contract is that callers don't retain them past the next
        ``begin_frame``.
        """
        for entity, start, n in self._entity_ranges:
            # (n_envs, n_links, 3) / (n_envs, n_links, 4) — one transfer each.
            self._pos_buf[start : start + n] = entity.get_links_pos()[env_idx].cpu().numpy()
            self._quat_buf[start : start + n] = entity.get_links_quat()[env_idx].cpu().numpy()
        return self._pos_buf, self._quat_buf

    def _fetch_tracked_world_velocity(self, env_idx: int) -> np.ndarray | None:
        """World-frame linear velocity of the robot base — single sync.

        The tracked-body quaternion needed for the world→body rotation
        is already cached by the base class (it came out of
        ``_fetch_body_transforms``), so we only read the velocity
        here.
        """
        robot = self._scene_manager.entities.get("robot")
        if robot is None or not hasattr(robot, "get_vel"):
            return None
        return robot.get_vel()[env_idx].cpu().numpy().astype(np.float32)

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
