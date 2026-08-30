"""Newton simulator bridge for ViserScene.

Extracts mesh geometry from newton.Model and per-frame body transforms
from newton.State.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import numpy as np
import trimesh
import trimesh.visual
from newton import GeoType, Heightfield, Mesh, ShapeFlags
from scipy.spatial.transform import Rotation

from ..bridge import BodyMeshGroup, SimulatorBridge, SimulatorGeometry

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.newton.scene import NewtonSceneManager


# Primitives carry no source geometry; Newton's own viewer tessellates them
# from ``shape_scale`` through these same factories (``viewer.py:1640``).
# GeoType.PLANE is deliberately absent — the viewer draws its own ground,
# and an analytic plane has no finite extent to mesh.
_PRIMITIVE_MESH_FACTORIES = {
    GeoType.BOX: lambda s: Mesh.create_box(float(s[0]), float(s[1]), float(s[2]), compute_inertia=False),
    GeoType.SPHERE: lambda s: Mesh.create_sphere(float(s[0]), compute_inertia=False),
    GeoType.ELLIPSOID: lambda s: Mesh.create_ellipsoid(float(s[0]), float(s[1]), float(s[2]), compute_inertia=False),
    GeoType.CAPSULE: lambda s: Mesh.create_capsule(
        float(s[0]), float(s[1]), up_axis=newton.Axis.Z, compute_inertia=False
    ),
    GeoType.CYLINDER: lambda s: Mesh.create_cylinder(
        float(s[0]), float(s[1]), up_axis=newton.Axis.Z, compute_inertia=False
    ),
    GeoType.CONE: lambda s: Mesh.create_cone(float(s[0]), float(s[1]), up_axis=newton.Axis.Z, compute_inertia=False),
}


class NewtonBridge(SimulatorBridge):
    """Bridge between Newton simulator and ViserScene."""

    def __init__(self, scene_manager: NewtonSceneManager):
        super().__init__()
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

    @property
    def tracked_body_id(self) -> int | None:
        return self._tracked_body_local

    def extract_geometry(self) -> SimulatorGeometry:
        """Extract visual meshes from Newton's model for world 0."""
        mesh_groups: list[BodyMeshGroup] = []
        body_meshes: dict[int, list[trimesh.Trimesh]] = {}
        body_names: dict[int, str] = {}

        model = self._model

        # World-0 shapes are the ones whose body lives in world 0
        # (body < bodies_per_world) plus global static shapes (body < 0,
        # e.g. the terrain). Slicing the first shape_count // num_envs
        # entries instead silently dropped one robot shape whenever a
        # global terrain shape shifted the block.
        shape_flags = model.shape_flags.numpy()
        shape_bodies = model.shape_body.numpy()
        for shape_idx in range(model.shape_count):
            if int(shape_bodies[shape_idx]) >= self._bodies_per_world:
                continue
            # Skip non-visible shapes (collision-only geometry).
            if not (shape_flags[shape_idx] & ShapeFlags.VISIBLE):
                continue

            geo_src = model.shape_source[shape_idx]
            geo_type = GeoType(int(model.shape_type.numpy()[shape_idx]))
            # A primitive carries no source geometry — its dimensions live in
            # ``shape_scale`` and are turned into a mesh below. Skipping every
            # sourceless shape, which is what this used to do, made every box,
            # sphere and capsule in the scene invisible while mesh-based
            # robots rendered normally: a bench and a workpiece loaded from
            # URDF boxes simply were not there.
            if geo_src is None and geo_type not in _PRIMITIVE_MESH_FACTORIES:
                continue

            body_idx = int(model.shape_body.numpy()[shape_idx])
            # Convert to local body index (within world 0).
            local_body_idx = body_idx  # Already world-0 for first N shapes.

            if local_body_idx not in body_meshes:
                body_meshes[local_body_idx] = []
                # body_idx < 0 marks static world geometry (ground plane,
                # heightfield terrain) — it has no body, so don't index
                # ``body_label`` with a negative value (that would grab an
                # unrelated body's name).
                if 0 <= body_idx < len(model.body_label):
                    label = model.body_label[body_idx]
                elif body_idx < 0:
                    label = "terrain"
                else:
                    label = f"body_{local_body_idx}"
                body_names[local_body_idx] = label

            # Build trimesh from Newton mesh data.
            mesh = self._newton_mesh_to_trimesh(geo_src, shape_idx, geo_type)
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
            # A fixed mesh group is, by construction of ``_is_ground_body``,
            # a real ground/terrain mesh (e.g. the heightfield) — flat
            # analytic ground planes have no mesh and never appear here. Tell
            # the viewer so it skips its synthetic fallback ground.
            has_ground_mesh=any(group.is_fixed for group in mesh_groups),
        )

    def _fetch_body_transforms(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """One ``body_q.numpy()`` per frame covering all worlds.

        Returns ``(positions, quaternions)`` of shapes
        ``(num_bodies, 3)`` / ``(num_bodies, 4)`` (wxyz; Newton stores
        the rotation as xyzw).
        """
        body_q = self._scene_manager.state_0.body_q.numpy()  # (total_bodies, 7) — one transfer
        start = env_idx * self._bodies_per_world
        end = start + self._bodies_per_world
        block = body_q[start:end]
        positions = block[:, :3].astype(np.float32).copy()
        quaternions = block[:, [6, 3, 4, 5]].astype(np.float32).copy()  # xyzw -> wxyz
        return positions, quaternions

    def _fetch_tracked_world_velocity(self, env_idx: int) -> np.ndarray | None:
        """World-frame linear velocity at the tracked body.

        Only the velocity is fetched here — the base class supplies the
        quaternion needed for the world→body rotation from its cached
        transforms snapshot, so we never re-sync ``body_q``.
        """
        tracked = self._tracked_body_local
        if tracked is None:
            return None
        body_idx = env_idx * self._bodies_per_world + tracked
        body_qd = self._scene_manager.state_0.body_qd.numpy()[body_idx]  # (6,) [vx,vy,vz,wx,wy,wz]
        return body_qd[:3].astype(np.float32)

    def _is_ground_body(self, body_id: int) -> bool:
        """Check if a body is a ground/world plane by its label.

        Only the *leaf* segment of the label is tested so that MJCF
        hierarchical labels like ``T1/worldbody/Trunk`` are NOT
        misclassified as ground bodies (the ``worldbody`` prefix is
        just an XPath segment, not the body's own name).
        """
        # Static world geometry (ground plane, heightfield terrain) is
        # attached to body -1 — it has no body and is always world-fixed.
        if body_id < 0:
            return True
        if body_id >= len(self._model.body_label):
            return False
        # Use the last path segment as the body's own name.
        leaf = self._model.body_label[body_id].rsplit("/", 1)[-1].lower()
        return "ground" in leaf or "plane" in leaf

    def _find_tracked_body(self) -> int | None:
        """Find the floating base: the child body of a FREE joint.

        Mirrors the MuJoCo bridge. A name heuristic ("base"/"pelvis")
        picked the WRONG body on robots whose root is named otherwise
        (Mobile ALOHA's "footprint" fell through to a fallback that
        returned body 1 -- a wheel -- and the velocity arrow rotated
        the correct world velocity into a spinning wheel's frame).
        The ground plane is static geometry on body -1, so a fixed-base
        robot's root is simply the first body.
        """
        model = self._model
        joint_type = model.joint_type.numpy()
        joint_child = model.joint_child.numpy()
        for j in range(len(joint_type)):
            if int(joint_type[j]) == int(newton.JointType.FREE):
                child = int(joint_child[j])
                if child < self._bodies_per_world:
                    return child
        return 0 if self._bodies_per_world > 0 else None

    @staticmethod
    def _heightfield_to_vertices_faces(hf: Heightfield) -> tuple[np.ndarray, np.ndarray]:
        """Tessellate a Newton heightfield grid into (vertices, flat indices).

        Uses the same grid→world mapping as Newton's collision query
        (X spans columns, Y spans rows, ``z = min_z + data*(max_z-min_z)``)
        so the rendered surface matches the collision geometry. Extents are
        unscaled — the caller applies the shape's ``scale`` / ``transform``.
        """
        data = np.asarray(hf.data, dtype=np.float32)  # (nrow, ncol), normalised [0, 1]
        nrow, ncol = int(hf.nrow), int(hf.ncol)
        z_range = float(hf.max_z) - float(hf.min_z)

        xs = np.linspace(-hf.hx, hf.hx, ncol, dtype=np.float32)
        ys = np.linspace(-hf.hy, hf.hy, nrow, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        zz = float(hf.min_z) + data * z_range
        vertices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

        rr, cc = np.meshgrid(np.arange(nrow - 1), np.arange(ncol - 1), indexing="ij")
        v00 = (rr * ncol + cc).ravel()
        v01 = (rr * ncol + cc + 1).ravel()
        v10 = ((rr + 1) * ncol + cc).ravel()
        v11 = ((rr + 1) * ncol + cc + 1).ravel()
        # Winding: with X along columns and Y along rows, [v00, v10, v11]
        # winds clockwise seen from above (normal -Z) and viser back-face
        # culls the whole terrain when looked at from above — an invisible
        # ground. Order the triangles counter-clockwise instead.
        tri1 = np.stack([v00, v11, v10], axis=1)
        tri2 = np.stack([v00, v01, v11], axis=1)
        indices = np.concatenate([tri1, tri2], axis=0).reshape(-1).astype(np.int32)
        return vertices, indices

    def _newton_mesh_to_trimesh(
        self,
        geo_src,
        shape_idx: int,
        geo_type: GeoType,
    ) -> trimesh.Trimesh | None:
        """Convert a Newton mesh, heightfield or primitive to trimesh."""
        scale = self._model.shape_scale.numpy()[shape_idx]  # (3,)

        if geo_src is None:
            # Tessellated by Newton's own factories, from the same
            # ``shape_scale`` its viewer reads (``viewer.py:1650``), so a box
            # here is the box the simulator collides with rather than one
            # this bridge guessed the convention for. The vertices come out
            # in world units, so the scale must NOT be applied again.
            primitive = _PRIMITIVE_MESH_FACTORIES[geo_type](scale)
            vertices, indices = primitive.vertices, primitive.indices
            scale = np.ones(3)
        elif isinstance(geo_src, Heightfield):
            # Heightfields carry a 2D elevation grid, not a vertex/index mesh.
            vertices, indices = self._heightfield_to_vertices_faces(geo_src)
        else:
            vertices = geo_src.vertices  # (N, 3) float32
            indices = geo_src.indices  # (M,) int32

        if vertices is None or len(vertices) == 0:
            return None

        scaled_verts = vertices * scale

        # Apply shape local transform.
        shape_xform = self._model.shape_transform.numpy()[shape_idx]  # (7,)
        local_pos = shape_xform[:3]
        local_quat_xyzw = shape_xform[3:7]
        # Convert to rotation matrix.
        qx, qy, qz, qw = local_quat_xyzw
        rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        transformed_verts = (rot @ scaled_verts.T).T + local_pos

        faces = indices.reshape(-1, 3)

        # Color: the source mesh's own, else the colour Newton assigned the
        # shape (its palette gives adjacent shapes distinguishable colours,
        # which a single default grey does not).
        color = getattr(geo_src, "color", None)
        if color is None:
            color = self._model.shape_color.numpy()[shape_idx]
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
