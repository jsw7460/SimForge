"""Manages all Viser 3D scene handles and visualization state.

Simulator-agnostic: reads geometry and transforms through SimulatorBridge.

Camera tracking uses the Mjlab pattern: instead of moving the camera,
the entire scene is offset so the tracked body stays at the origin.
The user can freely orbit/pan/zoom around the origin.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
import viser
import viser.transforms as vtf

from .bridge import SimulatorBridge, SimulatorGeometry
from .scene_config import ViserSceneConfig


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _pbr_visual(
    rgb: tuple[int, int, int],
    metalness: float,
    roughness: float,
    opacity: float = 1.0,
) -> trimesh.visual.TextureVisuals:
    """A texture-less PBR visual (baseColorFactor + metallic/roughness).

    Exported to GLB by ``add_mesh_trimesh`` so viser renders it with a
    Three.js MeshStandardMaterial — i.e. real metalness/roughness.
    """
    a = float(np.clip(opacity, 0.0, 1.0))
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[c / 255.0 for c in rgb] + [a],
        metallicFactor=float(np.clip(metalness, 0.0, 1.0)),
        roughnessFactor=float(np.clip(roughness, 0.0, 1.0)),
        alphaMode="BLEND" if a < 1.0 else "OPAQUE",
        doubleSided=True,
    )
    return trimesh.visual.TextureVisuals(material=material)


def _create_plain_ground(
    color: tuple[int, int, int],
    size: float,
    metalness: float,
    roughness: float,
) -> trimesh.Trimesh:
    """A single large square ground quad with a flat PBR material."""
    half = size / 2.0
    vertices = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = _pbr_visual(color, metalness, roughness)
    return mesh


def _create_checkerboard_ground(
    size: float,
    divisions: int,
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
) -> trimesh.Trimesh:
    """Create a checkerboard ground plane mesh."""
    cell = size / divisions
    half = size / 2.0
    rgba_a = (*color_a, 255)
    rgba_b = (*color_b, 255)

    vertices = []
    faces = []
    face_colors = []

    for i in range(divisions):
        for j in range(divisions):
            x0 = -half + i * cell
            y0 = -half + j * cell
            x1 = x0 + cell
            y1 = y0 + cell

            vi = len(vertices)
            vertices.extend(
                [
                    [x0, y0, 0.0],
                    [x1, y0, 0.0],
                    [x1, y1, 0.0],
                    [x0, y1, 0.0],
                ]
            )
            faces.extend([[vi, vi + 1, vi + 2], [vi, vi + 2, vi + 3]])

            color = rgba_a if (i + j) % 2 == 0 else rgba_b
            face_colors.extend([color, color])

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        process=False,
    )
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=np.array(face_colors, dtype=np.uint8))
    return mesh


@dataclass
class _ArrowRequest:
    """Queued arrow for batch rendering."""

    start: np.ndarray
    end: np.ndarray
    color: tuple[int, int, int]
    radius: float


@dataclass
class _SphereRequest:
    """Queued sphere for batch rendering."""

    position: np.ndarray
    radius: float
    color: tuple[int, int, int]


class ViserScene:
    """Manages Viser scene handles for simulator-agnostic 3D visualization.

    Camera tracking: offsets the entire scene so the tracked body stays at
    the origin. This lets the user freely orbit/pan/zoom while the robot
    stays centered (same pattern as Mjlab).
    """

    def __init__(
        self,
        server: viser.ViserServer,
        bridge: SimulatorBridge,
        geometry: SimulatorGeometry,
        scene_config: ViserSceneConfig | None = None,
    ):
        self.server = server
        self.bridge = bridge
        self.geometry = geometry
        self.scene_config = scene_config or ViserSceneConfig()

        # State.
        self.env_idx: int = 0
        self.camera_tracking_enabled: bool = True
        self.needs_update: bool = True

        # Scene offset for camera tracking (Mjlab pattern).
        self._scene_offset: np.ndarray = np.zeros(3)

        # Mesh handles: body_id -> list of handles.
        self._body_handles: dict[int, list[Any]] = {}
        self._fixed_frame: viser.SceneNodeHandle | None = None
        self._ground_handle: Any = None

        # Debug visualization queues.
        self._arrow_queue: deque[_ArrowRequest] = deque()
        self._sphere_queue: deque[_SphereRequest] = deque()
        self._arrow_handles: list[Any] = []
        self._sphere_handles: list[Any] = []

        # Callback.
        self._on_env_switch = None

        # Build scene.
        self._create_mesh_handles()
        self._create_ground_plane()

    @classmethod
    def create(
        cls,
        server: viser.ViserServer,
        bridge: SimulatorBridge,
        scene_config: ViserSceneConfig | None = None,
    ) -> ViserScene:
        """Factory method."""
        geometry = bridge.extract_geometry()
        return cls(server=server, bridge=bridge, geometry=geometry, scene_config=scene_config)

    def _create_mesh_handles(self) -> None:
        """Create Viser mesh handles from geometry."""
        self._fixed_frame = self.server.scene.add_frame("/fixed_bodies")
        cfg = self.scene_config

        for group in self.geometry.mesh_groups:
            handles = []
            for mesh_idx, mesh in enumerate(group.meshes):
                name = f"/body_{group.body_id}/mesh_{mesh_idx}"
                if group.is_fixed:
                    name = f"/fixed_bodies/body_{group.body_id}/mesh_{mesh_idx}"

                if cfg.robot_color is not None:
                    mesh = mesh.copy()
                    # Fresh material per mesh — sharing a TextureVisuals would
                    # fight over its back-reference to ``mesh``.
                    mesh.visual = _pbr_visual(
                        cfg.robot_color, cfg.robot_metalness, cfg.robot_roughness, cfg.robot_opacity
                    )

                handle = self.server.scene.add_mesh_trimesh(
                    name=name,
                    mesh=mesh,
                    cast_shadow=cfg.cast_shadow,
                    receive_shadow=cfg.receive_shadow,
                )
                handles.append(handle)

            if handles:
                self._body_handles[group.body_id] = handles

    def _create_ground_plane(self) -> None:
        """Add the ground plane to the scene (kind/look from ViserSceneConfig)."""
        cfg = self.scene_config
        if cfg.ground_kind == "none":
            self._ground_handle = None
            return
        if cfg.ground_kind == "checkerboard":
            ground_mesh = _create_checkerboard_ground(
                cfg.ground_size, cfg.ground_divisions, cfg.ground_color, cfg.ground_color_alt
            )
        else:  # "plane"
            ground_mesh = _create_plain_ground(
                cfg.ground_color, cfg.ground_size, cfg.ground_metalness, cfg.ground_roughness
            )
        self._ground_handle = self.server.scene.add_mesh_trimesh(
            name="/ground_plane",
            mesh=ground_mesh,
            cast_shadow=False,
            receive_shadow=cfg.receive_shadow,
        )

    def update(self) -> None:
        """Update all dynamic body transforms from the bridge."""
        # Single GPU→CPU read per frame; the tracked-body position is sliced
        # from the result rather than re-queried.
        positions, quaternions = self.bridge.get_body_transforms(self.env_idx)

        # Compute scene offset for camera tracking.
        scene_offset = np.zeros(3)
        if self.camera_tracking_enabled and self.geometry.tracked_body_id is not None:
            scene_offset = -positions[self.geometry.tracked_body_id].astype(np.float64)
            # Only offset XY; keep Z so the ground stays at Z=0.
            scene_offset[2] = 0.0
        self._scene_offset = scene_offset

        # Update ground plane position.
        if self._ground_handle is not None:
            self._ground_handle.position = tuple(scene_offset)

        # Update fixed bodies frame.
        if self._fixed_frame is not None:
            self._fixed_frame.position = tuple(scene_offset)

        # Update dynamic bodies.
        for group in self.geometry.mesh_groups:
            if group.is_fixed:
                continue

            handles = self._body_handles.get(group.body_id)
            if not handles:
                continue

            body_pos = positions[group.body_id] + scene_offset
            # ``quaternions`` is a buffer the bridge reuses across frames — copy
            # the slice before handing it to viser handles.
            body_quat = quaternions[group.body_id].copy()  # wxyz

            for mesh_idx, handle in enumerate(handles):
                if group.local_positions and group.local_quaternions:
                    local_pos = group.local_positions[mesh_idx]
                    local_quat = group.local_quaternions[mesh_idx]
                    final_quat = _quaternion_multiply(body_quat, local_quat)
                    rot = vtf.SO3(wxyz=body_quat)
                    final_pos = body_pos + rot.as_matrix() @ local_pos
                    handle.wxyz = final_quat
                    handle.position = final_pos
                else:
                    handle.wxyz = body_quat
                    handle.position = body_pos

        # Debug visuals.
        self._sync_debug_visuals()
        self.needs_update = False

    # ==================== Debug Visualization ====================

    def add_arrow(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        radius: float = 0.01,
    ) -> None:
        """Queue an arrow for batch rendering."""
        self._arrow_queue.append(
            _ArrowRequest(
                start=np.asarray(start, dtype=np.float32),
                end=np.asarray(end, dtype=np.float32),
                color=color,
                radius=radius,
            )
        )

    def add_sphere(
        self,
        position: np.ndarray,
        radius: float = 0.03,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> None:
        """Queue a sphere for batch rendering."""
        self._sphere_queue.append(
            _SphereRequest(
                position=np.asarray(position, dtype=np.float32),
                radius=radius,
                color=color,
            )
        )

    def clear_debug(self) -> None:
        """Clear queued debug visuals."""
        self._arrow_queue.clear()
        self._sphere_queue.clear()

    def _sync_debug_visuals(self) -> None:
        """Render queued arrows and spheres, clearing previous ones."""
        for h in self._arrow_handles:
            h.remove()
        for h in self._sphere_handles:
            h.remove()
        self._arrow_handles.clear()
        self._sphere_handles.clear()

        offset = self._scene_offset

        for i, req in enumerate(self._arrow_queue):
            direction = req.end - req.start
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                continue
            r, g, b = req.color
            handle = self.server.scene.add_spline_catmull_rom(
                name=f"/debug/arrow_{i}",
                positions=np.stack([req.start + offset, req.end + offset]),
                color=(r, g, b),
                line_width=max(1.0, req.radius * 200),
            )
            self._arrow_handles.append(handle)

        for i, req in enumerate(self._sphere_queue):
            r, g, b = req.color
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=req.radius)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                face_colors=np.tile([r, g, b, 255], (len(mesh.faces), 1)),
            )
            handle = self.server.scene.add_mesh_trimesh(
                name=f"/debug/sphere_{i}",
                mesh=mesh,
                position=tuple(req.position + offset),
            )
            self._sphere_handles.append(handle)

        self._arrow_queue.clear()
        self._sphere_queue.clear()

    # ==================== GUI ====================

    def create_gui(self, tabs: Any) -> None:
        """Create scene-related GUI controls."""
        with tabs.add_tab("Scene", icon=viser.Icon.EYE):
            # Environment selector.
            if self.bridge.num_envs > 1:
                env_slider = self.server.gui.add_slider(
                    "Environment",
                    min=0,
                    max=self.bridge.num_envs - 1,
                    step=1,
                    initial_value=0,
                )

                @env_slider.on_update
                def _(event) -> None:
                    self.env_idx = int(event.target.value)
                    self.needs_update = True
                    if self._on_env_switch:
                        self._on_env_switch()

            # Camera tracking toggle.
            cam_track = self.server.gui.add_checkbox(
                "Camera tracking",
                initial_value=self.camera_tracking_enabled,
            )

            @cam_track.on_update
            def _(event) -> None:
                self.camera_tracking_enabled = event.target.value
                if self.camera_tracking_enabled:
                    # Reset camera look_at to origin when re-enabling.
                    for client in self.server.get_clients().values():
                        client.camera.look_at = (0.0, 0.0, 0.0)

    def set_on_env_switch(self, callback: Any) -> None:
        """Register a callback for environment switch events."""
        self._on_env_switch = callback

    def cleanup(self) -> None:
        """Remove all scene handles."""
        for handles in self._body_handles.values():
            for h in handles:
                h.remove()
        self._body_handles.clear()
        if self._ground_handle is not None:
            self._ground_handle.remove()
        for h in self._arrow_handles:
            h.remove()
        for h in self._sphere_handles:
            h.remove()
        self._arrow_handles.clear()
        self._sphere_handles.clear()
