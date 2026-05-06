"""Unified Viser-based visualization manager for SimForge.

Passive observer pattern: advance() is called from the environment's
_step_physics() method. The viewer does NOT own the simulation loop.

Works with any simulator through SimulatorBridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import trimesh.visual
import viser

from .bridge import SimulatorBridge
from .overlays import ViserDebugOverlays, ViserTermOverlays
from .scene import ViserScene

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


# Arrow colors (RGB 0-255).
_CMD_ARROW_COLOR = (50, 200, 50)  # Green for command velocity
_ACTUAL_ARROW_COLOR = (200, 130, 30)  # Orange for actual velocity
_ANG_VEL_POS_COLOR = (200, 50, 200)  # Magenta for positive angular vel
_ANG_VEL_NEG_COLOR = (50, 200, 200)  # Cyan for negative angular vel

# Arrow settings.
_ARROW_LENGTH_SCALE = 0.5
_MAX_ARROW_LENGTH = 1.0
_ARROW_Z_OFFSET = 0.8  # Above the robot's head for humanoids.
_ANG_VEL_THRESHOLD = 0.05

# Arrow mesh settings.
_ARROW_SHAFT_RADIUS = 0.015
_ARROW_HEAD_RADIUS = 0.035
_SHAFT_LENGTH_RATIO = 0.75
_HEAD_LENGTH_RATIO = 0.25


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Extract yaw angle from wxyz quaternion."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _rotation_quat_from_vectors(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """Compute quaternion (wxyz) that rotates from_vec to to_vec."""
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)

    if np.allclose(from_vec, to_vec):
        return np.array([1.0, 0.0, 0.0, 0.0])

    if np.allclose(from_vec, -to_vec):
        perp = np.array([1.0, 0.0, 0.0])
        if abs(from_vec[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(from_vec, perp)
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])

    cross = np.cross(from_vec, to_vec)
    dot = np.dot(from_vec, to_vec)
    w = 1.0 + dot
    quat = np.array([w, cross[0], cross[1], cross[2]])
    quat = quat / np.linalg.norm(quat)
    return quat


# Pre-computed unit meshes for arrows (created once, reused).
_UNIT_SHAFT_MESH: trimesh.Trimesh | None = None
_UNIT_HEAD_MESH: trimesh.Trimesh | None = None


def _get_unit_shaft_mesh() -> trimesh.Trimesh:
    """Unit cylinder: radius=1.0, height=1.0, centered at z=0.5."""
    global _UNIT_SHAFT_MESH
    if _UNIT_SHAFT_MESH is None:
        _UNIT_SHAFT_MESH = trimesh.creation.cylinder(radius=1.0, height=1.0)
        _UNIT_SHAFT_MESH.apply_translation(np.array([0, 0, 0.5]))
    return _UNIT_SHAFT_MESH


def _get_unit_head_mesh() -> trimesh.Trimesh:
    """Unit cone: radius=2.0, height=1.0, base at z=0."""
    global _UNIT_HEAD_MESH
    if _UNIT_HEAD_MESH is None:
        _UNIT_HEAD_MESH = trimesh.creation.cone(radius=2.0, height=1.0)
    return _UNIT_HEAD_MESH


@dataclass
class ViserViewerConfig:
    """Configuration for the Viser viewer."""

    port: int = 8080
    share: bool = True
    label: str = "SimForge"
    update_every_n_steps: int = 2  # 30Hz at 60Hz physics
    enable_reward_plots: bool = True
    enable_debug_viz: bool = False
    enable_command_arrows: bool = True
    enable_actual_vel_arrow: bool = True


class ViserVisualizationManager:
    """Passive Viser-based visualization manager.

    Integrates with the existing VisualizationManager pattern used by
    GenesisEnv and NewtonEnv. Call advance() every physics step.
    """

    def __init__(
        self,
        env: World,
        bridge: SimulatorBridge,
        config: ViserViewerConfig | None = None,
    ):
        self.env = env
        self.bridge = bridge
        self.config = config or ViserViewerConfig()

        self._step_counter = 0

        # Create Viser server.
        self.server = viser.ViserServer(
            port=self.config.port,
            label=self.config.label,
        )
        if self.config.share:
            self.server.request_share_url()

        # Create scene.
        self.scene = ViserScene.create(self.server, self.bridge)

        # Arrow handles: each is a (shaft_handle, head_handle) tuple or None.
        self._cmd_arrow_handles: tuple | None = None
        self._actual_arrow_handles: tuple | None = None
        self._ang_vel_handle = None

        # Setup GUI.
        self._setup_gui()

        print(f"[ViserViewer] Started on port {self.config.port}. Open the URL above to view.")

    def _setup_gui(self) -> None:
        """Create GUI tabs and overlays."""
        tabs = self.server.gui.add_tab_group()

        # Scene tab (env selector, camera controls).
        self.scene.create_gui(tabs)
        self.scene.set_on_env_switch(self._on_env_switch)

        # Reward plots tab.
        self._term_overlays: ViserTermOverlays | None = None
        if self.config.enable_reward_plots:
            self._term_overlays = ViserTermOverlays(
                server=self.server,
                env=self.env,
                scene=self.scene,
            )
            self._term_overlays.setup_tabs(tabs)

        # Debug overlays.
        self._debug_overlays: ViserDebugOverlays | None = None
        if self.config.enable_debug_viz:
            self._debug_overlays = ViserDebugOverlays(
                env=self.env,
                scene=self.scene,
            )

    def _on_env_switch(self) -> None:
        """Handle environment index change."""
        if self._term_overlays:
            self._term_overlays.on_env_switch()
        if self._debug_overlays:
            self._debug_overlays.on_env_switch()

    def setup(self) -> None:
        """Post-initialization setup (called after scene is built)."""
        pass

    def advance(self) -> None:
        """Called every physics step from the environment.

        Gates updates to reduce overhead (default: every 2 steps = 30Hz).
        """
        self._step_counter += 1
        if self._step_counter % self.config.update_every_n_steps != 0:
            return

        with self.server.atomic():
            # Update 3D scene.
            self.scene.update()

            # Draw command/velocity arrows.
            if self.config.enable_command_arrows:
                self._update_command_arrows()

            # Update debug visualization.
            if self._debug_overlays:
                self._debug_overlays.queue()

            # Update reward plots.
            if self._term_overlays:
                self._term_overlays.update()

    def _update_command_arrows(self) -> None:
        """Draw command velocity and actual velocity arrows above the robot."""
        env = self.env
        env_idx = self.scene.env_idx
        offset = self.scene._scene_offset

        # Get robot base position and orientation.
        base_pos = self.bridge.get_tracked_position(env_idx)
        base_quat = self.bridge.get_body_quaternions(env_idx)
        tracked_id = self.scene.geometry.tracked_body_id
        if tracked_id is None:
            return
        quat_wxyz = base_quat[tracked_id]
        yaw = _yaw_from_quat_wxyz(quat_wxyz)

        # Apply scene offset.
        arrow_origin = base_pos + offset
        arrow_origin[2] += _ARROW_Z_OFFSET

        # Get command data.
        cmd_manager = getattr(env, "command_manager", None)
        if cmd_manager is None:
            return

        cmd_vx = getattr(cmd_manager, "lin_vel_x", None)
        cmd_vy = getattr(cmd_manager, "lin_vel_y", None)
        cmd_ang = getattr(cmd_manager, "ang_vel", None)

        # Command arrow (green).
        if cmd_vx is not None and cmd_vy is not None:
            vx = float(cmd_vx[env_idx])
            vy = float(cmd_vy[env_idx])
            self._cmd_arrow_handles = self._draw_velocity_arrow(
                arrow_origin,
                vx,
                vy,
                yaw,
                color=_CMD_ARROW_COLOR,
                name="/overlay/cmd_arrow",
                old_handles=self._cmd_arrow_handles,
            )

        # Actual velocity arrow (orange).
        if self.config.enable_actual_vel_arrow:
            actual_vel = self.bridge.get_body_velocity(env_idx)
            if actual_vel is not None:
                origin_actual = arrow_origin.copy()
                origin_actual[2] -= 0.05  # Slightly below command arrow.
                self._actual_arrow_handles = self._draw_velocity_arrow(
                    origin_actual,
                    float(actual_vel[0]),
                    float(actual_vel[1]),
                    yaw,
                    color=_ACTUAL_ARROW_COLOR,
                    name="/overlay/actual_arrow",
                    old_handles=self._actual_arrow_handles,
                )

        # Angular velocity indicator (sphere).
        if cmd_ang is not None:
            ang_vel = float(cmd_ang[env_idx])
            self._ang_vel_handle = self._draw_angular_indicator(
                arrow_origin,
                ang_vel,
                old_handle=self._ang_vel_handle,
            )

    def _draw_velocity_arrow(
        self,
        origin: np.ndarray,
        vel_x: float,
        vel_y: float,
        yaw: float,
        color: tuple[int, int, int],
        name: str,
        old_handles: tuple | None = None,
    ) -> tuple | None:
        """Draw a velocity arrow (shaft + cone head) in world frame."""
        # Rotate body-frame velocity to world frame.
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        world_vx = cos_yaw * vel_x - sin_yaw * vel_y
        world_vy = sin_yaw * vel_x + cos_yaw * vel_y

        magnitude = np.sqrt(world_vx**2 + world_vy**2)
        if magnitude < 1e-4:
            if old_handles is not None:
                for h in old_handles:
                    h.remove()
            return None

        arrow_length = min(_MAX_ARROW_LENGTH, magnitude * _ARROW_LENGTH_SCALE)
        direction = np.array([world_vx, world_vy, 0.0]) / magnitude

        # Remove old handles.
        if old_handles is not None:
            for h in old_handles:
                h.remove()

        z_axis = np.array([0.0, 0.0, 1.0])
        rotation_quat = _rotation_quat_from_vectors(z_axis, direction)

        # Shaft: cylinder from origin along direction.
        shaft_length = _SHAFT_LENGTH_RATIO * arrow_length
        shaft_mesh = _get_unit_shaft_mesh()
        r, g, b = color
        shaft_colored = shaft_mesh.copy()
        shaft_colored.visual = trimesh.visual.ColorVisuals(
            mesh=shaft_colored,
            face_colors=np.tile([r, g, b, 255], (len(shaft_colored.faces), 1)),
        )
        shaft_handle = self.server.scene.add_mesh_trimesh(
            name=f"{name}/shaft",
            mesh=shaft_colored,
            position=tuple(origin),
            wxyz=tuple(rotation_quat),
            scale=(_ARROW_SHAFT_RADIUS, _ARROW_SHAFT_RADIUS, shaft_length),
        )

        # Head: cone at end of shaft.
        head_length = _HEAD_LENGTH_RATIO * arrow_length
        head_pos = origin + direction * shaft_length
        head_mesh = _get_unit_head_mesh()
        head_colored = head_mesh.copy()
        head_colored.visual = trimesh.visual.ColorVisuals(
            mesh=head_colored,
            face_colors=np.tile([r, g, b, 255], (len(head_colored.faces), 1)),
        )
        head_handle = self.server.scene.add_mesh_trimesh(
            name=f"{name}/head",
            mesh=head_colored,
            position=tuple(head_pos),
            wxyz=tuple(rotation_quat),
            scale=(_ARROW_HEAD_RADIUS, _ARROW_HEAD_RADIUS, head_length),
        )

        return (shaft_handle, head_handle)

    def _draw_angular_indicator(
        self,
        origin: np.ndarray,
        ang_vel: float,
        old_handle=None,
    ):
        """Draw angular velocity indicator sphere."""
        if old_handle is not None:
            old_handle.remove()

        if abs(ang_vel) < _ANG_VEL_THRESHOLD:
            return None

        color = _ANG_VEL_POS_COLOR if ang_vel > 0 else _ANG_VEL_NEG_COLOR
        radius = 0.03 + 0.03 * min(1.0, abs(ang_vel))

        pos = origin.copy()
        pos[2] += 0.15

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        r, g, b = color
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=np.tile([r, g, b, 255], (len(mesh.faces), 1)),
        )
        handle = self.server.scene.add_mesh_trimesh(
            name="/overlay/ang_vel",
            mesh=mesh,
            position=tuple(pos),
        )
        return handle

    def start_recording(self) -> None:
        """Start video recording (placeholder)."""
        pass

    def stop_recording(self) -> None:
        """Stop video recording (placeholder)."""
        pass

    def close(self) -> None:
        """Shut down the Viser server."""
        self.scene.cleanup()
        if self._term_overlays:
            self._term_overlays.cleanup()

    def reset(self, env_ids=None) -> None:
        """Reset visualization state."""
        pass
