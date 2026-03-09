from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.ext import pyrender
from genesis.utils.misc import tensor_to_array

from .base import Base3DOverlay, Overlay3DConfig

if TYPE_CHECKING:
    from rlworld.rl.vis.rasterizer_context import RLWorldRasterizerContext


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Extract yaw angle from wxyz quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


@dataclass
class CommandArrowConfig(Overlay3DConfig):
    """Configuration for command arrow overlay."""

    enabled: bool = True

    # Arrow appearance
    arrow_radius: float = 0.02
    arrow_length_scale: float = 0.5
    max_arrow_length: float = 1.0
    arrow_head_length_ratio: float = 0.3
    arrow_head_radius_ratio: float = 2.0

    # Colors (RGBA)
    linear_vel_body_color: tuple[float, ...] = (0.2, 0.8, 0.2, 0.9)
    linear_vel_head_color: tuple[float, ...] = (0.1, 0.6, 0.1, 0.9)
    actual_vel_body_color: tuple[float, ...] = (0.8, 0.4, 0.1, 0.9)
    actual_vel_head_color: tuple[float, ...] = (0.6, 0.3, 0.1, 0.9)
    angular_vel_positive_color: tuple[float, ...] = (0.8, 0.2, 0.8, 0.9)
    angular_vel_negative_color: tuple[float, ...] = (0.2, 0.8, 0.8, 0.9)

    # Actual velocity arrow
    show_actual_vel: bool = True

    # Position offset from base
    z_offset: float = 0.3

    # Angular velocity visualization
    show_angular_vel: bool = True
    angular_vel_threshold: float = 0.05


class CommandArrowOverlay(Base3DOverlay):
    """
    Overlay for visualizing velocity commands as 3D arrows.

    Renders:
    - Linear velocity: Arrow pointing in command direction
    - Angular velocity: Colored sphere indicating rotation direction
    """

    def __init__(
        self,
        context: "RLWorldRasterizerContext",
        config: CommandArrowConfig | None = None
    ):
        config = config or CommandArrowConfig()
        super().__init__(context, config)
        self.config: CommandArrowConfig = config

    def update(self, state: dict[str, Any]) -> None:
        """
        Update command arrow visualization.

        Expected state keys:
            - base_pos: (num_envs, 3) base positions
            - base_quat: (num_envs, 4) base quaternions (wxyz)
            - cmd_lin_vel_x: (num_envs,) x velocity commands (body frame)
            - cmd_lin_vel_y: (num_envs,) y velocity commands (body frame)
            - cmd_ang_vel: (num_envs,) angular velocity commands (optional)
            - render_env_ids: list[int] environment indices to render
        """
        if not self.enabled:
            return

        base_pos = state.get("base_pos")
        base_quat = state.get("base_quat")
        cmd_lin_vel_x = state.get("cmd_lin_vel_x")
        cmd_lin_vel_y = state.get("cmd_lin_vel_y")
        cmd_ang_vel = state.get("cmd_ang_vel")
        render_env_ids = state.get("render_env_ids", [0])

        if base_pos is None or cmd_lin_vel_x is None or cmd_lin_vel_y is None:
            return

        # Convert to numpy
        base_pos = tensor_to_array(base_pos)
        if base_quat is not None:
            base_quat = tensor_to_array(base_quat)
        cmd_lin_vel_x = tensor_to_array(cmd_lin_vel_x)
        cmd_lin_vel_y = tensor_to_array(cmd_lin_vel_y)

        for env_idx in render_env_ids:
            pos = base_pos[env_idx]
            vel_x = float(cmd_lin_vel_x[env_idx])
            vel_y = float(cmd_lin_vel_y[env_idx])

            # Extract yaw from quaternion to rotate body-frame command to world frame
            yaw = 0.0
            if base_quat is not None:
                quat = base_quat[env_idx]  # wxyz
                yaw = _yaw_from_quat_wxyz(quat)

            # Draw command arrow (green, rotated to world frame)
            self._draw_arrow(
                pos, vel_x, vel_y, yaw,
                body_color=self.config.linear_vel_body_color,
                head_color=self.config.linear_vel_head_color,
            )

            # Draw actual velocity arrow (orange, already body frame → rotate to world)
            actual_lin_vel = state.get("actual_lin_vel")
            if self.config.show_actual_vel and actual_lin_vel is not None:
                actual = tensor_to_array(actual_lin_vel)
                actual_vx = float(actual[env_idx, 0])
                actual_vy = float(actual[env_idx, 1])
                self._draw_arrow(
                    pos, actual_vx, actual_vy, yaw,
                    body_color=self.config.actual_vel_body_color,
                    head_color=self.config.actual_vel_head_color,
                    z_extra_offset=-0.05,
                )

            # Draw angular velocity indicator
            if self.config.show_angular_vel and cmd_ang_vel is not None:
                ang_vel = float(tensor_to_array(cmd_ang_vel)[env_idx])
                if abs(ang_vel) > self.config.angular_vel_threshold:
                    self._draw_angular_velocity_indicator(pos, ang_vel)

    def _draw_arrow(
        self,
        pos: np.ndarray,
        vel_x: float,
        vel_y: float,
        yaw: float = 0.0,
        body_color: tuple[float, ...] = (0.2, 0.8, 0.2, 0.9),
        head_color: tuple[float, ...] = (0.1, 0.6, 0.1, 0.9),
        z_extra_offset: float = 0.0,
    ) -> None:
        """Draw arrow for a body-frame velocity vector.

        Rotates by robot yaw to display in world frame.
        """
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        world_vel_x = cos_yaw * vel_x - sin_yaw * vel_y
        world_vel_y = sin_yaw * vel_x + cos_yaw * vel_y
        velocity = np.array([world_vel_x, world_vel_y, 0.0])
        magnitude = np.linalg.norm(velocity)

        if magnitude < 1e-6:
            return

        arrow_length = min(
            self.config.max_arrow_length,
            magnitude * self.config.arrow_length_scale
        )

        direction = velocity / magnitude

        arrow_mesh = mu.create_arrow(
            length=arrow_length,
            radius=self.config.arrow_radius,
            l_ratio=self.config.arrow_head_length_ratio,
            r_ratio=self.config.arrow_head_radius_ratio,
            sections=16,
            body_color=body_color,
            head_color=head_color,
        )

        rotation = gu.z_to_R(direction)
        arrow_mesh.vertices = gu.transform_by_R(arrow_mesh.vertices, rotation)

        arrow_pos = pos.copy()
        arrow_pos[2] += self.config.z_offset + z_extra_offset
        arrow_mesh.vertices += arrow_pos

        node = pyrender.Mesh.from_trimesh(arrow_mesh, is_marker=True)
        self.add_dynamic_mesh(node)

    def _draw_angular_velocity_indicator(
        self,
        pos: np.ndarray,
        angular_vel: float
    ) -> None:
        """Draw indicator for angular velocity command."""
        indicator_pos = pos.copy()
        indicator_pos[2] += self.config.z_offset + 0.15

        # Choose color based on rotation direction
        if angular_vel > 0:
            color = self.config.angular_vel_positive_color
        else:
            color = self.config.angular_vel_negative_color

        # Sphere size based on magnitude
        sphere_radius = 0.03 + 0.03 * min(1.0, abs(angular_vel))
        mesh = mu.create_sphere(radius=sphere_radius, color=color)

        pose = np.eye(4)
        pose[:3, 3] = indicator_pos
        node = pyrender.Mesh.from_trimesh(mesh, is_marker=True)
        self.context.add_dynamic_node(None, node, pose=pose)