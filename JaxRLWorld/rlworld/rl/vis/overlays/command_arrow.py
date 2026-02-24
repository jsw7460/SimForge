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
    angular_vel_positive_color: tuple[float, ...] = (0.8, 0.2, 0.8, 0.9)
    angular_vel_negative_color: tuple[float, ...] = (0.2, 0.8, 0.8, 0.9)

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
            - cmd_lin_vel_x: (num_envs,) x velocity commands
            - cmd_lin_vel_y: (num_envs,) y velocity commands
            - cmd_ang_vel: (num_envs,) angular velocity commands (optional)
            - render_env_ids: list[int] environment indices to render
        """
        if not self.enabled:
            return

        base_pos = state.get("base_pos")
        cmd_lin_vel_x = state.get("cmd_lin_vel_x")
        cmd_lin_vel_y = state.get("cmd_lin_vel_y")
        cmd_ang_vel = state.get("cmd_ang_vel")
        render_env_ids = state.get("render_env_ids", [0])

        if base_pos is None or cmd_lin_vel_x is None or cmd_lin_vel_y is None:
            return

        # Convert to numpy
        base_pos = tensor_to_array(base_pos)
        cmd_lin_vel_x = tensor_to_array(cmd_lin_vel_x)
        cmd_lin_vel_y = tensor_to_array(cmd_lin_vel_y)

        for env_idx in render_env_ids:
            pos = base_pos[env_idx]
            vel_x = float(cmd_lin_vel_x[env_idx])
            vel_y = float(cmd_lin_vel_y[env_idx])

            # Draw linear velocity arrow
            self._draw_linear_velocity_arrow(pos, vel_x, vel_y)

            # Draw angular velocity indicator
            if self.config.show_angular_vel and cmd_ang_vel is not None:
                ang_vel = float(tensor_to_array(cmd_ang_vel)[env_idx])
                if abs(ang_vel) > self.config.angular_vel_threshold:
                    self._draw_angular_velocity_indicator(pos, ang_vel)

    def _draw_linear_velocity_arrow(
        self,
        pos: np.ndarray,
        vel_x: float,
        vel_y: float
    ) -> None:
        """Draw arrow for linear velocity command."""
        velocity = np.array([vel_x, vel_y, 0.0])
        magnitude = np.linalg.norm(velocity)

        if magnitude < 1e-6:
            return

        # Calculate arrow length (clamped)
        arrow_length = min(
            self.config.max_arrow_length,
            magnitude * self.config.arrow_length_scale
        )

        # Direction vector
        direction = velocity / magnitude

        # Create arrow mesh
        arrow_mesh = mu.create_arrow(
            length=arrow_length,
            radius=self.config.arrow_radius,
            l_ratio=self.config.arrow_head_length_ratio,
            r_ratio=self.config.arrow_head_radius_ratio,
            sections=16,
            body_color=self.config.linear_vel_body_color,
            head_color=self.config.linear_vel_head_color,
        )

        # Rotate arrow from z-axis to velocity direction
        rotation = gu.z_to_R(direction)
        arrow_mesh.vertices = gu.transform_by_R(arrow_mesh.vertices, rotation)

        # Position arrow above robot base
        arrow_pos = pos.copy()
        arrow_pos[2] += self.config.z_offset
        arrow_mesh.vertices += arrow_pos

        # Add to scene
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