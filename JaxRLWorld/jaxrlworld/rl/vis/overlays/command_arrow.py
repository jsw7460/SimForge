from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import Base3DOverlay, Overlay3DConfig, OverlayState

if TYPE_CHECKING:
    import genesis as gs


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Yaw angle of each wxyz quaternion, in radians."""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass
class CommandArrowConfig(Overlay3DConfig):
    """Configuration for command arrow overlay."""

    enabled: bool = True

    # Arrow geometry: length = min(max_arrow_length, |velocity| * arrow_length_scale)
    arrow_radius: float = 0.02
    arrow_length_scale: float = 0.5
    max_arrow_length: float = 1.0

    # Colors (RGBA)
    linear_vel_color: tuple[float, ...] = (0.2, 0.8, 0.2, 0.9)
    actual_vel_color: tuple[float, ...] = (0.8, 0.4, 0.1, 0.9)
    angular_vel_positive_color: tuple[float, ...] = (0.8, 0.2, 0.8, 0.9)
    angular_vel_negative_color: tuple[float, ...] = (0.2, 0.8, 0.8, 0.9)

    # Actual velocity arrow, drawn just below the command arrow
    show_actual_vel: bool = True
    actual_vel_z_offset: float = -0.05

    # Height of the arrows above the base
    z_offset: float = 0.3

    # Yaw-rate arrow: vertical, up for a positive (counter-clockwise) command, down for a negative one
    show_angular_vel: bool = True
    angular_vel_threshold: float = 0.05
    angular_vel_z_offset: float = 0.15


class CommandArrowOverlay(Base3DOverlay):
    """
    Overlay for visualizing velocity commands as 3D arrows.

    Renders, per rendered environment:
    - Commanded planar velocity: arrow in the commanded direction (world frame)
    - Measured planar velocity: a second arrow just below it
    - Commanded yaw rate: a vertical arrow, up or down by the sign of the command
    """

    def __init__(self, scene: "gs.Scene", config: CommandArrowConfig | None = None):
        config = config or CommandArrowConfig()
        super().__init__(scene, config)
        self.config: CommandArrowConfig = config

    def update(self, state: OverlayState) -> None:
        self.clear()
        if not self.enabled or state.base_pos is None or state.base_quat is None or state.cmd_lin_vel is None:
            return

        pos = state.base_pos
        yaw = _yaw_from_quat_wxyz(state.base_quat)
        self._draw_planar_arrow(pos, state.cmd_lin_vel, yaw, self.config.linear_vel_color, z_extra_offset=0.0)
        if self.config.show_actual_vel and state.actual_lin_vel is not None:
            self._draw_planar_arrow(
                pos,
                state.actual_lin_vel[:, :2],
                yaw,
                self.config.actual_vel_color,
                z_extra_offset=self.config.actual_vel_z_offset,
            )
        if self.config.show_angular_vel and state.cmd_ang_vel is not None:
            self._draw_yaw_rate_arrows(pos, state.cmd_ang_vel)

    def _arrow_length(self, magnitude: torch.Tensor) -> torch.Tensor:
        return (magnitude * self.config.arrow_length_scale).clamp(max=self.config.max_arrow_length)

    def _draw_planar_arrow(
        self,
        pos: torch.Tensor,
        vel_body: torch.Tensor,
        yaw: torch.Tensor,
        color: tuple[float, ...],
        z_extra_offset: float,
    ) -> None:
        """Draw a body-frame planar velocity as a world-frame arrow above the base."""
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        vel_x, vel_y = vel_body.unbind(-1)
        vel_world = torch.stack((cos_yaw * vel_x - sin_yaw * vel_y, sin_yaw * vel_x + cos_yaw * vel_y), dim=-1)
        magnitude = torch.linalg.norm(vel_world, dim=-1, keepdim=True)
        # A zero command keeps a zero vector, which the marker leaves out for that environment
        direction = torch.where(magnitude > 1e-6, vel_world / magnitude.clamp(min=1e-6), 0.0)
        vec = torch.zeros_like(pos)
        vec[:, :2] = direction * self._arrow_length(magnitude)
        arrow_pos = pos.clone()
        arrow_pos[:, 2] += self.config.z_offset + z_extra_offset
        self.draw_arrows(arrow_pos, vec, self.config.arrow_radius, color)

    def _draw_yaw_rate_arrows(self, pos: torch.Tensor, ang_vel: torch.Tensor) -> None:
        """Draw the yaw-rate command as a vertical arrow, one color per sign."""
        ang_vel = torch.where(ang_vel.abs() > self.config.angular_vel_threshold, ang_vel, 0.0)
        arrow_pos = pos.clone()
        arrow_pos[:, 2] += self.config.z_offset + self.config.angular_vel_z_offset
        length = self._arrow_length(ang_vel.abs())
        for sign, color in (
            (1.0, self.config.angular_vel_positive_color),
            (-1.0, self.config.angular_vel_negative_color),
        ):
            vec = torch.zeros_like(pos)
            vec[:, 2] = torch.where(ang_vel * sign > 0, sign * length, 0.0)
            self.draw_arrows(arrow_pos, vec, self.config.arrow_radius, color)
