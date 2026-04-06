"""RobotStateAccessor — thin adapter around Newton's ArticulationView.

Converts ArticulationView's warp arrays (shaped [W, 1, ...]) into
torch tensors (shaped [W, ...]) with the count_per_world=1 dimension
squeezed out, and handles xyzw ↔ wxyz quaternion reordering.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from torch import Tensor

if TYPE_CHECKING:
    from newton._src.utils.selection import ArticulationView
    from newton import Model, State, Control


class RobotStateAccessor:
    """Adapter that wraps ArticulationView for single-robot-per-world setups."""

    def __init__(self, view: "ArticulationView", device: torch.device) -> None:
        self._view = view
        self._device = device

    # ------------------------------------------------------------------
    # Read helpers (return torch tensors)
    # ------------------------------------------------------------------

    def dof_positions(self, state: "State") -> Tensor:
        """Joint coordinate positions. Shape: (W, joint_coord_count)."""
        wp_arr = self._view.get_dof_positions(state)  # (W, 1, coord_count)
        return wp.to_torch(wp_arr).squeeze(1)

    def dof_velocities(self, state: "State") -> Tensor:
        """Joint coordinate velocities. Shape: (W, joint_dof_count)."""
        wp_arr = self._view.get_dof_velocities(state)  # (W, 1, dof_count)
        return wp.to_torch(wp_arr).squeeze(1)

    def root_pos_w(self, state: "State") -> Tensor:
        """Root position in world frame. Shape: (W, 3)."""
        wp_arr = self._view.get_root_transforms(state)  # (W, 1, 1) wp.transform
        t = wp.to_torch(wp_arr).reshape(-1, 7)  # (W, 7)
        return t[:, 0:3]

    def root_quat_wxyz(self, state: "State") -> Tensor:
        """Root quaternion in wxyz convention. Shape: (W, 4)."""
        wp_arr = self._view.get_root_transforms(state)
        t = wp.to_torch(wp_arr).reshape(-1, 7)
        quat_xyzw = t[:, 3:7]
        return quat_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz

    def root_quat_xyzw(self, state: "State") -> Tensor:
        """Root quaternion in xyzw convention (Newton native). Shape: (W, 4)."""
        wp_arr = self._view.get_root_transforms(state)
        t = wp.to_torch(wp_arr).reshape(-1, 7)
        return t[:, 3:7]

    def root_lin_vel_w(self, state: "State") -> Tensor:
        """Root linear velocity in world frame. Shape: (W, 3)."""
        wp_arr = self._view.get_root_velocities(state)  # (W, 1, 1) wp.spatial_vector
        v = wp.to_torch(wp_arr).reshape(-1, 6)  # (W, 6)
        return v[:, 0:3]

    def root_ang_vel_w(self, state: "State") -> Tensor:
        """Root angular velocity in world frame. Shape: (W, 3)."""
        wp_arr = self._view.get_root_velocities(state)
        v = wp.to_torch(wp_arr).reshape(-1, 6)
        return v[:, 3:6]

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def set_dof_positions(self, state: "State", values: Tensor, mask=None) -> None:
        """Set joint coordinate positions. values shape: (W, joint_coord_count)."""
        wp_arr = wp.from_torch(values.unsqueeze(1).contiguous(), dtype=wp.float32)
        self._view.set_dof_positions(state, wp_arr, mask=self._resolve_mask(mask))

    def set_dof_velocities(self, state: "State", values: Tensor, mask=None) -> None:
        """Set joint coordinate velocities. values shape: (W, joint_dof_count)."""
        wp_arr = wp.from_torch(values.unsqueeze(1).contiguous(), dtype=wp.float32)
        self._view.set_dof_velocities(state, wp_arr, mask=self._resolve_mask(mask))

    def set_root_state(
        self,
        state: "State",
        pos: Tensor,
        quat_xyzw: Tensor,
        lin_vel: Tensor | None = None,
        ang_vel: Tensor | None = None,
        mask=None,
    ) -> None:
        """Set root transforms and optionally velocities.

        Args:
            pos: (W, 3) position
            quat_xyzw: (W, 4) quaternion in xyzw (Newton native)
            lin_vel: (W, 3) linear velocity, optional
            ang_vel: (W, 3) angular velocity, optional
            mask: boolean mask or None
        """
        resolved_mask = self._resolve_mask(mask)

        # Transforms: [pos(3), quat_xyzw(4)] = 7 floats per transform
        transform = torch.cat([pos, quat_xyzw], dim=-1)  # (W, 7)
        wp_t = wp.from_torch(transform.unsqueeze(1).contiguous(), dtype=wp.transform)
        self._view.set_root_transforms(state, wp_t, mask=resolved_mask)

        # Velocities
        if lin_vel is not None and ang_vel is not None:
            vel = torch.cat([lin_vel, ang_vel], dim=-1)  # (W, 6)
            wp_v = wp.from_torch(vel.unsqueeze(1).contiguous(), dtype=wp.spatial_vector)
            self._view.set_root_velocities(state, wp_v, mask=resolved_mask)

    def eval_fk(self, state: "State", mask=None) -> None:
        """Evaluate forward kinematics for selected environments."""
        self._view.eval_fk(state, mask=self._resolve_mask(mask))

    # ------------------------------------------------------------------
    # Mask utility
    # ------------------------------------------------------------------

    @staticmethod
    def env_ids_to_mask(
        env_ids: torch.Tensor, num_worlds: int, device: torch.device
    ) -> "wp.array":
        """Convert integer env_ids to a boolean warp mask of shape (num_worlds,)."""
        mask = torch.zeros(num_worlds, dtype=torch.bool, device=device)
        mask[env_ids] = True
        return wp.from_torch(mask)

    def _resolve_mask(self, mask):
        """Pass through None or wp.array masks; convert torch bool tensors."""
        if mask is None:
            return None
        if isinstance(mask, wp.array):
            return mask
        if isinstance(mask, torch.Tensor):
            return wp.from_torch(mask)
        return mask
