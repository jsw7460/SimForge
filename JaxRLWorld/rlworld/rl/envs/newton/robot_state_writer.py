"""NewtonRobotStateWriter — write API for Newton entity state.

Mirror of the read-only :class:`NewtonRobotData` (``robot_data.py``).
The writer holds the same ``ArticulationView`` reference but only
exposes mutation methods used by event terms and reset functions:

- joint coordinate / velocity writes
- root pose + root velocity writes
- forward kinematics evaluation

Each write method takes an explicit Newton ``State`` object so that
callers can mutate either the active state or a staged state. A
boolean ``mask`` (``wp.array`` or ``torch.Tensor``) optionally limits
the write to a subset of worlds; pass ``None`` to write to all worlds.

This class is intentionally **sim-specific** — Newton's
``ArticulationView`` write protocol does not generalize across
backends. Genesis and mjlab write paths look entirely different and
will get their own writer classes if they need a unified call site.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from torch import Tensor

if TYPE_CHECKING:
    from newton.selection import ArticulationView
    from newton import State
    from rlworld.rl.envs.newton.newton_env import NewtonEnv


class NewtonRobotStateWriter:
    """Write-side companion to :class:`NewtonRobotData`."""

    def __init__(self, env: "NewtonEnv", view: "ArticulationView") -> None:
        self._env = env
        self._view = view

    # ------------------------------------------------------------------
    # Joint writes
    # ------------------------------------------------------------------

    def set_dof_positions(self, state: "State", values: Tensor, mask=None) -> None:
        """Set joint coordinate positions. ``values`` shape: ``(W, joint_coord_count)``."""
        wp_arr = wp.from_torch(values.unsqueeze(1).contiguous(), dtype=wp.float32)
        self._view.set_dof_positions(state, wp_arr, mask=self._resolve_mask(mask))

    def set_dof_velocities(self, state: "State", values: Tensor, mask=None) -> None:
        """Set joint coordinate velocities. ``values`` shape: ``(W, joint_dof_count)``."""
        wp_arr = wp.from_torch(values.unsqueeze(1).contiguous(), dtype=wp.float32)
        self._view.set_dof_velocities(state, wp_arr, mask=self._resolve_mask(mask))

    # ------------------------------------------------------------------
    # Root state writes
    # ------------------------------------------------------------------

    def set_root_state(
        self,
        state: "State",
        pos: Tensor,
        quat_xyzw: Tensor,
        lin_vel: Tensor | None = None,
        ang_vel: Tensor | None = None,
        mask=None,
    ) -> None:
        """Set root transforms and (optionally) root velocities.

        Args:
            state: Newton ``State`` object to mutate.
            pos: ``(W, 3)`` root position.
            quat_xyzw: ``(W, 4)`` root quaternion in **xyzw** (Newton native).
            lin_vel: ``(W, 3)`` linear velocity, optional. Pass alongside
                ``ang_vel`` to write root velocities; pass ``None`` for both
                to leave velocities untouched.
            ang_vel: ``(W, 3)`` angular velocity, optional.
            mask: World mask (``wp.array`` or ``torch.Tensor``). ``None``
                writes all worlds.
        """
        resolved_mask = self._resolve_mask(mask)

        transform = torch.cat([pos, quat_xyzw], dim=-1)
        wp_t = wp.from_torch(transform.unsqueeze(1).contiguous(), dtype=wp.transform)
        self._view.set_root_transforms(state, wp_t, mask=resolved_mask)

        if lin_vel is not None and ang_vel is not None:
            vel = torch.cat([lin_vel, ang_vel], dim=-1)
            wp_v = wp.from_torch(vel.unsqueeze(1).contiguous(), dtype=wp.spatial_vector)
            self._view.set_root_velocities(state, wp_v, mask=resolved_mask)

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def eval_fk(self, state: "State", mask=None) -> None:
        """Evaluate forward kinematics for the selected worlds."""
        self._view.eval_fk(state, mask=self._resolve_mask(mask))

    # ------------------------------------------------------------------
    # Mask utility
    # ------------------------------------------------------------------

    @staticmethod
    def env_ids_to_mask(
        env_ids: torch.Tensor, num_worlds: int, device: torch.device
    ) -> "wp.array":
        """Convert integer env_ids to a boolean warp mask of shape ``(num_worlds,)``."""
        mask = torch.zeros(num_worlds, dtype=torch.bool, device=device)
        mask[env_ids] = True
        return wp.from_torch(mask)

    def _resolve_mask(self, mask):
        if mask is None:
            return None
        if isinstance(mask, wp.array):
            return mask
        if isinstance(mask, torch.Tensor):
            return wp.from_torch(mask)
        return mask
