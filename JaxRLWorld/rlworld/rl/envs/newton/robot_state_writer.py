"""NewtonRobotStateWriter — write API for Newton entity state.

Implements :class:`RobotStateWriterProtocol` against Newton's
``ArticulationView`` + warp double-buffered ``State``. The protocol
hides Newton-specific quirks from callers:

- Quaternions are accepted in **wxyz** (protocol convention) and
  converted to Newton's native **xyzw** layout internally.
- ``env_ids`` is a torch tensor; the writer builds the corresponding
  warp boolean mask.
- The active state (``scene_manager.state``, which is ``state_0``)
  is grabbed automatically — callers do not pass an explicit state.
- ``values`` for joint / root writes is the **subset** for the given
  ``env_ids``; the writer reads the current full tensor and splices
  the subset into it before calling warp's masked API. When
  ``env_ids`` is ``None`` the caller passes a full ``(num_envs, ...)``
  tensor and no read-modify-write is needed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from torch import Tensor

if TYPE_CHECKING:
    from newton.selection import ArticulationView
    from rlworld.rl.envs.newton.newton_env import NewtonEnv


class NewtonRobotStateWriter:
    """Write-side companion to :class:`NewtonRobotData`."""

    def __init__(self, env: "NewtonEnv", view: "ArticulationView") -> None:
        self._env = env
        self._view = view
        # Cache actuated joint index mappings for splice operations
        self._q_indices = env.act_manager.actuated_q_indices
        self._qd_indices = env.act_manager.actuated_qd_indices

    # ------------------------------------------------------------------
    # Joint writes
    # ------------------------------------------------------------------

    def set_dof_positions(
        self, values: Tensor, env_ids: "Tensor | None" = None
    ) -> None:
        """Write actuated joint positions.

        ``values`` has shape ``(N, num_actuated)`` — the writer reads
        the full joint_q tensor, splices actuated values into the
        correct coordinate indices, and writes the merged tensor back.
        """
        full = self._build_full_dof_tensor(
            values, env_ids, self._view.get_dof_positions, self._q_indices
        )
        wp_arr = wp.from_torch(full.unsqueeze(1).contiguous(), dtype=wp.float32)
        self._view.set_dof_positions(self._state, wp_arr, mask=self._mask(env_ids))

    def set_dof_velocities(
        self, values: Tensor, env_ids: "Tensor | None" = None
    ) -> None:
        """Write actuated joint velocities.

        Same splice logic as :meth:`set_dof_positions` but for the
        velocity (joint_qd) tensor using ``actuated_qd_indices``.
        """
        full = self._build_full_dof_tensor(
            values, env_ids, self._view.get_dof_velocities, self._qd_indices
        )
        wp_arr = wp.from_torch(full.unsqueeze(1).contiguous(), dtype=wp.float32)
        self._view.set_dof_velocities(self._state, wp_arr, mask=self._mask(env_ids))

    # ------------------------------------------------------------------
    # Root writes
    # ------------------------------------------------------------------

    def set_root_pose(
        self,
        pos: Tensor,
        quat_wxyz: Tensor,
        env_ids: "Tensor | None" = None,
    ) -> None:
        """Write root link position + orientation (wxyz)."""
        # wxyz → xyzw (Newton native)
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]

        full_pos = self._splice_root(
            pos, env_ids, self._read_root_pos
        )
        full_quat = self._splice_root(
            quat_xyzw, env_ids, self._read_root_quat_xyzw
        )

        transform = torch.cat([full_pos, full_quat], dim=-1)
        wp_t = wp.from_torch(transform.unsqueeze(1).contiguous(), dtype=wp.transform)
        self._view.set_root_transforms(self._state, wp_t, mask=self._mask(env_ids))

    def set_root_velocity(
        self,
        lin_vel: Tensor,
        ang_vel: Tensor,
        env_ids: "Tensor | None" = None,
    ) -> None:
        """Write root link linear + angular velocity."""
        full_lin = self._splice_root(
            lin_vel, env_ids, self._read_root_lin_vel
        )
        full_ang = self._splice_root(
            ang_vel, env_ids, self._read_root_ang_vel
        )

        vel = torch.cat([full_lin, full_ang], dim=-1)
        wp_v = wp.from_torch(vel.unsqueeze(1).contiguous(), dtype=wp.spatial_vector)
        self._view.set_root_velocities(self._state, wp_v, mask=self._mask(env_ids))

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def eval_fk(self, env_ids: "Tensor | None" = None) -> None:
        """Re-evaluate forward kinematics for the selected environments."""
        self._view.eval_fk(self._state, mask=self._mask(env_ids))

    # ==================================================================
    # Internals
    # ==================================================================

    @property
    def _state(self):
        return self._env.scene_manager.state

    # -- value helpers --------------------------------------------------

    def _build_full_dof_tensor(
        self,
        values: Tensor,
        env_ids: "Tensor | None",
        getter,
        actuated_indices: "Tensor | None" = None,
    ) -> Tensor:
        """Read current full DOF tensor, splice actuated values in, return full.

        When ``actuated_indices`` is provided, ``values`` has shape
        ``(N, num_actuated)`` and gets written into the columns
        specified by ``actuated_indices`` within the ``(num_envs,
        full_dof_count)`` tensor.
        """
        current = wp.to_torch(getter(self._state)).squeeze(1).clone()
        if actuated_indices is not None:
            if env_ids is not None:
                current[env_ids.unsqueeze(1), actuated_indices.unsqueeze(0)] = values
            else:
                current[:, actuated_indices] = values
        else:
            if env_ids is not None:
                current[env_ids] = values
            else:
                current[:] = values
        return current

    def _splice_root(
        self,
        values: Tensor,
        env_ids: "Tensor | None",
        reader,
    ) -> Tensor:
        if env_ids is None:
            return values
        current = reader().clone()
        current[env_ids] = values
        return current

    # -- root readers (used only by the splice path) --------------------

    def _root_transform_floats(self) -> Tensor:
        wp_arr = self._view.get_root_transforms(self._state)
        return wp.to_torch(wp_arr).reshape(-1, 7)

    def _root_velocity_floats(self) -> Tensor:
        wp_arr = self._view.get_root_velocities(self._state)
        return wp.to_torch(wp_arr).reshape(-1, 6)

    def _read_root_pos(self) -> Tensor:
        return self._root_transform_floats()[:, 0:3]

    def _read_root_quat_xyzw(self) -> Tensor:
        return self._root_transform_floats()[:, 3:7]

    def _read_root_lin_vel(self) -> Tensor:
        return self._root_velocity_floats()[:, 0:3]

    def _read_root_ang_vel(self) -> Tensor:
        return self._root_velocity_floats()[:, 3:6]

    # -- mask -----------------------------------------------------------

    def _mask(self, env_ids: "Tensor | None"):
        if env_ids is None:
            return None
        num_worlds = self._env.scene_manager.model.world_count
        mask = torch.zeros(num_worlds, dtype=torch.bool, device=self._env.device)
        mask[env_ids] = True
        return wp.from_torch(mask)
