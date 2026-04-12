"""NewtonRobotStateWriter — zero-copy write API for Newton entity state.

Implements :class:`RobotStateWriterProtocol` using **zero-copy torch
views** of Newton's warp state arrays.  ``wp.to_torch()`` returns a
tensor that shares the underlying GPU memory with the warp array, so
writing to the torch view directly mutates the simulator state — no
conversion overhead.

The views are created once at construction time and reused on every
call.  This mirrors the pattern used by Newton's own RL examples
(``newton/solvers/kamino/examples/rl/simulation.py``).

Conventions
-----------

**wxyz quaternion.** ``set_root_pose`` accepts wxyz and converts to
Newton's native xyzw before writing.

**Actuated-only values.** ``set_dof_positions`` / ``set_dof_velocities``
accept tensors of shape ``(N, num_actuated)`` and write them to the
correct generalized-coordinate indices via ``actuated_q_indices`` /
``actuated_qd_indices``.

**eval_fk.** Must be called after joint/root writes to recompute
body transforms (``body_q``) from the updated ``joint_q``.
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
    """Write-side companion to :class:`NewtonRobotData`.

    Uses zero-copy torch views of ``state.joint_q`` and
    ``state.joint_qd`` for maximum write performance.
    """

    def __init__(self, env: "NewtonEnv", view: "ArticulationView") -> None:
        self._env = env
        self._view = view

        # Actuated joint index mappings
        self._q_indices = env.act_manager.actuated_q_indices
        self._qd_indices = env.act_manager.actuated_qd_indices

        # Zero-copy torch views of the warp state arrays.
        # These share GPU memory — torch writes update warp directly.
        model = env.scene_manager.model
        num_worlds = model.world_count
        coords_per_world = model.joint_coord_count // num_worlds
        dofs_per_world = model.joint_dof_count // num_worlds
        state = env.scene_manager.state

        self._joint_q = wp.to_torch(state.joint_q).reshape(num_worlds, coords_per_world)
        self._joint_qd = wp.to_torch(state.joint_qd).reshape(num_worlds, dofs_per_world)

    # ------------------------------------------------------------------
    # Joint writes
    # ------------------------------------------------------------------

    def set_dof_positions(
        self, values: Tensor, env_ids: "Tensor | None" = None
    ) -> None:
        """Write actuated joint positions via zero-copy view."""
        if env_ids is not None:
            self._joint_q[env_ids.unsqueeze(1), self._q_indices.unsqueeze(0)] = values
        else:
            self._joint_q[:, self._q_indices] = values

    def set_dof_velocities(
        self, values: Tensor, env_ids: "Tensor | None" = None
    ) -> None:
        """Write actuated joint velocities via zero-copy view."""
        if env_ids is not None:
            self._joint_qd[env_ids.unsqueeze(1), self._qd_indices.unsqueeze(0)] = values
        else:
            self._joint_qd[:, self._qd_indices] = values

    # ------------------------------------------------------------------
    # Root writes
    # ------------------------------------------------------------------

    def set_root_pose(
        self,
        pos: Tensor,
        quat_wxyz: Tensor,
        env_ids: "Tensor | None" = None,
    ) -> None:
        """Write root link position + orientation (wxyz → xyzw)."""
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
        rows = env_ids if env_ids is not None else slice(None)
        self._joint_q[rows, 0:3] = pos
        self._joint_q[rows, 3:7] = quat_xyzw

    def set_root_velocity(
        self,
        lin_vel: Tensor,
        ang_vel: Tensor,
        env_ids: "Tensor | None" = None,
    ) -> None:
        """Write root link linear + angular velocity."""
        rows = env_ids if env_ids is not None else slice(None)
        self._joint_qd[rows, 0:3] = lin_vel
        self._joint_qd[rows, 3:6] = ang_vel

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def eval_fk(self, env_ids: "Tensor | None" = None) -> None:
        """Re-evaluate forward kinematics for the selected environments."""
        self._view.eval_fk(self._env.scene_manager.state, mask=self._mask(env_ids))

    # ==================================================================
    # Internals
    # ==================================================================

    def _mask(self, env_ids: "Tensor | None"):
        if env_ids is None:
            return None
        num_worlds = self._env.scene_manager.model.world_count
        mask = torch.zeros(num_worlds, dtype=torch.bool, device=self._env.device)
        mask[env_ids] = True
        return wp.from_torch(mask)
