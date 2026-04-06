"""NewtonRobotData — thin wrapper that satisfies the RobotData protocol.

Uses RobotStateAccessor (backed by ArticulationView) to access Newton state.
Newton uses **xyzw** quaternions, so we reorder to wxyz for the protocol.
Newton velocities are in **world frame**, so we rotate to body frame.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, quat_to_euler_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs.newton.newton_env import NewtonEnv


class NewtonRobotData:
    """RobotData implementation for Newton environments."""

    def __init__(self, env: NewtonEnv) -> None:
        self._env = env
        self._gravity_vec: Tensor | None = None

    def _get_gravity_vec(self) -> Tensor:
        if self._gravity_vec is None:
            self._gravity_vec = torch.tensor(
                [[0.0, 0.0, -1.0]],
                device=self._env.device,
                dtype=torch.float32,
            ).expand(self._env.num_envs, -1).contiguous()
        return self._gravity_vec

    @property
    def _accessor(self):
        return self._env.scene_manager.robot_state

    @property
    def _state(self):
        return self._env.scene_manager.state

    @property
    def root_link_pos_w(self) -> Tensor:
        return self._accessor.root_pos_w(self._state)

    @property
    def root_link_quat_w(self) -> Tensor:
        """Quaternion in wxyz."""
        return self._accessor.root_quat_wxyz(self._state)

    @property
    def root_link_lin_vel_b(self) -> Tensor:
        quat_wxyz = self._accessor.root_quat_wxyz(self._state)
        lin_vel_w = self._accessor.root_lin_vel_w(self._state)
        return quat_rotate_inverse_wxyz(quat_wxyz, lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> Tensor:
        quat_wxyz = self._accessor.root_quat_wxyz(self._state)
        ang_vel_w = self._accessor.root_ang_vel_w(self._state)
        return quat_rotate_inverse_wxyz(quat_wxyz, ang_vel_w)

    @property
    def projected_gravity_b(self) -> Tensor:
        quat = self.root_link_quat_w
        return quat_rotate_inverse_wxyz(quat, self._get_gravity_vec())

    @property
    def heading_w(self) -> Tensor:
        euler = quat_to_euler_wxyz(self.root_link_quat_w)
        return euler[:, 2]

    @property
    def joint_pos(self) -> Tensor:
        dof_pos = self._accessor.dof_positions(self._state)
        return dof_pos[:, self._env.act_manager.indexing.newton_q_indices]

    @property
    def joint_vel(self) -> Tensor:
        dof_vel = self._accessor.dof_velocities(self._state)
        return dof_vel[:, self._env.act_manager.indexing.newton_qd_indices]
