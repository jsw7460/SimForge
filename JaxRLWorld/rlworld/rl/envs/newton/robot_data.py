"""NewtonRobotData — thin wrapper that satisfies the RobotData protocol.

Lazily computes each property from Newton's warp state tensors.
Newton uses **xyzw** quaternions, so we reorder to wxyz for the protocol.
Newton velocities are in **world frame**, so we rotate to body frame.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
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

    def _get_state_tensors(self):
        """Get joint_q and joint_qd as reshaped torch tensors."""
        sm = self._env.scene_manager
        state = sm.state
        num_worlds = sm.model.world_count

        joint_q = wp.to_torch(state.joint_q)
        joint_qd = wp.to_torch(state.joint_qd)

        coords_per_world = joint_q.numel() // num_worlds
        dofs_per_world = joint_qd.numel() // num_worlds

        joint_q = joint_q.reshape(num_worlds, coords_per_world)
        joint_qd = joint_qd.reshape(num_worlds, dofs_per_world)

        return joint_q, joint_qd

    @property
    def root_link_pos_w(self) -> Tensor:
        joint_q, _ = self._get_state_tensors()
        return joint_q[:, 0:3]

    @property
    def root_link_quat_w(self) -> Tensor:
        """Quaternion in wxyz. Newton stores xyzw at joint_q[:, 3:7]."""
        joint_q, _ = self._get_state_tensors()
        quat_xyzw = joint_q[:, 3:7]
        return quat_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz

    @property
    def root_link_lin_vel_b(self) -> Tensor:
        joint_q, joint_qd = self._get_state_tensors()
        quat_wxyz = joint_q[:, [6, 3, 4, 5]]  # xyzw -> wxyz
        lin_vel_w = joint_qd[:, 0:3]
        return quat_rotate_inverse_wxyz(quat_wxyz, lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> Tensor:
        joint_q, joint_qd = self._get_state_tensors()
        quat_wxyz = joint_q[:, [6, 3, 4, 5]]  # xyzw -> wxyz
        ang_vel_w = joint_qd[:, 3:6]
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
        joint_q, _ = self._get_state_tensors()
        return joint_q[:, self._env.act_manager.actuated_q_indices]

    @property
    def joint_vel(self) -> Tensor:
        _, joint_qd = self._get_state_tensors()
        return joint_qd[:, self._env.act_manager.actuated_qd_indices]
