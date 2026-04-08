"""GenesisRobotData — thin wrapper that satisfies the RobotData protocol.

Lazily computes each property from the Genesis ``RigidEntity`` API.
Genesis uses **wxyz** quaternions natively, so no reordering is needed.
Genesis velocities are in **world frame**, so we rotate to body frame.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, quat_to_euler_wxyz

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


class GenesisRobotData:
    """RobotData implementation for a Genesis RigidEntity."""

    def __init__(
        self,
        entity: RigidEntity,
        actuated_dof_ids: Tensor | list[int],
        num_envs: int,
        device: torch.device,
    ) -> None:
        self._entity = entity
        self._actuated_dof_ids = actuated_dof_ids
        self._gravity_vec: Tensor | None = None
        self._num_envs = num_envs
        self._device = device

    def _get_gravity_vec(self) -> Tensor:
        if self._gravity_vec is None:
            self._gravity_vec = torch.tensor(
                [[0.0, 0.0, -1.0]],
                device=self._device,
                dtype=torch.float32,
            ).expand(self._num_envs, -1).contiguous()
        return self._gravity_vec

    @property
    def root_link_pos_w(self) -> Tensor:
        return self._entity.get_pos()

    @property
    def root_link_quat_w(self) -> Tensor:
        return self._entity.get_quat()  # already wxyz

    @property
    def root_link_lin_vel_b(self) -> Tensor:
        quat = self.root_link_quat_w
        vel_w = self._entity.get_vel()
        return quat_rotate_inverse_wxyz(quat, vel_w)

    @property
    def root_link_ang_vel_b(self) -> Tensor:
        quat = self.root_link_quat_w
        ang_w = self._entity.get_ang()
        return quat_rotate_inverse_wxyz(quat, ang_w)

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
        return self._entity.get_dofs_position(self._actuated_dof_ids)

    @property
    def joint_vel(self) -> Tensor:
        return self._entity.get_dofs_velocity(self._actuated_dof_ids)
