"""MujocoRobotData — wrapper that reindexes mjlab EntityData to action manager order.

mjlab's ``EntityData`` exposes joint states in MuJoCo's internal joint
definition order, which may differ from the action manager's actuated
joint order.  This wrapper applies ``joint_ids`` (from the action
manager) so that :attr:`joint_pos` and :attr:`joint_vel` are aligned
with the action/observation ordering used by the rest of JaxRLWorld.

All other properties (root pose, velocities, gravity) are forwarded
from the underlying ``EntityData`` without reindexing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, quat_to_euler_wxyz

if TYPE_CHECKING:
    from mjlab.entity import Entity


class MujocoRobotData:
    """RobotData implementation for MuJoCo/mjlab entities."""

    def __init__(
        self,
        entity: Entity,
        joint_ids: Tensor,
        num_envs: int,
        device: torch.device,
    ) -> None:
        self._entity = entity
        self._joint_ids = joint_ids
        self._gravity_vec: Tensor | None = None
        self._num_envs = num_envs
        self._device = device

    def _get_gravity_vec(self) -> Tensor:
        """Lazily create gravity vector matching current batch size."""
        # Use quat batch size to handle eval env with different num_envs
        n = self._entity.data.root_link_quat_w.shape[0]
        if self._gravity_vec is None or self._gravity_vec.shape[0] != n:
            self._gravity_vec = torch.tensor(
                [[0.0, 0.0, -1.0]],
                device=self._device,
                dtype=torch.float32,
            ).expand(n, -1).contiguous()
        return self._gravity_vec

    @property
    def root_link_pos_w(self) -> Tensor:
        return self._entity.data.root_link_pos_w

    @property
    def root_link_quat_w(self) -> Tensor:
        return self._entity.data.root_link_quat_w

    @property
    def root_link_lin_vel_b(self) -> Tensor:
        return self._entity.data.root_link_lin_vel_b

    @property
    def root_link_ang_vel_b(self) -> Tensor:
        return self._entity.data.root_link_ang_vel_b

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
        """Actuated joint positions in action manager order."""
        return self._entity.data.joint_pos[:, self._joint_ids]

    @property
    def joint_vel(self) -> Tensor:
        """Actuated joint velocities in action manager order."""
        return self._entity.data.joint_vel[:, self._joint_ids]
