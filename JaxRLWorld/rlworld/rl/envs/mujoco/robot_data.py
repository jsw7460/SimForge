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

    @property
    def joint_pos_limits(self) -> "tuple[Tensor, Tensor]":
        """Hard joint position limits — not exposed by mjlab.

        mjlab only stores the *soft* limits (already scaled by the soft
        limit factor) in ``entity.data.soft_joint_pos_limits`` with shape
        ``(num_envs, num_joints, 2)``. There is no separate hard-limit
        accessor.

        Phase D-1 only migrates Newton + Genesis ``joint_pos_limits_mjlab``,
        so this stub is never called from active code paths. MuJoCo's
        ``joint_pos_limits`` reward function (in
        ``mdp/rewards/mujoco/reward_terms.py``) reads
        ``soft_joint_pos_limits`` directly and is unchanged.

        Raises:
            NotImplementedError: Always. See note above for the alternative.
        """
        raise NotImplementedError(
            "MujocoRobotData does not expose hard joint position limits. "
            "mjlab only stores soft limits via "
            "``entity.data.soft_joint_pos_limits``. Use mjlab's "
            "``joint_pos_limits`` reward function in "
            "``mdp/rewards/mujoco/reward_terms.py`` instead."
        )

    # ------------------------------------------------------------------
    # Body-level reads
    # ------------------------------------------------------------------

    def find_body_index(self, body_name: str) -> int:
        """Resolve a body name to mjlab's body index.

        Calls ``entity.find_bodies([body_name])`` which returns a tuple
        ``(body_ids: list[int], body_names: list[str])``. We return the
        first index. mjlab's name→index map is precomputed at scene
        compile time, so this lookup is cheap.
        """
        body_ids, _ = self._entity.find_bodies([body_name], preserve_order=True)
        if not body_ids:
            raise ValueError(
                f"Body name {body_name!r} not found in mjlab entity"
            )
        return body_ids[0]

    def body_ang_vel_w(self, body_index: int) -> Tensor:
        """World-frame angular velocity of a single body.

        Reads mjlab's pre-computed ``entity.data.body_link_ang_vel_w``
        which already has shape ``(num_envs, num_bodies, 3)``.
        """
        return self._entity.data.body_link_ang_vel_w[:, body_index, :]
