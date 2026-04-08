"""RobotData Protocol — unified read-only interface for robot state.

Property names match mjlab's ``EntityData`` exactly so that mjlab satisfies
the protocol with zero adapter code.  Genesis and Newton provide thin
wrapper classes (see ``genesis/robot_data.py`` and ``newton/robot_data.py``)
that lazily compute each property from their native APIs.

All quaternions are **wxyz**.  All velocities labelled ``_b`` are in the
**body frame**.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class RobotData(Protocol):
    """Minimal robot state readable by any simulator backend."""

    @property
    def root_link_pos_w(self) -> Tensor:
        """Root link position in world frame. Shape (num_envs, 3)."""
        ...

    @property
    def root_link_quat_w(self) -> Tensor:
        """Root link quaternion in world frame (wxyz). Shape (num_envs, 4)."""
        ...

    @property
    def root_link_lin_vel_b(self) -> Tensor:
        """Root link linear velocity in body frame. Shape (num_envs, 3)."""
        ...

    @property
    def root_link_ang_vel_b(self) -> Tensor:
        """Root link angular velocity in body frame. Shape (num_envs, 3)."""
        ...

    @property
    def projected_gravity_b(self) -> Tensor:
        """Gravity vector projected into body frame. Shape (num_envs, 3)."""
        ...

    @property
    def heading_w(self) -> Tensor:
        """Heading angle (yaw) in world frame. Shape (num_envs,)."""
        ...

    @property
    def joint_pos(self) -> Tensor:
        """Actuated joint positions. Shape (num_envs, num_joints)."""
        ...

    @property
    def joint_vel(self) -> Tensor:
        """Actuated joint velocities. Shape (num_envs, num_joints)."""
        ...

    @property
    def joint_torque(self) -> Tensor:
        """Most-recently-applied actuator torques on actuated joints.

        Returned in the same joint order as ``joint_pos`` and ``joint_vel``.
        Shape ``(num_envs, num_joints)``.

        Implementations should expose the torques actually applied by the
        physics solver in the previous step (i.e. after PD-controller and
        actuator-network processing), not the raw policy actions.
        """
        ...
