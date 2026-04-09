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

# Type alias for (lower, upper) joint limit tuple
JointLimitsTuple = "tuple[Tensor, Tensor]"


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
    def joint_pos_limits(self) -> "tuple[Tensor, Tensor]":
        """Hard joint position limits in canonical actuated order.

        Returns ``(lower, upper)``, each of shape ``(num_joints,)``. These
        are the *hard* limits as stored in the simulator's model — apply
        any soft-limit factor in the consumer.

        Implementations that don't natively expose hard limits (e.g. mjlab,
        which only stores soft limits) may raise ``NotImplementedError``.
        """
        ...

    # ── Body-level reads (named bodies / links) ──────────────────────

    def find_body_index(self, body_name: str) -> int:
        """Resolve a body/link name to its sim-internal integer index.

        Each simulator stores bodies in its own indexing space:
        - Newton: index into ``body_cache`` (per-env body slot)
        - Genesis: ``link.idx_local``
        - mjlab: index returned by ``entity.find_bodies([name])``

        Args:
            body_name: Exact body name (no regex). For Newton this may
                include the entity prefix (e.g. ``"g1_29dof/torso_link"``).
                For Genesis and mjlab this is typically the bare body name.

        Returns:
            Integer index suitable for passing to ``body_*_w`` accessors.
        """
        ...

    def body_ang_vel_w(self, body_index: int) -> Tensor:
        """World-frame angular velocity of a single body.

        Args:
            body_index: Index returned by :meth:`find_body_index`.

        Returns:
            Tensor of shape ``(num_envs, 3)`` — angular velocity in
            world frame, x/y/z order.
        """
        ...
