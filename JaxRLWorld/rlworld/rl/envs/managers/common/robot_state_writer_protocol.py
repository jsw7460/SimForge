"""RobotStateWriterProtocol — unified write API across simulator backends.

Companion to the read-only :class:`RobotData` protocol. Each backend
provides a concrete writer class (``NewtonRobotStateWriter``,
``GenesisRobotStateWriter``, ``MujocoRobotStateWriter``) that satisfies
this protocol structurally — no inheritance is required.

The unified shape lets future sim-agnostic event / reset terms call

    writer = env.get_robot_state_writer()
    writer.set_root_pose(pos, quat_wxyz, env_ids=env_ids)
    writer.set_dof_velocities(zeros, env_ids=env_ids)

without branching on simulator type. Each backend implementation hides
its sim-specific quirks (Newton's explicit ``state_0`` swap + warp
mask, Genesis treating root velocity as the first 6 DOFs, mjlab's
``write_*_to_sim`` API and required pos+vel pairs).

Conventions
-----------

**Subset values.** ``set_dof_positions`` / ``set_dof_velocities`` /
``set_root_pose`` / ``set_root_velocity`` all take tensors whose first
dimension matches ``len(env_ids)`` (not ``num_envs``). Pass
``env_ids=None`` for the full-update case, in which case the first
dimension must equal ``num_envs``. Newton's writer internally reads
the current full state, splices in the subset, and writes the merged
tensor through warp's masked API; Genesis and mjlab pass the subset
straight to their native APIs.

**wxyz quaternion.** ``set_root_pose`` always receives a wxyz
quaternion. Newton's writer flips it to its native xyzw layout
internally; Genesis and mjlab use wxyz natively.

**Pose vs velocity split.** The old ``set_root_state`` was split into
``set_root_pose`` (positions + quaternions) and ``set_root_velocity``
(linear + angular). The split mirrors mjlab's
``write_root_link_pose_to_sim`` / ``write_root_link_velocity_to_sim``
and lets callers update only what they need without re-passing the
unchanged half.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class RobotStateWriterProtocol(Protocol):
    """Minimum sim-agnostic write contract for robot state."""

    def set_dof_positions(
        self,
        values: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write joint positions for the actuated DOFs.

        Args:
            values: Tensor of shape ``(N, num_joints)`` where ``N``
                equals ``len(env_ids)`` (or ``num_envs`` if ``env_ids``
                is ``None``).
            env_ids: Subset of environment indices to update. ``None``
                writes all environments.
        """
        ...

    def set_dof_velocities(
        self,
        values: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write joint velocities for the actuated DOFs."""
        ...

    def set_root_pose(
        self,
        pos: torch.Tensor,
        quat_wxyz: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write the root link pose (position + orientation).

        Args:
            pos: ``(N, 3)`` root position in world frame.
            quat_wxyz: ``(N, 4)`` root quaternion, **wxyz** convention.
                Backends that use a different native order (Newton:
                xyzw) convert internally.
            env_ids: Subset of environments to update. ``None`` writes
                all environments.
        """
        ...

    def set_root_velocity(
        self,
        lin_vel: torch.Tensor,
        ang_vel: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write the root link velocity (linear + angular)."""
        ...

    def eval_fk(self, env_ids: torch.Tensor | None = None) -> None:
        """Re-evaluate forward kinematics for the selected environments.

        Newton needs an explicit FK pass after a root / DOF write
        because its solver uses double-buffered state. Genesis and
        mjlab manage FK internally and implement this as a no-op.
        """
        ...
