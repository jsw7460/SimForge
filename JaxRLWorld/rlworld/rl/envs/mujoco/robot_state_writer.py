"""MujocoRobotStateWriter — write API for an mjlab ``Entity``.

Implements :class:`RobotStateWriterProtocol` against mjlab's
``write_*_to_sim`` API. mjlab uses **wxyz** quaternions natively
(matching the protocol convention) and accepts an ``env_ids`` torch
tensor for per-env scoping, so the writer is mostly a thin shim that
adapts argument layout.

mjlab-specific quirks the writer hides from callers:

- ``write_joint_state_to_sim(joint_pos, joint_vel, env_ids=...)`` is
  the only joint write API and requires **both** position and
  velocity at once. ``set_dof_positions`` reads the current velocity
  for the affected envs and feeds it back through; ``set_dof_velocities``
  does the symmetric thing for position. Callers that need to update
  both should call them in sequence — there is no extra overhead
  beyond two reads.
- Pose / velocity are passed as concatenated 7-vec (pos + quat) and
  6-vec (lin + ang) respectively. ``set_root_pose`` /
  ``set_root_velocity`` build these tensors internally.
- ``eval_fk`` is a no-op: mjlab's ``Simulation.step()`` and
  ``Simulation.forward()`` handle FK internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from mjlab.entity import Entity

    from rlworld.rl.envs.mujoco.mjlab_env import MujocoEnv


class MujocoRobotStateWriter:
    """Write-side companion to :class:`MujocoRobotData`."""

    def __init__(
        self,
        env: MujocoEnv,
        entity: Entity,
        joint_ids: Tensor,
    ) -> None:
        self._env = env
        self._entity = entity
        self._joint_ids = joint_ids

    # ------------------------------------------------------------------
    # Joint writes
    # ------------------------------------------------------------------

    def set_dof_positions(self, values: Tensor, env_ids: Tensor | None = None) -> None:
        """Write actuated joint positions.

        mjlab's ``write_joint_state_to_sim`` requires both pos and vel,
        so we read the current velocity for the affected envs and pass
        it through unchanged.
        """
        env_ids = self._resolve_env_ids(env_ids)
        current_vel = self._entity.data.joint_vel[env_ids][:, self._joint_ids]
        self._entity.write_joint_state_to_sim(
            values,
            current_vel,
            env_ids=env_ids,
            joint_ids=self._joint_ids,
        )

    def set_dof_velocities(self, values: Tensor, env_ids: Tensor | None = None) -> None:
        """Write actuated joint velocities.

        Symmetric to :meth:`set_dof_positions` — reads the current
        position and feeds it back through ``write_joint_state_to_sim``.
        """
        env_ids = self._resolve_env_ids(env_ids)
        current_pos = self._entity.data.joint_pos[env_ids][:, self._joint_ids]
        self._entity.write_joint_state_to_sim(
            current_pos,
            values,
            env_ids=env_ids,
            joint_ids=self._joint_ids,
        )

    # ------------------------------------------------------------------
    # Root writes
    # ------------------------------------------------------------------

    def set_root_pose(
        self,
        pos: Tensor,
        quat_wxyz: Tensor,
        env_ids: Tensor | None = None,
    ) -> None:
        """Write root link pose. mjlab is wxyz native."""
        env_ids = self._resolve_env_ids(env_ids)
        pose = torch.cat([pos, quat_wxyz], dim=-1)
        self._entity.write_root_link_pose_to_sim(pose, env_ids=env_ids)

    def set_root_velocity(
        self,
        lin_vel: Tensor,
        ang_vel: Tensor,
        env_ids: Tensor | None = None,
    ) -> None:
        """Write root link linear + angular velocity."""
        env_ids = self._resolve_env_ids(env_ids)
        vel = torch.cat([lin_vel, ang_vel], dim=-1)
        self._entity.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def eval_fk(self, env_ids: Tensor | None = None) -> None:
        """No-op: mjlab updates kinematics inside ``Simulation.step()``."""
        return None

    # ==================================================================
    # Internals
    # ==================================================================

    def _resolve_env_ids(self, env_ids: Tensor | None) -> Tensor:
        if env_ids is not None:
            return env_ids
        return torch.arange(self._env.num_envs, device=self._env.device)
