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

    def set_dof_state(self, positions: Tensor, velocities: Tensor, env_ids: Tensor | None = None) -> None:
        """Write joint positions and velocities in one native call.

        ``write_joint_state_to_sim`` wants the pair anyway; passing both
        halves skips the two current-state gathers and one of the two
        full writes the split setters would perform.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self._entity.write_joint_state_to_sim(
            positions,
            velocities,
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
        """Write root link pose. mjlab is wxyz native.

        A welded entity has no free joint, so its pose does not live in
        ``qpos`` and ``write_root_link_pose_to_sim`` refuses it. mjlab's answer
        is to wrap every fixed-base entity in a ``mocap_base`` body
        (``mjlab/utils/spec.py``), whose pose lives in ``data.mocap_pos`` —
        per-environment state, which is what makes a table placeable per env at
        all. Route there, as mjlab's own reset events do.
        """
        env_ids = self._resolve_env_ids(env_ids)
        pose = torch.cat([pos, quat_wxyz], dim=-1)
        if self._entity.is_fixed_base:
            if not self._entity.is_mocap:
                raise ValueError(
                    "Cannot write root pose for a fixed-base entity without a mocap base. mjlab wraps "
                    "fixed-base entities in one automatically, so this entity was built from a spec that "
                    "already had a non-mocap root body."
                )
            self._entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)
            return
        self._entity.write_root_link_pose_to_sim(pose, env_ids=env_ids)

    def set_root_velocity(
        self,
        lin_vel: Tensor,
        ang_vel: Tensor,
        env_ids: Tensor | None = None,
    ) -> None:
        """Write root link linear + angular velocity.

        Raises:
            ValueError: If the entity is welded to the world. A fixed base has
                no root velocity to write — the same refusal Genesis and Newton
                give, so a preset that tries fails identically everywhere.
        """
        env_ids = self._resolve_env_ids(env_ids)
        if self._entity.is_fixed_base:
            raise ValueError(
                "Cannot write root velocity for fixed-base entity: it is welded to the world. Its pose "
                "can still be set per environment (mocap), but it has no velocity state."
            )
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
