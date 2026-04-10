"""GenesisRobotStateWriter — write API for a Genesis ``RigidEntity``.

Implements :class:`RobotStateWriterProtocol` against Genesis's native
``entity.set_*`` mutation methods. Genesis hides simulation state
inside ``gs.Scene`` and uses ``envs_idx`` (a torch tensor of integer
env indices) to scope per-env writes — much closer to the unified
protocol shape than Newton's masked warp arrays.

Genesis-specific quirks the writer hides from callers:

- Root velocity is not a separate API call — Genesis treats the base
  6 DOFs (x, y, z, roll, pitch, yaw) as plain DOFs that you write
  through ``set_dofs_velocity`` with ``dofs_idx_local=list(range(6))``.
  ``set_root_velocity`` packs the linear + angular vectors into a
  single 6-vec and forwards.
- ``eval_fk`` is a no-op: Genesis re-evaluates kinematics inside its
  next ``scene.step()``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    import torch
    from genesis.engine.entities import RigidEntity
    from rlworld.rl.envs.genesis.genesis_env import GenesisEnv


class GenesisRobotStateWriter:
    """Write-side companion to :class:`GenesisRobotData`."""

    def __init__(
        self,
        env: "GenesisEnv",
        entity: "RigidEntity",
        actuated_dof_ids: "Tensor | list[int]",
    ) -> None:
        self._env = env
        self._entity = entity
        self._actuated_dof_ids = actuated_dof_ids

    # ------------------------------------------------------------------
    # Joint writes
    # ------------------------------------------------------------------

    def set_dof_positions(
        self, values: Tensor, env_ids: "torch.Tensor | None" = None
    ) -> None:
        """Write actuated joint positions."""
        self._entity.set_dofs_position(
            position=values,
            dofs_idx_local=self._actuated_dof_ids,
            envs_idx=env_ids,
        )

    def set_dof_velocities(
        self, values: Tensor, env_ids: "torch.Tensor | None" = None
    ) -> None:
        """Write actuated joint velocities."""
        self._entity.set_dofs_velocity(
            velocity=values,
            dofs_idx_local=self._actuated_dof_ids,
            envs_idx=env_ids,
        )

    # ------------------------------------------------------------------
    # Root writes
    # ------------------------------------------------------------------

    def set_root_pose(
        self,
        pos: Tensor,
        quat_wxyz: Tensor,
        env_ids: "torch.Tensor | None" = None,
    ) -> None:
        """Write root link position + orientation. Genesis is wxyz native."""
        self._entity.set_pos(pos, envs_idx=env_ids)
        self._entity.set_quat(quat_wxyz, envs_idx=env_ids)

    def set_root_velocity(
        self,
        lin_vel: Tensor,
        ang_vel: Tensor,
        env_ids: "torch.Tensor | None" = None,
    ) -> None:
        """Write root link linear + angular velocity.

        Genesis has no dedicated root-velocity setter; root velocity is
        the first 6 DOFs (3 linear + 3 angular). Pack and forward.
        """
        import torch
        combined = torch.cat([lin_vel, ang_vel], dim=-1)
        self._entity.set_dofs_velocity(
            velocity=combined,
            dofs_idx_local=list(range(6)),
            envs_idx=env_ids,
        )

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def eval_fk(self, env_ids: "torch.Tensor | None" = None) -> None:
        """No-op: Genesis updates kinematics during ``scene.step()``."""
        return None
