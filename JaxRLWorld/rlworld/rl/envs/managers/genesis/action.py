from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.common.action import (
    ActionManagerBase,
    ActionManagerBaseConfig,
)
from rlworld.rl.utils import entity_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


@dataclass
class ActionManagerConfig(ActionManagerBaseConfig):
    """Genesis-specific action manager configuration.

    Inherits all fields from ActionManagerBaseConfig.
    """

    pass


class ActionManager(ActionManagerBase):
    """Genesis action manager.

    Extends ActionManagerBase with Genesis-specific joint resolution
    and DOF-based position control.

    Additional properties beyond base class:
        - actuated_dof_ids: DOF indices (may differ from joint indices
          for multi-DOF joints)
    """

    def __init__(self, env: "GenesisEnv", config: ActionManagerConfig):
        self._genesis_env = env

        # Compute DOF indices before super().__init__,
        # since _initialize_clip may call _get_joint_limits which needs them.
        _actuated_dofs, _ = entity_utils.find_dofs(
            entity=self._genesis_env.scene_manager.robot,
            name_keys=config.actuated_dof_names,
            # preserve_order=True
        )
        self._actuated_dofs = torch.tensor(_actuated_dofs, device=env.device)

        super().__init__(env, config)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _resolve_joints(self) -> tuple[list[int], list[str]]:
        """Resolve joints using Genesis entity_utils."""
        dof_ids, joint_names = entity_utils.find_dofs(
            entity=self._genesis_env.scene_manager.robot,
            name_keys=self.config.actuated_dof_names,
        )
        return dof_ids, joint_names

    def _get_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint limits from Genesis entity."""
        entity = self._genesis_env.scene_manager.robot
        dof_lower, dof_upper = entity.get_dofs_limit(
            dofs_idx_local=self._actuated_dofs
        )
        # get_dofs_limit returns (num_envs, num_dofs) — use first env row
        return dof_lower[0], dof_upper[0]

    def _apply_position(self, targets: torch.Tensor) -> None:
        """Set DOF position targets on the Genesis robot."""
        self._genesis_env.robot.control_dofs_position(
            targets, self.actuated_dof_ids
        )

    def _apply_force(self, torques: torch.Tensor) -> None:
        """Set DOF force (torque) on the Genesis robot."""
        self._genesis_env.robot.control_dofs_force(
            torques, self.actuated_dof_ids
        )

    # ------------------------------------------------------------------
    # Genesis-specific properties
    # ------------------------------------------------------------------

    @property
    def actuated_dof_ids(self) -> torch.Tensor:
        """DOF indices used for Genesis control."""
        return self._actuated_dofs