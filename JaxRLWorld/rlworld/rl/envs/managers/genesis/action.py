from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlworld.rl.envs.managers.common.action import (
    ActionManagerBase,
    ActionManagerBaseConfig,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


@dataclass
class ActionManagerConfig(ActionManagerBaseConfig):
    """Genesis-specific action manager configuration."""

    pass


class ActionManager(ActionManagerBase):
    """Genesis action manager.

    Uses ArticulationIndexing.sim_indices as DOF indices for
    Genesis control_dofs_position / control_dofs_force.
    """

    def __init__(self, env: "GenesisEnv", config: ActionManagerConfig):
        self._genesis_env = env
        super().__init__(env, config)

    def _apply_position(self, targets):
        self._genesis_env.robot.control_dofs_position(
            targets, self._indexing.sim_indices
        )

    def _apply_force(self, torques):
        self._genesis_env.robot.control_dofs_force(
            torques, self._indexing.sim_indices
        )

    @property
    def actuated_dof_ids(self):
        """DOF indices (alias for indexing.sim_indices)."""
        return self._indexing.sim_indices
