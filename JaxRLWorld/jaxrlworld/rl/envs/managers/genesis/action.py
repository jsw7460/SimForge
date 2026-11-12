from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jaxrlworld.rl.envs.managers.common.action import (
    ActionManagerBase,
    ActionManagerBaseConfig,
)

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import GenesisEnv


@dataclass
class ActionManagerConfig(ActionManagerBaseConfig):
    """Genesis-specific action manager configuration."""

    pass


class ActionManager(ActionManagerBase):
    """Genesis action manager.

    Hands Genesis the entity's :class:`GenesisDofSelection`: the control
    targets go in simulator DOF order through the ``range`` form, which
    takes Genesis's zero-copy path (an index tensor would cost a host
    synchronisation per call, every substep).
    """

    def __init__(self, env: GenesisEnv, config: ActionManagerConfig):
        self._genesis_env = env
        super().__init__(env, config)

    def _apply_position(self, targets, entity_name):
        entity = self._genesis_env.scene_manager.entities[entity_name]
        dofs = self._genesis_env.dof_selection(entity_name)
        entity.control_dofs_position(dofs.to_sim_order(targets), dofs.dofs)

    def _apply_force(self, torques, entity_name):
        entity = self._genesis_env.scene_manager.entities[entity_name]
        dofs = self._genesis_env.dof_selection(entity_name)
        entity.control_dofs_force(dofs.to_sim_order(torques), dofs.dofs)

    @property
    def actuated_dof_ids(self):
        """DOF indices (alias for indexing.sim_indices)."""
        return self._indexing.sim_indices
