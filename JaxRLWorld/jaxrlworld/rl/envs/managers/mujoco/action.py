from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jaxrlworld.rl.envs.managers.common.action import (
    ActionManagerBase,
    ActionManagerBaseConfig,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity

    from jaxrlworld.rl.envs import World


@dataclass
class MujocoActionManagerConfig(ActionManagerBaseConfig):
    """MuJoCo/mjlab-specific action manager configuration."""

    pass


class MujocoActionManager(ActionManagerBase):
    """MuJoCo/mjlab action manager.

    Uses ArticulationIndexing.sim_indices as joint_ids for
    mjlab set_joint_position_target / set_joint_effort_target.
    """

    def __init__(self, env: World, config: MujocoActionManagerConfig):
        self._entity: Entity = env.scene_manager.robot
        super().__init__(env, config)

    def _sim_indices(self, entity_name: str):
        return self.env.entity_indexing(entity_name).sim_indices

    def _apply_position(self, targets, entity_name):
        entity = self.env.scene_manager.entities[entity_name]
        sim_indices = self._sim_indices(entity_name)
        encoder_bias = entity.data.encoder_bias[:, sim_indices]
        entity.set_joint_position_target(targets - encoder_bias, joint_ids=sim_indices)

    def _apply_force(self, torques, entity_name):
        entity = self.env.scene_manager.entities[entity_name]
        entity.set_joint_effort_target(torques, joint_ids=self._sim_indices(entity_name))

    # -- Backward compat --
    @property
    def _joint_ids(self):
        return self._indexing.sim_indices
