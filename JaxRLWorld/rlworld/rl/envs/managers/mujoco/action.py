from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlworld.rl.envs.managers.common.action import (
    ActionManagerBase,
    ActionManagerBaseConfig,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from mjlab.entity import Entity


@dataclass
class MjlabActionManagerConfig(ActionManagerBaseConfig):
    """MuJoCo/mjlab-specific action manager configuration."""

    pass


class MjlabActionManager(ActionManagerBase):
    """MuJoCo/mjlab action manager.

    Uses ArticulationIndexing.sim_indices as joint_ids for
    mjlab set_joint_position_target / set_joint_effort_target.
    """

    def __init__(self, env: "World", config: MjlabActionManagerConfig):
        self._entity: "Entity" = env.scene_manager.robot
        super().__init__(env, config)

    def _apply_position(self, targets):
        encoder_bias = self._entity.data.encoder_bias[:, self._indexing.sim_indices]
        target = targets - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self._indexing.sim_indices)

    def _apply_force(self, torques):
        self._entity.set_joint_effort_target(torques, joint_ids=self._indexing.sim_indices)

    # -- Backward compat --
    @property
    def _joint_ids(self):
        return self._indexing.sim_indices
