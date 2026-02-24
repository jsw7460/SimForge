from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.common.action import (
    ActionManagerBase,
    ActionManagerBaseConfig,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from mjlab.entity import Entity


@dataclass
class MjlabActionManagerConfig(ActionManagerBaseConfig):
    """MuJoCo/mjlab-specific action manager configuration.

    Attributes:
        entity_name: Name of the entity to control.
    """

    entity_name: str = "robot"


class MjlabActionManager(ActionManagerBase):
    """MuJoCo/mjlab action manager.

    Extends ActionManagerBase with mjlab-specific joint resolution,
    joint-limit queries, and position target control with encoder bias.
    """

    def __init__(self, env: "World", config: MjlabActionManagerConfig):
        self._mjlab_config = config
        self._entity: "Entity" = env.scene_manager.get_entity(config.entity_name)

        # Resolve joint IDs before super().__init__,
        # since _initialize_clip may call _get_joint_limits which needs them.
        if config.actuated_dof_names:
            indices, _ = self._entity.find_joints(
                config.actuated_dof_names, preserve_order=True
            )
        else:
            indices = list(range(len(self._entity.joint_names)))

        self._joint_ids = torch.tensor(
            indices, device=env.device, dtype=torch.long
        )

        super().__init__(env, config)

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _resolve_joints(self) -> tuple[list[int], list[str]]:
        """Resolve joints using mjlab Entity.find_joints."""
        if self._mjlab_config.actuated_dof_names:
            indices, names = self._entity.find_joints(
                self._mjlab_config.actuated_dof_names, preserve_order=True
            )
            return indices, names

        # Fallback: all joints
        all_names = list(self._entity.joint_names)
        return list(range(len(all_names))), all_names

    def _get_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint limits from mjlab entity."""
        soft_limits = self._entity.data.soft_joint_pos_limits
        # soft_joint_pos_limits shape: (num_envs, num_joints, 2)
        lower = soft_limits[0, self._joint_ids, 0]
        upper = soft_limits[0, self._joint_ids, 1]
        return lower, upper

    def apply_actions(self, processed_actions: torch.Tensor) -> None:
        """Apply processed actions to mjlab Entity as joint position targets."""
        encoder_bias = self._entity.data.encoder_bias[:, self._joint_ids]
        target = processed_actions - encoder_bias
        self._entity.set_joint_position_target(target, joint_ids=self._joint_ids)