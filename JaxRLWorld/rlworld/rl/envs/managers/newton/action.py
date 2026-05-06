from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from rlworld.rl.envs.managers.common.action import (
    ActionManagerBase,
    ActionManagerBaseConfig,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class NewtonActionManagerConfig(ActionManagerBaseConfig):
    """Newton-specific action manager configuration."""

    pass


class NewtonActionManager(ActionManagerBase):
    """Newton action manager.

    Uses ArticulationIndexing for all index mappings.
    newton_qd_indices is used for _apply_position/_apply_force.
    newton_q_indices is used by NewtonRobotData for joint_pos.
    """

    def __init__(self, env: World, config: NewtonActionManagerConfig):
        super().__init__(env, config)

    def _apply_position(self, targets: torch.Tensor) -> None:
        """Apply position targets via Newton/Warp."""
        control = self.env.scene_manager.control
        model = self.env.scene_manager.model

        num_worlds = model.world_count
        dof_per_world = model.joint_dof_count // num_worlds

        full_targets = torch.zeros(
            (num_worlds, dof_per_world),
            device=self.device,
            dtype=torch.float32,
        )
        full_targets[:, self._indexing.newton_qd_indices] = targets

        wp.copy(
            control.joint_target_pos,
            wp.from_torch(full_targets.flatten(), dtype=wp.float32, requires_grad=False),
        )

    def _apply_force(self, torques: torch.Tensor) -> None:
        """Apply torques directly via Newton/Warp."""
        control = self.env.scene_manager.control
        model = self.env.scene_manager.model

        num_worlds = model.world_count
        dof_per_world = model.joint_dof_count // num_worlds

        full_forces = torch.zeros(
            (num_worlds, dof_per_world),
            device=self.device,
            dtype=torch.float32,
        )
        full_forces[:, self._indexing.newton_qd_indices] = torques

        wp.copy(
            control.joint_f,
            wp.from_torch(full_forces.flatten(), dtype=wp.float32, requires_grad=False),
        )

    # -- Backward compat properties ------------------------------------------

    @property
    def actuated_q_indices(self) -> torch.Tensor:
        return self._indexing.newton_q_indices

    @property
    def actuated_qd_indices(self) -> torch.Tensor:
        return self._indexing.newton_qd_indices
