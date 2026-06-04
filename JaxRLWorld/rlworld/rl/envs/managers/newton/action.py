from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
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

    Uses ArticulationIndexing for all index mappings. Position targets are
    written into ``joint_target_q``; under the legacy DOF layout this matches
    the ``joint_qd`` stride so ``newton_qd_indices`` is the right indexer,
    but under the coord layout (Newton PR #2965) the same array follows the
    ``joint_q`` stride and ``newton_q_indices`` must be used instead. Forces
    are always ``joint_qd``-strided, so ``_apply_force`` keeps
    ``newton_qd_indices`` unconditionally.
    """

    def __init__(self, env: World, config: NewtonActionManagerConfig):
        super().__init__(env, config)

    def _apply_position(self, targets: torch.Tensor) -> None:
        """Apply position targets via Newton/Warp."""
        control = self.env.scene_manager.control
        model = self.env.scene_manager.model

        num_worlds = model.world_count
        # Under the coord layout introduced in Newton PR #2965, ``joint_target_q``
        # follows the ``joint_q`` stride (7 slots for a FREE joint) and is indexed
        # by ``newton_q_indices``; under the legacy DOF layout it follows
        # ``joint_qd`` (6 slots for a FREE joint) and is indexed by
        # ``newton_qd_indices``. The FREE joint quaternion slot needs identity
        # init under coord layout so a fresh zero buffer does not corrupt it.
        if newton.use_coord_layout_targets:
            slots_per_world = model.joint_coord_count // num_worlds
            write_indices = self._indexing.newton_q_indices
        else:
            slots_per_world = model.joint_dof_count // num_worlds
            write_indices = self._indexing.newton_qd_indices

        full_targets = torch.zeros(
            (num_worlds, slots_per_world),
            device=self.device,
            dtype=torch.float32,
        )
        if newton.use_coord_layout_targets:
            # FREE joint coord order is [x, y, z, qx, qy, qz, qw] — set qw = 1.
            full_targets[:, 6] = 1.0
        full_targets[:, write_indices] = targets

        wp.copy(
            control.joint_target_q,
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
