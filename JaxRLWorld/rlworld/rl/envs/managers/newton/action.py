from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from rlworld.rl.envs.managers.common.action import (
    ActionManagerBase,
    ActionManagerBaseConfig,
)
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class NewtonActionManagerConfig(ActionManagerBaseConfig):
    """Newton-specific action manager configuration.

    Attributes:
        num_actions: Legacy support — use num_actions directly instead of
            actuated_dof_names. Deprecated; prefer actuated_dof_names.
        default_joint_pos: Legacy support — list-based default positions.
            Deprecated; prefer offset dict.
    """

    num_actions: int | None = None
    default_joint_pos: list[float] | None = None


class NewtonActionManager(ActionManagerBase):
    """Newton action manager.

    Extends ActionManagerBase with Newton/Warp-specific joint resolution,
    joint-limit queries via model.joint_limit_lower/upper, and Warp-based
    position control.

    Additional properties beyond base class:
        - actuated_q_indices: joint_q array indices for actuated joints
        - actuated_qd_indices: joint_qd array indices for actuated joints
    """

    def __init__(self, env: "World", config: NewtonActionManagerConfig):
        self._newton_config = config
        self._model = env.scene_manager.model

        # Extract joint names before super().__init__
        self._all_joint_names = self._get_joint_names(self._model)

        # Resolve joints early to compute qd/q indices before super().__init__,
        # since _initialize_clip may call _get_joint_limits which needs them.
        resolved_indices, resolved_names = self._resolve_joints()

        # Validate: check matched joints have DOFs
        joint_q_start = wp.to_torch(self._model.joint_q_start).cpu().numpy()
        invalid_joints = []
        for idx, name in zip(resolved_indices, resolved_names):
            dof_count = joint_q_start[idx + 1] - joint_q_start[idx]
            if dof_count == 0:
                invalid_joints.append(name)
        if invalid_joints:
            raise ValueError(
                f"actuated_dof_names matched joints with 0 DOF (fixed joints): "
                f"{invalid_joints}\n"
                f"Only joints with DOF > 0 can be actuated."
            )

        # Compute qd and q indices (needed by _get_joint_limits during init)
        joint_qd_start = wp.to_torch(self._model.joint_qd_start).cpu().numpy()
        self._actuated_qd_indices = torch.tensor(
            [int(joint_qd_start[j]) for j in resolved_indices],
            device=env.device,
        )

        self._actuated_q_indices = torch.tensor(
            [int(joint_q_start[j]) for j in resolved_indices],
            device=env.device,
        )

        super().__init__(env, config)

        # Handle legacy default_joint_pos (overrides offset if provided)
        if config.default_joint_pos is not None and config.offset is None:
            default_pos = torch.tensor(
                config.default_joint_pos, device=self.device
            )
            self._offset[:] = default_pos.unsqueeze(0)
            # Recompute clip bounds since offset changed
            self._clip_low, self._clip_high = self._initialize_clip()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _resolve_joints(self) -> tuple[list[int], list[str]]:
        """Resolve joints from Newton model joint names."""
        config = self._newton_config

        if config.actuated_dof_names:
            # Filter out floating_base (not actuatable)
            actuatable = [
                (i, name)
                for i, name in enumerate(self._all_joint_names)
                if name != "floating_base"
            ]
            actuatable_names = [name for _, name in actuatable]
            actuatable_indices = [i for i, _ in actuatable]

            matched_indices, matched_names = string_utils.resolve_matching_names(
                config.actuated_dof_names, actuatable_names, preserve_order=True
            )
            # Map back to original indices
            original_indices = [actuatable_indices[i] for i in matched_indices]

            return original_indices, matched_names

        elif config.num_actions is not None:
            # Legacy mode
            num = config.num_actions
            indices = list(range(num))
            # Skip floating_base
            names = self._all_joint_names[1: 1 + num]
            return indices, names

        raise ValueError(
            "Must provide either 'actuated_dof_names' or 'num_actions' in config"
        )

    def _get_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint limits from Newton model."""
        dofs_per_world = (
            self._model.joint_dof_count // self._model.world_count
        )
        lower_all = wp.to_torch(self._model.joint_limit_lower)[:dofs_per_world]
        upper_all = wp.to_torch(self._model.joint_limit_upper)[:dofs_per_world]

        lower = lower_all[self._actuated_qd_indices]
        upper = upper_all[self._actuated_qd_indices]
        return lower, upper

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
        full_targets[:, self._actuated_qd_indices] = targets

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
        full_forces[:, self._actuated_qd_indices] = torques

        wp.copy(
            control.joint_f,
            wp.from_torch(full_forces.flatten(), dtype=wp.float32, requires_grad=False),
        )

    # ------------------------------------------------------------------
    # Newton-specific properties
    # ------------------------------------------------------------------

    @property
    def actuated_q_indices(self) -> torch.Tensor:
        """joint_q array indices for actuated joints."""
        return self._actuated_q_indices

    @property
    def actuated_qd_indices(self) -> torch.Tensor:
        """joint_qd array indices for actuated joints."""
        return self._actuated_qd_indices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_joint_names(model) -> list[str]:
        """Extract joint names from Newton model (single world only)."""
        joint_names = getattr(model, "joint_label", None) or getattr(model, "joint_key", None)
        if not joint_names:
            return []

        all_joint_names = list(joint_names)
        num_worlds = model.world_count
        joints_per_world = len(all_joint_names) // num_worlds

        return all_joint_names[:joints_per_world]  # full name 유지
