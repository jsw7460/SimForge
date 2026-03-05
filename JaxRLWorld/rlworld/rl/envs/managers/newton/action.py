from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from rlworld.rl.envs.managers.common.action_jax import (
    JaxActionManagerBase,
    ActionManagerBaseConfig,
)
import newton
from rlworld.rl.envs.utils.warp_jax_utils import wp_to_jax, wp_from_jax
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class NewtonActionManagerConfig(ActionManagerBaseConfig):
    """Newton-specific action manager configuration."""
    num_actions: int | None = None
    default_joint_pos: list[float] | None = None


class NewtonActionManager(JaxActionManagerBase):
    """Newton action manager (JAX-native).

    Extends ActionManagerBase with Newton/Warp-specific joint resolution,
    joint-limit queries via model.joint_limit_lower/upper, and Warp-based
    position control. All tensor operations use JAX arrays.
    """

    def __init__(self, env: "World", config: NewtonActionManagerConfig):
        self._newton_config = config
        self._model = env.scene_manager.model

        self._all_joint_names = self._get_joint_names(self._model)

        resolved_indices, resolved_names = self._resolve_joints()

        # Validate: check matched joints have DOFs
        joint_q_start = np.array(wp_to_jax(self._model.joint_q_start))
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

        joint_qd_start = np.array(wp_to_jax(self._model.joint_qd_start))
        self._actuated_qd_indices = jnp.array(
            [int(joint_qd_start[j]) for j in resolved_indices],
        )

        self._actuated_q_indices = jnp.array(
            [int(joint_q_start[j]) for j in resolved_indices],
        )

        super().__init__(env, config)

        # Handle legacy default_joint_pos
        if config.default_joint_pos is not None and config.offset is None:
            default_pos = jnp.array(config.default_joint_pos)
            self._offset = jnp.broadcast_to(
                jnp.expand_dims(default_pos, 0), self._offset.shape
            )
            self._clip_low, self._clip_high = self._initialize_clip()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _resolve_joints(self) -> tuple[list[int], list[str]]:
        """Resolve joints from Newton model joint names."""
        config = self._newton_config

        if config.actuated_dof_names:
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
            original_indices = [actuatable_indices[i] for i in matched_indices]
            return original_indices, matched_names

        elif config.num_actions is not None:
            num = config.num_actions
            indices = list(range(num))
            names = self._all_joint_names[1 : 1 + num]
            return indices, names

        raise ValueError(
            "Must provide either 'actuated_dof_names' or 'num_actions' in config"
        )

    def _get_joint_limits(self) -> tuple[jax.Array, jax.Array]:
        """Get joint limits from Newton model."""
        dofs_per_world = (
            self._model.joint_dof_count // self._model.world_count
        )
        lower_all = wp_to_jax(self._model.joint_limit_lower)[:dofs_per_world]
        upper_all = wp_to_jax(self._model.joint_limit_upper)[:dofs_per_world]

        lower = lower_all[self._actuated_qd_indices]
        upper = upper_all[self._actuated_qd_indices]
        return lower, upper

    def apply_actions(self, processed_actions: jax.Array) -> None:
        """Apply processed actions via Newton/Warp position control."""
        scene_manager = self.env.scene_manager
        control = scene_manager.control
        model = scene_manager.model

        num_worlds = model.world_count
        dof_per_world = model.joint_dof_count // num_worlds

        full_targets = jnp.zeros(
            (num_worlds, dof_per_world),
            dtype=jnp.float32,
        )

        full_targets = full_targets.at[:, self._actuated_qd_indices].set(processed_actions)

        targets_flat = full_targets.flatten()
        wp.copy(
            control.joint_target_pos,
            wp_from_jax(targets_flat, dtype=wp.float32),
        )

    # ------------------------------------------------------------------
    # Newton-specific properties
    # ------------------------------------------------------------------

    @property
    def actuated_q_indices(self) -> jax.Array:
        """joint_q array indices for actuated joints."""
        return self._actuated_q_indices

    @property
    def actuated_qd_indices(self) -> jax.Array:
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

        return all_joint_names[:joints_per_world]
