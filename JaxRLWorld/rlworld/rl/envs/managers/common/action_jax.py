"""JAX-native base class for action managers.

Drop-in replacement for ActionManagerBase that uses JAX arrays instead of torch.
Used by Newton's JAX-native environment.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World


JOINT_LIMIT_CLIP = "joint_limit"


@dataclass
class ActionManagerBaseConfig:
    """Base configuration for action processing and control."""
    actuated_dof_names: list[str] = field(default_factory=list)
    clip: (
        tuple[float, float]
        | dict[str, tuple[float, float]]
        | Literal["joint_limit"]
        | None
    ) = (-1.0, 1.0)
    scale: float | dict[str, float] = 1.0
    offset: dict[str, float] | None = None
    control_mode: Literal["position", "force"] = "position"


class JaxActionManagerBase(BaseManager):
    """JAX-native base class for action managers.

    Subclasses must implement:
        - _resolve_joints() -> tuple[list[int], list[str]]
        - _get_joint_limits() -> tuple[jax.Array, jax.Array]
        - apply_actions(processed_actions: jax.Array) -> None
    """

    def __init__(self, env: "World", config: ActionManagerBaseConfig):
        super().__init__(env)
        self.config = config

        self._actuated_joint_indices, self._actuated_joint_names = (
            self._resolve_joints()
        )
        self._total_action_dim = len(self._actuated_joint_indices)

        self._raw_actions = jnp.zeros(
            (self.env.num_envs, self._total_action_dim)
        )
        self._processed_actions = jnp.zeros_like(self._raw_actions)
        self._prev_raw_actions = jnp.zeros_like(self._raw_actions)
        self._prev_processed_actions = jnp.zeros_like(self._raw_actions)

        self._offset = self._initialize_offsets()

        self._scale = self._initialize_scale()
        self._clip_low, self._clip_high = self._initialize_clip()

    # ------------------------------------------------------------------
    # Abstract methods (simulator-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def _resolve_joints(self) -> tuple[list[int], list[str]]:
        ...

    @abstractmethod
    def _get_joint_limits(self) -> tuple[jax.Array, jax.Array]:
        ...

    @abstractmethod
    def apply_actions(self, processed_actions: jax.Array) -> None:
        ...

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_scale(self) -> jax.Array:
        scale = jnp.ones(self._total_action_dim)

        if isinstance(self.config.scale, (int, float)):
            scale = jnp.full_like(scale, self.config.scale)
        elif isinstance(self.config.scale, dict):
            indices, _, values = string_utils.resolve_matching_names_values(
                self.config.scale, self._actuated_joint_names
            )
            scale = scale.at[jnp.array(indices)].set(jnp.array(values))

        return scale

    def _initialize_clip(self) -> tuple[jax.Array, jax.Array]:
        clip_low = jnp.full((self._total_action_dim,), -float("inf"))
        clip_high = jnp.full((self._total_action_dim,), float("inf"))

        if self.config.clip is None:
            pass

        elif self.config.clip == JOINT_LIMIT_CLIP:
            if jnp.any(self._scale > 1.0):
                violating = [
                    f"{self._actuated_joint_names[i]} (scale={float(self._scale[i]):.4f})"
                    for i in range(self._total_action_dim)
                    if float(self._scale[i]) > 1.0
                ]
                raise ValueError(
                    f'clip="joint_limit" requires all scale values <= 1.0. '
                    f"Violating joints: {violating}"
                )

            joint_lower, joint_upper = self._get_joint_limits()
            default_pos = self._offset[0]
            clip_low = joint_lower - default_pos
            clip_high = joint_upper - default_pos

        elif isinstance(self.config.clip, (tuple, list)):
            clip_low = jnp.full_like(clip_low, self.config.clip[0])
            clip_high = jnp.full_like(clip_high, self.config.clip[1])

        elif isinstance(self.config.clip, dict):
            clip_dict_low = {k: v[0] for k, v in self.config.clip.items()}
            clip_dict_high = {k: v[1] for k, v in self.config.clip.items()}

            indices, _, low_values = string_utils.resolve_matching_names_values(
                clip_dict_low, self._actuated_joint_names
            )
            _, _, high_values = string_utils.resolve_matching_names_values(
                clip_dict_high, self._actuated_joint_names
            )

            clip_low = clip_low.at[jnp.array(indices)].set(jnp.array(low_values))
            clip_high = clip_high.at[jnp.array(indices)].set(jnp.array(high_values))

        return clip_low, clip_high

    def _initialize_offsets(self) -> jax.Array:
        offset = jnp.zeros((self.env.num_envs, self._total_action_dim))

        if self.config.offset is not None and isinstance(self.config.offset, dict):
            offset_indices, _, offset_values = (
                string_utils.resolve_matching_names_values(
                    self.config.offset, self._actuated_joint_names
                )
            )
            offset = offset.at[:, jnp.array(offset_indices)].set(
                jnp.array(offset_values)
            )

        return offset

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_action_dim(self) -> int:
        return self._total_action_dim

    @property
    def num_actions(self) -> int:
        return self._total_action_dim

    @property
    def offset(self) -> jax.Array:
        return self._offset

    @property
    def actuated_joint_names(self) -> list[str]:
        return self._actuated_joint_names

    @property
    def actuated_joint_indices(self) -> list[int]:
        return self._actuated_joint_indices

    @property
    def raw_actions(self) -> jax.Array:
        return self._raw_actions

    @property
    def processed_actions(self) -> jax.Array:
        return self._processed_actions

    @property
    def prev_raw_actions(self) -> jax.Array:
        return self._prev_raw_actions

    @property
    def prev_processed_actions(self) -> jax.Array:
        return self._prev_processed_actions

    @property
    def clip_bounds(self) -> tuple[float, float] | None:
        if isinstance(self.config.clip, tuple):
            return self.config.clip
        return None

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def process_actions(self, actions: jax.Array) -> jax.Array:
        """Process raw actions: clip -> scale -> offset."""
        self._raw_actions = jnp.array(actions)
        clipped = jnp.clip(actions, self._clip_low, self._clip_high)
        self._processed_actions = clipped * self._scale + self._offset
        return self._processed_actions

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            return
        self._raw_actions = self._raw_actions.at[env_ids].set(0.0)
        self._processed_actions = self._processed_actions.at[env_ids].set(0.0)
        self._prev_raw_actions = self._prev_raw_actions.at[env_ids].set(0.0)
        self._prev_processed_actions = self._prev_processed_actions.at[env_ids].set(0.0)

    def advance(self) -> None:
        self._prev_raw_actions = jnp.array(self._raw_actions)
        self._prev_processed_actions = jnp.array(self._processed_actions)

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        rows = []
        for idx, joint_name in enumerate(self._actuated_joint_names):
            clip_low = float(self._clip_low[idx])
            clip_high = float(self._clip_high[idx])

            if clip_low == float("-inf") and clip_high == float("inf"):
                clip_str = "[-inf, inf]"
            else:
                clip_str = f"[{clip_low:.1f}, {clip_high:.1f}]"

            scale_str = f"{float(self._scale[idx]):.4f}"

            offset_val = float(self._offset[0, idx])
            offset_str = f"{offset_val:.2f}" if offset_val != 0 else "0.0"

            rows.append([idx, joint_name, clip_str, scale_str, offset_str])

        table = create_manager_table(
            title="Action Space",
            columns=["Idx", "Joint", "Clip Range", "Scale", "Offset"],
            rows=rows,
            footer=f"Total: {self._total_action_dim} dims",
        )
        return table_to_string(table)
