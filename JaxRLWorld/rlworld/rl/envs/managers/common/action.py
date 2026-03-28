"""Common base class for action managers across all simulators.

Provides ActionManagerBaseConfig and ActionManagerBase with shared
action processing logic (clip, scale, offset, buffers, history).
Simulator-specific subclasses implement joint resolution, joint-limit
queries, and action application.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World


JOINT_LIMIT_CLIP = "joint_limit"


@dataclass
class ActionManagerBaseConfig:
    """Base configuration for action processing and control.

    Attributes:
        actuated_dof_names: List of regex patterns to match actuated joint names.
        clip: Clipping bounds for actions. Can be:
            - tuple[float, float]: (min, max) applied to all dimensions
            - dict[str, tuple[float, float]]: per-joint bounds via regex
            - "joint_limit": auto-compute from joint limits and default positions
            - None: no clipping
        scale: Scaling factor applied to actions after clipping. Can be:
            - float: applied to all dimensions
            - dict[str, float]: per-joint scale via regex
        offset: Dictionary mapping joint name regex patterns to offset values.
            If None, offset is zero for all joints.
    """

    actuated_dof_names: list[str] = field(default_factory=list)
    clip: (
        tuple[float, float]
        | dict[str, tuple[float, float]]
        | Literal["joint_limit"]
        | None
    ) = (-1.0, 1.0)
    scale: float | dict[str, float] = 1.0
    offset: dict[str, float] | None = None


class ActionManagerBase(BaseManager):
    """Base class for action managers across all simulators.

    Subclasses must implement:
        - _resolve_joints() -> tuple[list[int], list[str]]
        - _get_joint_limits() -> tuple[Tensor, Tensor]
        - _apply_position(targets: Tensor) -> None
        - _apply_force(torques: Tensor) -> None

    Processing pipeline: raw_action -> clip -> scale -> offset -> processed_action
    """

    def __init__(self, env: "World", config: ActionManagerBaseConfig):
        super().__init__(env)
        self.config = config

        # Resolve actuated joints (simulator-specific)
        self._actuated_joint_indices, self._actuated_joint_names = (
            self._resolve_joints()
        )
        self._total_action_dim = len(self._actuated_joint_indices)

        # Action buffers
        self._raw_actions = torch.zeros(
            (self.env.num_envs, self._total_action_dim), device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._prev_raw_actions = torch.zeros_like(self._raw_actions)
        self._prev_processed_actions = torch.zeros_like(self._raw_actions)

        # Initialize offset first (needed for joint_limit clip computation)
        self._offset = self._initialize_offsets()

        # Initialize scale and clip bounds
        self._scale = self._initialize_scale()
        self._clip_low, self._clip_high = self._initialize_clip()

        # Build actuator model from entity articulation config
        self._actuator = None
        self._actuator = self._build_actuator_from_entity()

    # ------------------------------------------------------------------
    # Abstract methods (simulator-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def _resolve_joints(self) -> tuple[list[int], list[str]]:
        """Resolve actuated joint indices and names from config patterns.

        Returns:
            Tuple of (joint_indices, joint_names).
        """
        ...

    @abstractmethod
    def _get_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint position limits for actuated joints.

        Returns:
            Tuple of (lower_limits, upper_limits), each shape (num_actuated,).
        """
        ...

    @abstractmethod
    def _apply_position(self, targets: torch.Tensor) -> None:
        """Apply position targets to the simulator (uses simulator PD).

        Args:
            targets: Joint position targets, shape (num_envs, num_actuated).
        """
        ...

    @abstractmethod
    def _apply_force(self, torques: torch.Tensor) -> None:
        """Apply torques directly to simulator joints (bypasses simulator PD).

        Args:
            torques: Joint torques, shape (num_envs, num_actuated).
        """
        ...

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_scale(self) -> torch.Tensor:
        """Initialize per-dimension scale from configuration.

        Returns:
            Tensor of shape (total_action_dim,).
        """
        scale = torch.ones(self._total_action_dim, device=self.device)

        if isinstance(self.config.scale, (int, float)):
            scale[:] = self.config.scale
        elif isinstance(self.config.scale, dict):
            indices, _, values = string_utils.resolve_matching_names_values(
                self.config.scale, self._actuated_joint_names
            )
            scale[indices] = torch.tensor(values, device=self.device)

        return scale

    def _initialize_clip(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize per-dimension clip bounds from configuration.

        Returns:
            Tuple of (clip_low, clip_high), each shape (total_action_dim,).

        Raises:
            ValueError: If clip="joint_limit" and any scale value exceeds 1.0.
        """
        clip_low = torch.full(
            (self._total_action_dim,), -float("inf"), device=self.device
        )
        clip_high = torch.full(
            (self._total_action_dim,), float("inf"), device=self.device
        )

        if self.config.clip is None:
            pass

        elif self.config.clip == JOINT_LIMIT_CLIP:
            # Validate: scale must not exceed 1.0 with joint_limit clip
            if (self._scale > 1.0).any():
                violating = [
                    f"{self._actuated_joint_names[i]} (scale={self._scale[i].item():.4f})"
                    for i in range(self._total_action_dim)
                    if self._scale[i] > 1.0
                ]
                raise ValueError(
                    f'clip="joint_limit" requires all scale values <= 1.0. '
                    f"Violating joints: {violating}"
                )

            joint_lower, joint_upper = self._get_joint_limits()
            # offset shape: (num_envs, num_actuated) — use first env row
            default_pos = self._offset[0]
            clip_low = joint_lower - default_pos
            clip_high = joint_upper - default_pos

        elif isinstance(self.config.clip, (tuple, list)):
            clip_low[:] = self.config.clip[0]
            clip_high[:] = self.config.clip[1]

        elif isinstance(self.config.clip, dict):
            clip_dict_low = {k: v[0] for k, v in self.config.clip.items()}
            clip_dict_high = {k: v[1] for k, v in self.config.clip.items()}

            indices, _, low_values = string_utils.resolve_matching_names_values(
                clip_dict_low, self._actuated_joint_names
            )
            _, _, high_values = string_utils.resolve_matching_names_values(
                clip_dict_high, self._actuated_joint_names
            )

            clip_low[indices] = torch.tensor(low_values, device=self.device)
            clip_high[indices] = torch.tensor(high_values, device=self.device)

        return clip_low, clip_high

    def _initialize_offsets(self) -> torch.Tensor:
        """Initialize action offsets from configuration.

        Returns:
            Tensor of shape (num_envs, total_action_dim).
        """
        offset = torch.zeros(
            (self.env.num_envs, self._total_action_dim), device=self.device
        )

        if self.config.offset is not None and isinstance(self.config.offset, dict):
            offset_indices, _, offset_values = (
                string_utils.resolve_matching_names_values(
                    self.config.offset, self._actuated_joint_names
                )
            )
            offset[:, offset_indices] = torch.tensor(
                offset_values, device=self.device
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
        """Alias for total_action_dim."""
        return self._total_action_dim

    @property
    def offset(self) -> torch.Tensor:
        return self._offset

    @property
    def actuated_joint_names(self) -> list[str]:
        return self._actuated_joint_names

    @property
    def actuated_joint_indices(self) -> list[int]:
        return self._actuated_joint_indices

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def prev_raw_actions(self) -> torch.Tensor:
        return self._prev_raw_actions

    @property
    def prev_processed_actions(self) -> torch.Tensor:
        return self._prev_processed_actions

    @property
    def clip_bounds(self) -> tuple[float, float] | None:
        """Get clip bounds for compatibility with World.action_space."""
        if isinstance(self.config.clip, tuple):
            return self.config.clip
        return None

    # ------------------------------------------------------------------
    # Actuator helpers
    # ------------------------------------------------------------------

    @property
    def actuator(self):
        """The actuator model, or None if not configured."""
        return self._actuator

    def _build_actuator_from_entity(self):
        """Build actuator from the entity's ArticulationCfg.

        Scans the entity config for explicit (non-implicit) actuator
        configs that match the actuated joints.  If found, builds the
        corresponding actuator model.  Returns None if all actuators
        are implicit (simulator PD).
        """
        from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg

        # Find the entity config from the scene manager
        entity_cfg = self._get_entity_cfg()
        if entity_cfg is None:
            return None

        # Collect non-implicit actuator configs that cover our joints
        for act_cfg in entity_cfg.articulation.actuators:
            if isinstance(act_cfg, ImplicitActuatorCfg):
                continue
            # Build the actuator model from this config
            return self._build_actuator(act_cfg)

        return None

    def _get_entity_cfg(self):
        """Get the entity config for the robot from scene manager."""
        scene_mgr = self.env.scene_manager
        entities = getattr(scene_mgr.config if hasattr(scene_mgr, 'config') else scene_mgr, 'entities', None)
        if entities is None:
            entities = getattr(scene_mgr.config, 'entities', None)
        if isinstance(entities, dict):
            # Try "robot" first, then first EntityCfg found
            from rlworld.rl.configs.scene.unified_entity_config import EntityCfg
            if "robot" in entities:
                cfg = entities["robot"]
                if isinstance(cfg, EntityCfg):
                    return cfg
            for cfg in entities.values():
                if isinstance(cfg, EntityCfg):
                    return cfg
        return None

    def _build_actuator(self, cfg):
        """Instantiate an actuator model from its configuration."""
        from rlworld.rl.actuators.actuator_cfg import (
            ActuatorNetLSTMCfg,
            ActuatorNetMLPCfg,
            DCMotorCfg,
            DelayedPDActuatorCfg,
            IdealPDActuatorCfg,
        )
        from rlworld.rl.actuators.actuator_net import ActuatorNetLSTM, ActuatorNetMLP
        from rlworld.rl.actuators.actuator_pd import (
            DCMotor,
            DelayedPDActuator,
            IdealPDActuator,
        )

        cls_map = [
            (ActuatorNetLSTMCfg, ActuatorNetLSTM),
            (ActuatorNetMLPCfg, ActuatorNetMLP),
            (DCMotorCfg, DCMotor),
            (DelayedPDActuatorCfg, DelayedPDActuator),
            (IdealPDActuatorCfg, IdealPDActuator),
        ]
        for cfg_type, actuator_cls in cls_map:
            if isinstance(cfg, cfg_type):
                return actuator_cls(
                    cfg,
                    num_envs=self.env.num_envs,
                    num_joints=self._total_action_dim,
                    device=self.device,
                    joint_names=self._actuated_joint_names,
                )
        raise ValueError(f"Unknown actuator config type: {type(cfg)}")

    def _get_joint_pos(self) -> torch.Tensor:
        """Get current joint positions via the RobotData protocol."""
        return self.env.get_robot_data().joint_pos

    def _get_joint_vel(self) -> torch.Tensor:
        """Get current joint velocities via the RobotData protocol."""
        return self.env.get_robot_data().joint_vel

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def apply_actions(self, processed_actions: torch.Tensor) -> None:
        """Apply processed actions.

        If an explicit actuator model is active (from the entity's
        ArticulationCfg), position targets are converted to torques and
        applied as direct forces.  Otherwise, position targets are sent
        to the simulator's built-in PD controller.

        Args:
            processed_actions: Tensor of shape (num_envs, num_actuated).
        """
        if self._actuator is not None:
            torques = self._actuator.compute(
                target_pos=processed_actions,
                joint_pos=self._get_joint_pos(),
                joint_vel=self._get_joint_vel(),
            )
            self._apply_force(torques)
        else:
            self._apply_position(processed_actions)

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Process raw actions: clip -> scale -> offset.

        Args:
            actions: Raw action tensor of shape (num_envs, total_action_dim).

        Returns:
            Processed action tensor of shape (num_envs, total_action_dim).
        """
        self._raw_actions = actions.clone()
        clipped = torch.clip(actions, self._clip_low, self._clip_high)
        self._processed_actions = clipped * self._scale + self._offset
        return self._processed_actions

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action buffers and actuator state for specified environments."""
        if env_ids is None:
            return
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._prev_raw_actions[env_ids] = 0.0
        self._prev_processed_actions[env_ids] = 0.0
        if self._actuator is not None:
            self._actuator.reset(env_ids)

    def advance(self) -> None:
        """Advance action history by one step."""
        self._prev_raw_actions = self._raw_actions.clone()
        self._prev_processed_actions = self._processed_actions.clone()

    def __str__(self) -> str:
        """Pretty print action manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        rows = []
        for idx, joint_name in enumerate(self._actuated_joint_names):
            clip_low = self._clip_low[idx].item()
            clip_high = self._clip_high[idx].item()

            if clip_low == float("-inf") and clip_high == float("inf"):
                clip_str = "[-inf, inf]"
            else:
                clip_str = f"[{clip_low:.1f}, {clip_high:.1f}]"

            scale_str = f"{self._scale[idx].item():.4f}"

            offset_val = self._offset[0, idx].item()
            offset_str = f"{offset_val:.2f}" if offset_val != 0 else "0.0"

            rows.append([idx, joint_name, clip_str, scale_str, offset_str])

        table = create_manager_table(
            title="Action Space",
            columns=["Idx", "Joint", "Clip Range", "Scale", "Offset"],
            rows=rows,
            footer=f"Total: {self._total_action_dim} dims",
        )
        return table_to_string(table)