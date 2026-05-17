"""Common base class for action managers across all simulators.

Provides ActionManagerBaseConfig and ActionManagerBase with shared
action processing logic (clip, scale, offset, buffers, history).
Simulator-specific subclasses implement joint resolution, joint-limit
queries, and action application.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from rlworld.rl.actuators.actuator_cfg import (
    ActuatorNetLSTMCfg,
    ActuatorNetMLPCfg,
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
)
from rlworld.rl.actuators.actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from rlworld.rl.actuators.actuator_pd import (
    DCMotor,
    DelayedPDActuator,
    IdealPDActuator,
)
from rlworld.rl.configs.scene.unified_entity_config import EntityCfg
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import string as string_utils
from rlworld.rl.utils.pretty import create_manager_table, table_to_string

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
        settle_steps: Number of steps at the start of each episode during
            which the processed action is overridden to hold the current
            joint position (target = current joint_pos). Used by fall-
            recovery tasks so the robot can physically settle after a
            drop/impact before the policy's output takes effect. ``0``
            disables settling (default — no behavior change for
            existing presets).
    """

    actuated_dof_names: list[str] = field(default_factory=list)
    clip: tuple[float, float] | dict[str, tuple[float, float]] | Literal["joint_limit"] | None = (-1.0, 1.0)
    scale: float | dict[str, float] = 1.0
    offset: dict[str, float] | None = None
    settle_steps: int = 0

    # New term-based action path. When ``action_terms`` is a non-empty
    # dict, the action manager builds each ``ActionTerm`` from the
    # config and routes ``process_actions`` / ``apply_actions``
    # through them. When ``action_terms`` is ``None`` or empty, the
    # legacy monolithic path is used: scale/clip/offset above are
    # applied directly in ``process_actions`` and the term system is
    # inactive. This dual path exists so existing go2/g1 presets
    # keep working unchanged while new tasks (T1 getup, etc.) can
    # declare explicit terms.
    action_terms: dict[str, Any] | None = None


class ActionManagerBase(BaseManager):
    """Base class for action managers across all simulators.

    Subclasses must implement:
        - _apply_position(targets: Tensor) -> None
        - _apply_force(torques: Tensor) -> None

    Processing pipeline: raw_action -> clip -> scale -> offset -> processed_action
    """

    def __init__(self, env: World, config: ActionManagerBaseConfig):
        super().__init__(env)
        self.config = config

        # Build ArticulationIndexing from scene manager
        self._indexing = self._build_indexing()
        self._actuated_joint_names = list(self._indexing.joint_names)
        self._actuated_joint_indices = self._indexing.sim_indices.tolist()
        # Total policy output dim: term-based path sums each term's
        # action_dim; legacy path uses the actuated joint count.
        # Mirrors IsaacLab's ActionManager: terms own their own
        # action_dim, allowing non-joint terms (propeller thrust,
        # body-wrench, etc.) to participate in the action space.
        if config.action_terms:
            self._total_action_dim = sum(self._estimate_term_action_dim(c) for c in config.action_terms.values())
        else:
            self._total_action_dim = self._indexing.num_joints

        # Action history buffers: index 0 = current (t), 1 = t-1, 2 = t-2, ...
        self._action_history_len = 3
        _z = lambda: torch.zeros((self.env.num_envs, self._total_action_dim), device=self.device)
        self._raw_action_history = [_z() for _ in range(self._action_history_len)]
        self._processed_action_history = [_z() for _ in range(self._action_history_len)]

        # Last applied torque (written in ``apply_actions`` when explicit
        # actuator models are active, otherwise remains zero). Exposed via
        # the ``applied_torque`` property for reward/termination terms that
        # need the mechanical power, e.g. getup's energy termination.
        self._applied_torque = _z()

        # Per-env per-joint encoder bias, shape ``(num_envs, action_dim)``.
        # Written by ``randomize_encoder_bias`` (a startup / reset-DR
        # event term) and read by the biased ``dof_pos_biased``
        # observation so the policy sees a calibration-offset version
        # of the joint state. Zero-initialized so this is a no-op until
        # a DR term writes to it.
        self._encoder_bias = _z()

        # Initialize offset first (needed for joint_limit clip computation)
        self._offset = self._initialize_offsets()

        # Initialize scale and clip bounds
        self._scale = self._initialize_scale()
        self._clip_low, self._clip_high = self._initialize_clip()

        # Build per-group actuator models from entity ArticulationCfg.
        # Each actuator handles a subset of joints; implicit actuators are skipped.
        # _actuators: list of (actuator_instance, joint_indices_into_action_dim)
        self._actuators: list[tuple] = []
        self._has_explicit_actuators = False
        self._build_actuators_from_entity()

        # ── Term-based action system (optional) ──────────────────
        # If the preset supplied an ``action_terms`` dict, instantiate
        # each :class:`ActionTerm` and route process/apply through
        # them. Otherwise the legacy monolithic path is used (same as
        # before). See ``rlworld/rl/envs/mdp/actions/`` for the term
        # definitions.
        self._terms: dict[str, Any] = {}
        self._has_action_terms: bool = False
        if config.action_terms:
            for term_name, term_cfg in config.action_terms.items():
                term_class = term_cfg.class_type
                if term_class is None:
                    raise ValueError(f"ActionTermCfg for {term_name!r} has no class_type set — cannot instantiate.")
                self._terms[term_name] = term_class(term_cfg, env=self.env, manager=self)
            self._has_action_terms = len(self._terms) > 0
            # Sanity: total joint ids covered by all terms must equal
            # the action dim (single-term case trivially passes; multi
            # term requires the preset to cover every actuated joint
            # with disjoint slices).
            covered = sum(term.action_dim for term in self._terms.values())
            if covered != self._total_action_dim:
                raise ValueError(
                    f"ActionTerm joint coverage mismatch: terms cover "
                    f"{covered} joints but action_dim is "
                    f"{self._total_action_dim}. All actuated joints "
                    f"must be covered exactly once by the term set."
                )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @property
    def indexing(self):
        """The ArticulationIndexing for this action manager."""
        return self._indexing

    def _build_indexing(self):
        """Build ArticulationIndexing from scene manager."""
        scene_mgr = self.env.scene_manager
        return scene_mgr.build_articulation_indexing(
            actuated_dof_names=self.config.actuated_dof_names,
        )

    def _estimate_term_action_dim(self, term_cfg) -> int:
        """Pre-instantiation estimate of a term's action_dim.

        Called before the term itself is built (we need the total
        action dim to allocate raw/processed buffers). Resolution
        order:

        1. Explicit ``num_actions`` field on the cfg (used by
           non-joint terms like ``PropellerThrustActionCfg`` where
           the action dim isn't derivable from joint names).
        2. Joint-name regex match against the actuated-joint name
           list (matches the JointAction-style flow at
           :meth:`JointAction.__init__`). This must produce the same
           count as ``len(term._joint_ids)`` post-instantiation,
           otherwise the buffer sizes will mismatch and the
           coverage sanity check below will trip.
        """
        explicit = getattr(term_cfg, "num_actions", None)
        if explicit is not None:
            return int(explicit)
        joint_names = getattr(term_cfg, "joint_names", None)
        if joint_names is not None:
            matched, _ = string_utils.resolve_matching_names(
                joint_names, self._actuated_joint_names, preserve_order=True
            )
            return len(matched)
        raise ValueError(
            f"ActionTermCfg {type(term_cfg).__name__} cannot determine "
            f"action_dim: provide either a ``num_actions`` field or a "
            f"``joint_names`` regex list."
        )

    def _get_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint limits from indexing (canonical order)."""
        return self._indexing.joint_limits_lower, self._indexing.joint_limits_upper

    # ------------------------------------------------------------------
    # Abstract methods (simulator-specific)
    # ------------------------------------------------------------------

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

        if isinstance(self.config.scale, int | float):
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
        clip_low = torch.full((self._total_action_dim,), -float("inf"), device=self.device)
        clip_high = torch.full((self._total_action_dim,), float("inf"), device=self.device)

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
                raise ValueError(f'clip="joint_limit" requires all scale values <= 1.0. Violating joints: {violating}')

            joint_lower, joint_upper = self._get_joint_limits()
            # offset shape: (num_envs, num_actuated) — use first env row
            default_pos = self._offset[0]
            clip_low = joint_lower - default_pos
            clip_high = joint_upper - default_pos

        elif isinstance(self.config.clip, tuple | list):
            clip_low[:] = self.config.clip[0]
            clip_high[:] = self.config.clip[1]

        elif isinstance(self.config.clip, dict):
            clip_dict_low = {k: v[0] for k, v in self.config.clip.items()}
            clip_dict_high = {k: v[1] for k, v in self.config.clip.items()}

            indices, _, low_values = string_utils.resolve_matching_names_values(
                clip_dict_low, self._actuated_joint_names
            )
            _, _, high_values = string_utils.resolve_matching_names_values(clip_dict_high, self._actuated_joint_names)

            clip_low[indices] = torch.tensor(low_values, device=self.device)
            clip_high[indices] = torch.tensor(high_values, device=self.device)

        return clip_low, clip_high

    def _initialize_offsets(self) -> torch.Tensor:
        """Initialize action offsets from configuration.

        Returns:
            Tensor of shape (num_envs, total_action_dim).
        """
        offset = torch.zeros((self.env.num_envs, self._total_action_dim), device=self.device)

        if self.config.offset is not None and isinstance(self.config.offset, dict):
            offset_indices, _, offset_values = string_utils.resolve_matching_names_values(
                self.config.offset, self._actuated_joint_names
            )
            offset[:, offset_indices] = torch.tensor(offset_values, device=self.device)

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
    def applied_torque(self) -> torch.Tensor:
        """Torques computed by explicit actuator models in the last step.

        Shape ``(num_envs, total_action_dim)`` in canonical actuated-joint
        order. Only populated when the action config attaches explicit
        actuator models (e.g. ``DelayedPDActuatorCfg``); otherwise stays
        at zeros because the simulator computes PD torques internally and
        they are not routed through Python.
        """
        return self._applied_torque

    @property
    def encoder_bias(self) -> torch.Tensor:
        """Per-env per-joint encoder bias, shape ``(num_envs, action_dim)``.

        Written by ``randomize_encoder_bias`` (typically at startup /
        reset-DR) and read by the biased observation so the policy
        sees a calibration-offset version of the joint state. Zero
        when no DR term has written to it.
        """
        return self._encoder_bias

    def set_encoder_bias(self, bias: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Write the encoder bias tensor for the given envs (or all).

        Used by the ``randomize_encoder_bias`` event term.
        """
        if env_ids is None:
            self._encoder_bias.copy_(bias)
        else:
            self._encoder_bias[env_ids] = bias

    @property
    def actuated_joint_names(self) -> list[str]:
        return self._actuated_joint_names

    @property
    def actuated_joint_indices(self) -> list[int]:
        return self._actuated_joint_indices

    @property
    def raw_action_history(self) -> list[torch.Tensor]:
        """Raw action history: [0] = current (t), [1] = t-1, [2] = t-2, ..."""
        return self._raw_action_history

    @property
    def processed_action_history(self) -> list[torch.Tensor]:
        """Processed action history: [0] = current (t), [1] = t-1, [2] = t-2, ..."""
        return self._processed_action_history

    # Convenience aliases for common access patterns
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_action_history[0]

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_action_history[0]

    @property
    def prev_raw_actions(self) -> torch.Tensor:
        return self._raw_action_history[1]

    @property
    def prev_processed_actions(self) -> torch.Tensor:
        return self._processed_action_history[1]

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
    def actuators(self):
        """List of (actuator, joint_indices) tuples for explicit actuators."""
        return self._actuators

    @property
    def has_explicit_actuators(self) -> bool:
        """True if any non-implicit actuator is configured."""
        return self._has_explicit_actuators

    def _build_actuators_from_entity(self) -> None:
        """Build per-group actuator models from the entity's ArticulationCfg.

        For each actuator config in the entity:
        - ImplicitActuatorCfg → skipped (simulator PD handles it)
        - Any other type → build actuator instance, compute joint index
          mapping from the actuator's target_names_expr to this action
          manager's actuated joint ordering.

        Each actuator sees only its own joint subset (IsaacLab pattern).
        """
        entity_cfg = self._get_entity_cfg()
        if entity_cfg is None:
            return

        for act_cfg in entity_cfg.articulation.actuators:
            if isinstance(act_cfg, ImplicitActuatorCfg):
                continue

            # Find which of our actuated joints this actuator covers
            matched_indices, matched_names = string_utils.resolve_matching_names(
                list(act_cfg.target_names_expr),
                self._actuated_joint_names,
                preserve_order=True,
            )

            if not matched_indices:
                continue

            joint_indices = torch.tensor(matched_indices, device=self.device, dtype=torch.long)
            num_joints_in_group = len(matched_indices)

            # Build actuator for this subset
            actuator = self._build_actuator(
                act_cfg,
                num_joints=num_joints_in_group,
                joint_names=matched_names,
            )
            self._actuators.append((actuator, joint_indices))

        self._has_explicit_actuators = len(self._actuators) > 0

    def _get_entity_cfg(self):
        """Get the unified EntityCfg for the robot from scene manager."""

        entities = getattr(self.env.scene_manager.config, "entities", None)
        if not isinstance(entities, dict):
            return None
        robot_name = getattr(self.env.scene_manager.config, "robot_entity_name", "robot")
        cfg = entities.get(robot_name)
        return cfg if isinstance(cfg, EntityCfg) else None

    def _build_actuator(self, cfg, num_joints: int, joint_names: list[str]):
        """Instantiate an actuator model for a joint subset."""
        cls_map = [
            (ActuatorNetLSTMCfg, ActuatorNetLSTM),
            (ActuatorNetMLPCfg, ActuatorNetMLP),
            (DCMotorCfg, DCMotor),
            (DelayedPDActuatorCfg, DelayedPDActuator),
            (IdealPDActuatorCfg, IdealPDActuator),
        ]
        # Strip simulator-specific prefixes (e.g. "g1_29dof/left_hip_joint" → "left_hip_joint")
        # so gain dicts from robot configs match without prefix awareness.
        bare_names = [name.rsplit("/", 1)[-1] for name in joint_names]

        for cfg_type, actuator_cls in cls_map:
            if isinstance(cfg, cfg_type):
                return actuator_cls(
                    cfg,
                    num_envs=self.env.num_envs,
                    num_joints=num_joints,
                    device=self.device,
                    joint_names=bare_names,
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
        """Apply processed actions to the simulator.

        Two code paths:

        1. **Term-based path** (``config.action_terms`` non-empty):
           dispatch each term's :meth:`ActionTerm.apply_actions`. Each
           term writes its own contribution to the sim — joint-space
           terms via :meth:`_apply_joint_target_via_actuators`,
           non-joint terms (e.g. propeller thrust) via direct sim
           link-wrench APIs. This mirrors IsaacLab's / mjlab's
           ``ActionManager.apply_actions`` and removes the
           joint-position assumption from the manager.

        2. **Legacy path** (no terms): ``processed_actions`` is the
           absolute joint position target; route through the actuator
           path as a single full-action-dim call. Preserved for
           existing presets (Go2 flat, G1 flat, …) that declare
           ``scale``/``clip``/``offset`` directly on the
           ``*ActionConfig`` instead of using terms.

        Args:
            processed_actions: Tensor of shape (num_envs, total_action_dim).
                The return value of :meth:`process_actions`; used only
                on the legacy path. With terms, each term carries its
                own ``processed_actions`` internally.
        """
        if self._has_action_terms:
            for term in self._terms.values():
                term.apply_actions()
            return

        # Legacy non-term path: processed_actions is the full target.
        target = processed_actions
        if not self._has_explicit_actuators:
            self._apply_position(target)
            return
        self._apply_joint_target_full(target)

    def _apply_joint_target_via_actuators(
        self,
        term_target: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        """Helper for :meth:`JointAction.apply_actions`.

        Scatter the term's joint-position target into the full
        action space and route through the actuator-compute path
        (or the direct position path if no explicit actuators are
        configured).

        Args:
            term_target: shape ``(num_envs, len(joint_ids))`` — target
                position for the term's joint subset.
            joint_ids: shape ``(len(joint_ids),)`` — indices into the
                full actuated joint space.
        """
        full_target = torch.zeros(
            (term_target.shape[0], self._total_action_dim),
            dtype=term_target.dtype,
            device=term_target.device,
        )
        full_target[:, joint_ids] = term_target
        if not self._has_explicit_actuators:
            self._apply_position(full_target)
            return
        self._apply_joint_target_full(full_target)

    def _apply_joint_target_full(self, target: torch.Tensor) -> None:
        """Run actuator compute + sim force apply on a
        full-action-dim joint position target. Internal helper used
        by the legacy path and by :meth:`_apply_joint_target_via_actuators`.
        """
        joint_pos = self._get_joint_pos()
        joint_vel = self._get_joint_vel()
        full_torques = torch.zeros_like(target)
        for actuator, joint_idx in self._actuators:
            target_subset = target[:, joint_idx]
            pos_subset = joint_pos[:, joint_idx]
            vel_subset = joint_vel[:, joint_idx]
            torques = actuator.compute(target_subset, pos_subset, vel_subset)
            full_torques[:, joint_idx] = torques
        self._applied_torque = full_torques
        self._apply_force(full_torques)

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Process raw actions: dispatch to term system or legacy path.

        Two code paths:

        1. **Term-based path** (``config.action_terms`` non-empty):
           the raw action is sliced by each term's ``joint_ids`` and
           each term's ``process_actions`` is called. The per-term
           processed outputs are scattered back into a full-action-dim
           tensor and stored in ``_processed_action_history[0]``.
           Final target computation (absolute vs relative vs
           settle-relative) happens later in :meth:`apply_actions`.

        2. **Legacy path** (``config.action_terms`` is None/empty):
           ``clip → scale → offset → optional settle-mask`` exactly
           as before the term system was introduced. Preserved for
           existing presets (Go2 flat, G1 flat, rod_stand, …) that
           declare ``scale``/``clip``/``offset`` directly on
           ``Newton/Genesis/MujocoActionConfig``.

        Args:
            actions: Raw action tensor of shape ``(num_envs, total_action_dim)``.

        Returns:
            Processed action tensor of shape ``(num_envs, total_action_dim)``.
        """
        self._raw_action_history[0] = actions.clone()

        if self._has_action_terms:
            full_processed = torch.zeros_like(actions)
            for term in self._terms.values():
                slice_actions = actions[:, term.joint_ids]
                term.process_actions(slice_actions)
                full_processed[:, term.joint_ids] = term.processed_actions
            self._processed_action_history[0] = full_processed
            return full_processed

        clipped = torch.clip(actions, self._clip_low, self._clip_high)
        processed = clipped * self._scale + self._offset

        if self.config.settle_steps > 0:
            in_settle = (self.env.episode_length_buf < self.config.settle_steps).unsqueeze(-1).float()
            current_pos = self._get_joint_pos()
            processed = in_settle * current_pos + (1.0 - in_settle) * processed

        self._processed_action_history[0] = processed
        return processed

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action buffers, actuator state, and per-term state."""
        if env_ids is None:
            return
        for buf in self._raw_action_history:
            buf[env_ids] = 0.0
        for buf in self._processed_action_history:
            buf[env_ids] = 0.0
        for actuator, _ in self._actuators:
            actuator.reset(env_ids)
        for term in self._terms.values():
            term.reset(env_ids)

    def advance(self) -> None:
        """Advance action history by one step (shift towards older)."""
        for i in range(self._action_history_len - 1, 0, -1):
            self._raw_action_history[i] = self._raw_action_history[i - 1].clone()
            self._processed_action_history[i] = self._processed_action_history[i - 1].clone()

    def print_joint_mapping(self) -> None:
        """Print joint names, indices, and actuator assignments for debugging.

        Shows which joints are actuated, their index in the action vector,
        and which actuator group drives them (with Kp/Kd if applicable).
        Useful for verifying cross-simulator joint ordering consistency.
        """
        sim_type = getattr(self.env, "sim_type", "unknown")
        header = f"Joint Mapping [{sim_type}]"
        print(f"\n{'=' * 60}")
        print(f"  {header}")
        print(f"{'=' * 60}")
        print(f"  {'Idx':<4} {'Joint Name':<40} {'Actuator':<15} {'Kp':<10} {'Kd':<10}")
        print(f"  {'-' * 4} {'-' * 40} {'-' * 15} {'-' * 10} {'-' * 10}")

        # Build actuator lookup: action_idx → (actuator, group_local_idx)
        actuator_lookup: dict[int, tuple] = {}
        for actuator, joint_idx in self._actuators:
            for local_i, global_i in enumerate(joint_idx.tolist()):
                actuator_lookup[global_i] = (actuator, local_i)

        for idx, name in enumerate(self._actuated_joint_names):
            if idx in actuator_lookup:
                act, local_i = actuator_lookup[idx]
                act_type = type(act).__name__
                kp = act.stiffness[0, local_i].item() if hasattr(act, "stiffness") else "-"
                kd = act.damping[0, local_i].item() if hasattr(act, "damping") else "-"
                kp_str = f"{kp:.2f}" if isinstance(kp, float) else kp
                kd_str = f"{kd:.2f}" if isinstance(kd, float) else kd
            else:
                act_type = "Implicit"
                kp_str = "-"
                kd_str = "-"

            print(f"  {idx:<4} {name:<40} {act_type:<15} {kp_str:<10} {kd_str:<10}")

        print(f"{'=' * 60}")
        print(f"  Total actuated joints: {self._total_action_dim}")
        if self._has_explicit_actuators:
            print(f"  Explicit actuator groups: {len(self._actuators)}")
        else:
            print("  Mode: Implicit (simulator PD)")
        print(f"{'=' * 60}\n")

    def __str__(self) -> str:
        """Pretty print action manager configuration."""
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
