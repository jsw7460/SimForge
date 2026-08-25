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
JOINT_LIMIT_SCALE = "joint_limit"
JOINT_LIMIT_CENTER_OFFSET = "joint_limit_center"


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
    scale: float | dict[str, float] | Literal["joint_limit"] = 1.0
    offset: dict[str, float] | Literal["joint_limit_center"] | None = None
    settle_steps: int = 0

    joint_limit_soft_factor: float = 0.9
    """Soft-limit factor for the ``scale="joint_limit"`` /
    ``offset="joint_limit_center"`` auto modes. The usable range per
    joint is the hard limit range shrunk symmetrically about its
    midpoint (IsaacLab / mjlab ``soft_joint_pos_limit_factor``
    convention)::

        mid        = (upper + lower) / 2
        soft_half  = (upper - lower) / 2 * joint_limit_soft_factor
        scale_j    = soft_half          # scale="joint_limit"
        offset_j   = mid                # offset="joint_limit_center"

    A policy emitting actions in [-1, 1] (e.g. a tanh-squashed SAC
    actor) then commands targets spanning exactly the soft limit
    range. Ignored unless one of the auto modes is selected."""

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

        # Per-entity per-joint encoder bias, created on first use by
        # ``encoder_bias_of``. Written by ``randomize_encoder_bias`` (a
        # startup / reset-DR event term) and read by the biased
        # ``dof_pos_biased`` observation so the policy sees a
        # calibration-offset version of the joint state. Absent until a
        # DR term or an observation asks for it, and zero until written.
        self._encoder_bias: dict[str, torch.Tensor] = {}

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
        # Persistent per-entity joint command buffers, keyed by entity
        # name. IsaacLab's equivalent is ``ArticulationData.joint_pos_target``
        # / ``joint_effort_target``: a term writes only the columns it owns
        # and the buffer is flushed to the simulator once per step. A fresh
        # zero buffer per term would instead command every joint the term
        # does NOT own to zero, so an arm term and a gripper term on the
        # same robot would each erase the other.
        self._entity_joint_target: dict[str, torch.Tensor] = {}
        self._entity_joint_effort: dict[str, torch.Tensor] = {}
        self._pending_target_entities: list[str] = []
        self._pending_effort_entities: list[str] = []

        self._terms: dict[str, Any] = {}
        self._term_action_slices: dict[str, slice] = {}
        self._has_action_terms: bool = False
        if config.action_terms:
            for term_name, term_cfg in config.action_terms.items():
                term_class = term_cfg.class_type
                if term_class is None:
                    raise ValueError(f"ActionTermCfg for {term_name!r} has no class_type set — cannot instantiate.")
                self._terms[term_name] = term_class(term_cfg, env=self.env, manager=self)
            self._has_action_terms = len(self._terms) > 0
            self._reject_explicit_actuators_off_the_driven_entity()
            # Sanity: the buffers were allocated from a pre-instantiation
            # estimate of each term's width, so the built terms have to
            # add up to the same total. A mismatch means an estimate and
            # its term disagree about which joints they cover — the
            # policy output would then be sliced along the wrong
            # boundaries and every term after the first would read
            # someone else's actions.
            # Each term owns a contiguous slice of the policy output, laid
            # out in declaration order. Indexing by the term's JOINT ids
            # instead — which coincides with the slice only while a single
            # term covers joints 0..n-1 from index 0 — makes a second term
            # read the first term's actions.
            offset = 0
            for term_name, term in self._terms.items():
                self._term_action_slices[term_name] = slice(offset, offset + term.action_dim)
                offset += term.action_dim

            covered = sum(term.action_dim for term in self._terms.values())
            if covered != self._total_action_dim:
                raise ValueError(
                    f"ActionTerm width mismatch: the built terms cover {covered} joints "
                    f"but the action space was allocated for {self._total_action_dim}."
                )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @property
    def indexing(self):
        """The ArticulationIndexing for this action manager."""
        return self._indexing

    def _build_indexing(self):
        """Build ArticulationIndexing for the entity this manager drives.

        The entity name has to be passed: every backend's builder already
        takes one, and leaving it at the default pinned all three to an
        entity literally named ``"robot"``.
        """
        scene_mgr = self.env.scene_manager
        return scene_mgr.build_articulation_indexing(
            actuated_dof_names=self.config.actuated_dof_names,
            entity_name=self.env.robot_entity_name,
        )

    def _estimate_term_action_dim(self, term_cfg) -> int:
        """Pre-instantiation estimate of a term's action_dim.

        Called before the term itself is built (we need the total
        action dim to allocate raw/processed buffers). Resolution
        order:

        1. Explicit ``num_actions`` field on the cfg (used by
           non-joint terms like ``PropellerThrustActionCfg`` where
           the action dim isn't derivable from joint names).
        2. Joint-name regex match against the joint list of the entity
           the term names (matches the JointAction-style flow at
           :meth:`JointAction.__init__`). Resolved against that entity,
           not the manager's own, so a term driving a second robot is
           sized by that robot's joints. This must produce the same
           count as ``len(term._joint_ids)`` post-instantiation,
           otherwise the buffer sizes will mismatch and the
           coverage sanity check below will trip.
        """
        explicit = getattr(term_cfg, "num_actions", None)
        if explicit is not None:
            return int(explicit)
        joint_names = getattr(term_cfg, "joint_names", None)
        if joint_names is not None:
            entity_name = getattr(term_cfg, "asset_name", self.env.robot_entity_name)
            matched, _ = string_utils.resolve_matching_names(
                joint_names, list(self.env.entity_indexing(entity_name).joint_names), preserve_order=True
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
    def _apply_position(self, targets: torch.Tensor, entity_name: str) -> None:
        """Apply position targets to one entity (uses simulator PD).

        Args:
            targets: Joint position targets, shape ``(num_envs, n)`` where
                ``n`` is that entity's joint count.
            entity_name: Scene entity to write to.
        """
        ...

    @abstractmethod
    def _apply_force(self, torques: torch.Tensor, entity_name: str) -> None:
        """Apply torques directly to one entity's joints (bypasses sim PD).

        Args:
            torques: Joint torques, shape ``(num_envs, n)``.
            entity_name: Scene entity to write to.
        """
        ...

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _soft_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-joint soft limit midpoints and half-ranges.

        Returns:
            Tuple of (mid, soft_half), each shape (num_actuated_joints,)::

                mid       = (upper + lower) / 2
                soft_half = (upper - lower) / 2 * joint_limit_soft_factor

        Used by the ``scale="joint_limit"`` / ``offset="joint_limit_center"``
        auto modes. Only valid on the legacy (non-term) action path where
        every action dimension is an actuated joint.
        """
        joint_lower, _ = self._get_joint_limits()
        if joint_lower.shape[0] != self._total_action_dim:
            raise ValueError(
                f"joint_limit auto mode requires every action dim to be an "
                f"actuated joint: got {joint_lower.shape[0]} joints vs "
                f"total_action_dim={self._total_action_dim} (term-based "
                f"action configs are not supported)."
            )
        return self.soft_joint_limits_of(self.env.robot_entity_name)

    def soft_joint_limits_of(self, entity_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The same, for any articulation in the scene.

        Split out from :meth:`_soft_joint_limits` because a reset acting
        on a second robot needs that robot's limits, in its own joint
        width — while the auto-scale modes above additionally require
        the width to be the action dim, which is a statement about the
        driven robot only.
        """
        indexing = self.env.entity_indexing(entity_name)
        joint_lower, joint_upper = indexing.joint_limits_lower, indexing.joint_limits_upper
        mid = (joint_upper + joint_lower) / 2.0
        soft_half = (joint_upper - joint_lower) / 2.0 * self.config.joint_limit_soft_factor
        if (soft_half <= 0).any():
            names = list(indexing.joint_names)
            bad = [f"{names[i]} (half={soft_half[i].item():.4f})" for i in range(len(soft_half)) if soft_half[i] <= 0]
            raise ValueError(f"{entity_name!r}: non-positive soft half-range for joints: {bad}")
        return mid, soft_half

    def _initialize_scale(self) -> torch.Tensor:
        """Initialize per-dimension scale from configuration.

        Returns:
            Tensor of shape (total_action_dim,).
        """
        scale = torch.ones(self._total_action_dim, device=self.device)

        if isinstance(self.config.scale, str):
            if self.config.scale != JOINT_LIMIT_SCALE:
                raise ValueError(f'Unknown scale mode {self.config.scale!r}; expected "{JOINT_LIMIT_SCALE}".')
            _, soft_half = self._soft_joint_limits()
            scale[:] = soft_half
        elif isinstance(self.config.scale, int | float):
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

        if isinstance(self.config.offset, str):
            if self.config.offset != JOINT_LIMIT_CENTER_OFFSET:
                raise ValueError(f'Unknown offset mode {self.config.offset!r}; expected "{JOINT_LIMIT_CENTER_OFFSET}".')
            mid, _ = self._soft_joint_limits()
            offset[:] = mid
        elif self.config.offset is not None and isinstance(self.config.offset, dict):
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
    def terms(self) -> dict[str, Any]:
        """The action terms, in declaration order."""
        return self._terms

    @property
    def term_action_slices(self) -> dict[str, slice]:
        """Which columns of the policy output each term owns.

        Declaration order, contiguous. This is the map a caller needs to
        hand one robot's slice to one policy and another's to another.
        """
        return self._term_action_slices

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

    def encoder_bias_of(self, entity_name: str | None = None) -> torch.Tensor:
        """Per-env per-joint encoder bias for one entity.

        Shape ``(num_envs, that entity's joint count)``. Written by
        ``randomize_encoder_bias`` (typically at startup / reset-DR) and
        read by the biased observation so the policy sees a
        calibration-offset version of the joint state. Zero when no DR
        term has written to it.

        Keyed by entity rather than shaped to the action dim: with
        several action terms the action dim is their sum, which is no
        robot's joint count, so adding it to a joint reading would not
        even have matching widths.
        """
        name = entity_name or self.env.robot_entity_name
        bias = self._encoder_bias.get(name)
        if bias is None:
            bias = torch.zeros(
                (self.env.num_envs, len(self.env.entity_indexing(name).joint_names)),
                device=self.device,
            )
            self._encoder_bias[name] = bias
        return bias

    def set_encoder_bias(
        self,
        bias: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        entity_name: str | None = None,
    ) -> None:
        """Write one entity's encoder bias for the given envs (or all).

        Used by the ``randomize_encoder_bias`` event term.
        """
        target = self.encoder_bias_of(entity_name)
        if env_ids is None:
            target.copy_(bias)
        else:
            target[env_ids] = bias

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

    def _reject_explicit_actuators_off_the_driven_entity(self) -> None:
        """Refuse a term that drives another entity's explicit actuators.

        The actuator models (``self._actuators``) are built for the
        driven entity alone, and a flat list of them is public API that
        DR terms, observations and several diags index directly. A term
        pointing at a second robot that declares non-implicit actuators
        would therefore be driven with the FIRST robot's actuator models
        — plausible torques computed from the wrong motor. Say so rather
        than produce them.

        Entities driven through the simulator's own PD
        (``ImplicitActuatorCfg``) are unaffected, which covers every
        multi-robot scene built so far.
        """
        driven = self.env.robot_entity_name
        entities = getattr(self.env.scene_manager.config, "entities", None) or {}
        for term_name, term in self._terms.items():
            name = term.entity_name
            if name == driven:
                continue
            cfg = entities.get(name)
            actuators = getattr(getattr(cfg, "articulation", None), "actuators", ())
            explicit = [a for a in actuators if not isinstance(a, ImplicitActuatorCfg)]
            if explicit:
                raise NotImplementedError(
                    f"Action term {term_name!r} drives entity {name!r}, which declares "
                    f"{len(explicit)} explicit actuator group(s). Explicit actuator models are "
                    f"currently built only for the driven entity ({driven!r}); "
                    f"use ImplicitActuatorCfg on {name!r}, or drive it from the driven entity."
                )

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

        has_implicit = any(isinstance(c, ImplicitActuatorCfg) for c in entity_cfg.articulation.actuators)

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

        # Mixing implicit and explicit actuator groups on one entity is not
        # supported: once ANY explicit actuator exists, apply_actions routes
        # ALL actuated joints through the force path, so implicit-group joints
        # are never given position targets — they go limp on Genesis (force
        # mode disables the sim PD) or freeze at the build-time target on
        # Newton/mjlab. Fail loudly instead of silently misbehaving. (Per-group
        # mixed control would require an IsaacLab-style per-group action split.)
        if self._has_explicit_actuators and has_implicit:
            raise NotImplementedError(
                "Entity mixes ImplicitActuatorCfg with explicit actuator configs; "
                "the action pipeline drives all joints through one mode. Use a "
                "single mode per entity (all implicit or all explicit)."
            )

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

    def _get_joint_pos(self, entity_name: str | None = None) -> torch.Tensor:
        """Current joint positions of one entity, via the RobotData protocol."""
        return self.env.get_robot_data(entity_name or self.env.robot_entity_name).joint_pos

    def _get_joint_vel(self, entity_name: str | None = None) -> torch.Tensor:
        """Current joint velocities of one entity, via the RobotData protocol."""
        return self.env.get_robot_data(entity_name or self.env.robot_entity_name).joint_vel

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
            self._flush_entity_commands()
            return

        # Legacy non-term path: processed_actions is the full target.
        target = processed_actions
        name = self.env.robot_entity_name
        if not self._has_explicit_actuators:
            self._apply_position(target, name)
            return
        self._apply_joint_target_full(target, name)

    def _apply_joint_target_via_actuators(
        self,
        term_target: torch.Tensor,
        joint_ids: torch.Tensor,
        entity_name: str | None = None,
    ) -> None:
        """Helper for :meth:`JointAction.apply_actions`.

        Scatter the term's joint-position target into ITS ENTITY's joint
        space and route through the actuator-compute path (or the direct
        position path if no explicit actuators are configured).

        The buffer is sized by the entity's joint count, not by the
        manager's ``total_action_dim``: with several terms the action
        dim is the sum over terms, which is neither the width of any one
        robot's joint space nor indexable by that robot's joint ids.

        Args:
            term_target: shape ``(num_envs, len(joint_ids))`` — target
                position for the term's joint subset.
            joint_ids: shape ``(len(joint_ids),)`` — indices into
                ``entity_name``'s joint space.
            entity_name: entity being driven; defaults to the manager's own.
        """
        name = entity_name or self.env.robot_entity_name
        self._entity_target_buffer(name)[:, joint_ids] = term_target
        if name not in self._pending_target_entities:
            self._pending_target_entities.append(name)

    def _compute_actuator_torques(self, target: torch.Tensor, entity_name: str) -> torch.Tensor:
        """Actuator-model torques for one entity's joint position target."""
        joint_pos = self._get_joint_pos(entity_name)
        joint_vel = self._get_joint_vel(entity_name)
        full_torques = torch.zeros_like(target)
        for actuator, joint_idx in self._actuators:
            target_subset = target[:, joint_idx]
            pos_subset = joint_pos[:, joint_idx]
            vel_subset = joint_vel[:, joint_idx]
            torques = actuator.compute(target_subset, pos_subset, vel_subset)
            full_torques[:, joint_idx] = torques
        return full_torques

    def _apply_joint_target_full(self, target: torch.Tensor, entity_name: str | None = None) -> None:
        """Run actuator compute + sim force apply on one entity's
        joint position target. Internal helper used by the legacy path
        and by :meth:`_apply_joint_target_via_actuators`.
        """
        name = entity_name or self.env.robot_entity_name
        full_torques = self._compute_actuator_torques(target, name)
        # ``applied_torque`` is the policy-facing torque of the driven
        # robot, shaped to its action dim; a second entity's torque must
        # not overwrite it.
        if name == self.env.robot_entity_name:
            self._applied_torque = full_torques
        self._apply_force(full_torques, name)

    def _entity_target_buffer(self, entity_name: str) -> torch.Tensor:
        """One entity's persistent joint-position-target buffer.

        Created on first use and held at the entity's default joint
        positions, so a joint that no term drives holds its home pose
        rather than being commanded to zero.
        """
        buf = self._entity_joint_target.get(entity_name)
        if buf is None:
            default = self.env._resolve_default_joint_pos(entity_name)
            buf = default.to(self.device).unsqueeze(0).repeat(self.env.num_envs, 1)
            self._entity_joint_target[entity_name] = buf
        return buf

    def _entity_effort_buffer(self, entity_name: str) -> torch.Tensor:
        """One entity's persistent joint-effort buffer.

        Zero-filled, and zeroed again after every flush: a torque is a
        per-step quantity, so a term that stops writing a joint must
        stop applying torque to it — unlike a position target, which is
        a hold.
        """
        buf = self._entity_joint_effort.get(entity_name)
        if buf is None:
            buf = torch.zeros(
                (self.env.num_envs, len(self.env.entity_indexing(entity_name).joint_names)),
                device=self.device,
            )
            self._entity_joint_effort[entity_name] = buf
        return buf

    def _flush_entity_commands(self) -> None:
        """Write each entity's accumulated joint command to the simulator.

        Once per entity per step, after every term has contributed. The
        per-term alternative would make the last term to run decide the
        whole entity's command.
        """
        for name in self._pending_target_entities:
            target = self._entity_joint_target[name]
            if not self._has_explicit_actuators:
                self._apply_position(target, name)
                continue
            torques = self._compute_actuator_torques(target, name)
            # An entity can carry position terms AND an effort term on
            # disjoint joints (an arm on wheels: PD arms, velocity-servo
            # wheels). Each backend's ``_apply_force`` ASSIGNS over the
            # entity's whole actuated joint set, so applying the two
            # halves as two calls makes the second silently zero the
            # first; they have to leave in one summed write.
            if name in self._pending_effort_entities:
                effort = self._entity_joint_effort[name]
                torques = torques + effort
                effort.zero_()
                self._pending_effort_entities.remove(name)
            if name == self.env.robot_entity_name:
                self._applied_torque = torques
            self._apply_force(torques, name)
        self._pending_target_entities.clear()

        for name in self._pending_effort_entities:
            effort = self._entity_joint_effort[name]
            if name == self.env.robot_entity_name:
                self._applied_torque = effort.clone()
            self._apply_force(effort, name)
            effort.zero_()
        self._pending_effort_entities.clear()

    def _apply_joint_effort_via_indices(
        self,
        term_torques: torch.Tensor,
        joint_ids: torch.Tensor,
        entity_name: str | None = None,
    ) -> None:
        """Apply a term's joint torques directly to the simulator.

        The effort-action counterpart of
        :meth:`_apply_joint_target_via_actuators`. Scatters the term's
        per-joint torques into the full actuated-joint space and writes
        them straight to the simulator via :meth:`_apply_force`,
        bypassing the actuator-PD compute and the position-target path
        entirely.

        The driven joints must be in the backend's direct-torque mode
        (no ``ImplicitActuatorCfg`` / internal PD); otherwise this
        torque is applied *on top of* the simulator's PD output. See
        :class:`~rlworld.rl.envs.mdp.actions.joint_actions.JointEffortActionCfg`.

        Args:
            term_torques: shape ``(num_envs, len(joint_ids))`` — joint
                torques for the term's joint subset [N·m].
            joint_ids: shape ``(len(joint_ids),)`` — indices into the
                full actuated joint space.
        """
        name = entity_name or self.env.robot_entity_name
        self._entity_effort_buffer(name)[:, joint_ids] = term_torques
        if name not in self._pending_effort_entities:
            self._pending_effort_entities.append(name)

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Process raw actions: dispatch to term system or legacy path.

        Two code paths:

        1. **Term-based path** (``config.action_terms`` non-empty):
           the raw action is sliced by each term's action slice and
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
            for term_name, term in self._terms.items():
                term_slice = self._term_action_slices[term_name]
                term.process_actions(actions[:, term_slice])
                full_processed[:, term_slice] = term.processed_actions
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
