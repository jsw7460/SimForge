"""Joint-space action terms: absolute / relative / settle-relative.

Three concrete :class:`ActionTerm` subclasses covering the action
interpretations every existing and planned JaxRLWorld task needs:

* :class:`JointPositionAction` — **absolute** target. Policy outputs
  are scaled and offset by a fixed reference (default joint pose
  when ``use_default_offset=True``). Mirrors IsaacLab's /
  mjlab's ``JointPositionAction``. Used by locomotion tasks
  (Go2, G1 flat) where the policy commands displacements around
  a known standing pose.

* :class:`RelativeJointPositionAction` — **delta** target. Target is
  ``current_joint_pos + raw * scale``. ``raw=0`` means "hold current".
  Mirrors mjlab's ``RelativeJointPositionAction``. Used by
  manipulation / reaching tasks where the policy commands
  incremental moves from the current state.

* :class:`SettleRelativeJointPositionAction` — **delta + settle**.
  Same as Relative but during the first ``settle_steps`` control
  steps after each reset the target is forcibly held at
  ``current_joint_pos`` regardless of the policy output. Added by
  mjlab_playground specifically for fall-recovery (getup) tasks
  where the robot is dropped from height and must not receive any
  policy command until it has physically settled on the ground.

All three share :class:`JointAction`'s scale/offset processing;
they differ only in :meth:`ActionTerm.compute_target_positions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.mdp.actions.base import ActionTerm, ActionTermCfg
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.common.action import ActionManagerBase
    from rlworld.rl.envs.world import World


# ── Shared base ─────────────────────────────────────────────────────


@dataclass
class JointActionCfg(ActionTermCfg):
    """Shared config for joint-space action terms.

    Attributes:
        scale: Per-joint scale. ``float`` applies to every joint in
            this term; ``dict[regex → float]`` resolves per-joint.
        offset: Per-joint offset added to ``raw * scale``. Used by
            the absolute :class:`JointPositionAction` to reference a
            default pose. Relative terms typically force this to 0.
    """

    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0


class JointAction(ActionTerm):
    """Base class implementing ``processed = raw * scale + offset``.

    Subclasses override :meth:`compute_target_positions` to turn the
    intermediate ``processed_actions`` into an absolute joint
    position target.
    """

    __name__ = "JointAction"

    def __init__(
        self,
        cfg: JointActionCfg,
        env: World,
        manager: ActionManagerBase,
    ) -> None:
        super().__init__(cfg, env, manager)

        # Resolve joint_names → indices into manager.actuated_joint_names.
        all_names = manager.actuated_joint_names
        matched_indices, _ = string_utils.resolve_matching_names(cfg.joint_names, all_names, preserve_order=True)
        if not matched_indices:
            raise ValueError(
                f"ActionTerm {type(self).__name__} got joint_names={cfg.joint_names!r} "
                f"but no actuated joints matched. Available: {all_names}"
            )
        self._joint_ids = torch.tensor(matched_indices, device=env.device, dtype=torch.long)
        self._joint_names_local = [all_names[i] for i in matched_indices]

        n = len(matched_indices)
        self._raw_actions = torch.zeros((env.num_envs, n), device=env.device, dtype=torch.float32)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Resolve scale dict → per-joint tensor (n,).
        self._scale = self._resolve_float_field(cfg.scale, default=1.0)
        # Resolve offset dict → per-joint tensor (n,).
        self._offset = self._resolve_float_field(cfg.offset, default=0.0)

    def _resolve_float_field(
        self,
        value: float | dict[str, float],
        default: float,
    ) -> torch.Tensor:
        """Map a float or regex-dict onto the term's joint set."""
        out = torch.full(
            (self.action_dim,),
            float(default),
            device=self._env.device,
            dtype=torch.float32,
        )
        if isinstance(value, (int, float)):
            out[:] = float(value)
        elif isinstance(value, dict):
            indices, _, values = string_utils.resolve_matching_names_values(value, self._joint_names_local)
            out[indices] = torch.tensor(values, device=self._env.device)
        else:
            raise TypeError(
                f"{type(self).__name__}: expected float or dict for scale/offset, got {type(value).__name__}"
            )
        return out

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clip (optional), scale, and offset the raw action slice."""
        if self._cfg.clip is not None:
            lo, hi = self._cfg.clip
            actions = torch.clamp(actions, lo, hi)
        self._raw_actions[:] = actions
        self._processed_actions = actions * self._scale + self._offset


# ── Absolute ────────────────────────────────────────────────────────


@dataclass
class JointPositionActionCfg(JointActionCfg):
    """Absolute joint position action.

    Target = ``raw * scale + offset``. If ``use_default_offset=True``,
    the ``offset`` field is overridden at construction with the
    robot's default joint angles (read from the scene's
    ``act_manager.offset[0]`` first row). Legacy-compatible with the
    monolithic ``ActionManagerBase`` code path when wrapped by the
    manager's auto-shim.

    Attributes:
        use_default_offset: When True, override ``offset`` with the
            robot's default joint angles. Matches IsaacLab's
            ``JointPositionActionCfg.use_default_offset``.
    """

    use_default_offset: bool = False


class JointPositionAction(JointAction):
    """Absolute joint position action — target = processed."""

    __name__ = "JointPositionAction"

    def __init__(
        self,
        cfg: JointPositionActionCfg,
        env: World,
        manager: ActionManagerBase,
    ) -> None:
        super().__init__(cfg, env, manager)
        if cfg.use_default_offset:
            # Override the per-joint offset tensor with the robot's
            # default joint angles. Manager exposes a pre-built
            # ``offset`` tensor of shape ``(num_envs, total_action_dim)``
            # on its legacy path; we slice out this term's joints
            # from env 0 (all envs share the same default pose).
            legacy_offset = manager.offset[0]
            self._offset = legacy_offset[self._joint_ids].clone()

    def compute_target_positions(self) -> torch.Tensor:
        # processed_actions is already the absolute target (scale+offset).
        return self._processed_actions


# ── Relative ────────────────────────────────────────────────────────


@dataclass
class RelativeJointPositionActionCfg(JointActionCfg):
    """Relative / delta joint position action.

    Target = ``current_joint_pos + raw * scale``. Offset is forced
    to 0 so that ``raw=0`` means "hold current". Mirrors mjlab's
    ``RelativeJointPositionActionCfg``.

    Attributes:
        use_zero_offset: When True (default), the resolved per-joint
            ``offset`` tensor is overwritten with zeros at
            construction — matches mjlab's semantics.
    """

    use_zero_offset: bool = True


class RelativeJointPositionAction(JointAction):
    """Delta joint position action.

    ``processed_actions = raw * scale`` (delta in joint-space).
    ``compute_target_positions`` returns ``current_joint_pos + processed``.
    """

    __name__ = "RelativeJointPositionAction"

    def __init__(
        self,
        cfg: RelativeJointPositionActionCfg,
        env: World,
        manager: ActionManagerBase,
    ) -> None:
        super().__init__(cfg, env, manager)
        if cfg.use_zero_offset:
            self._offset = torch.zeros_like(self._offset)

    def compute_target_positions(self) -> torch.Tensor:
        current_pos = self._manager._get_joint_pos()[:, self._joint_ids]
        return current_pos + self._processed_actions


# ── Settle relative (getup) ─────────────────────────────────────────


@dataclass
class SettleRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Delta joint position action with an initial settle-and-hold phase.

    Extends :class:`RelativeJointPositionActionCfg` with a
    ``settle_steps`` field. During the first ``settle_steps`` control
    steps after each reset, the target is held at the current joint
    position regardless of the policy output, giving the robot time
    to settle physically after a drop/impact before the policy takes
    over.

    Mirrors mjlab_playground's
    ``SettleRelativeJointPositionActionCfg`` used by the T1 getup
    task at ``mjlab_playground/getup/mdp/actions.py``.

    Attributes:
        settle_steps: Number of control steps post-reset during
            which the policy output is masked and the target equals
            the current joint position. Set to 0 to disable (then
            this term behaves identically to
            :class:`RelativeJointPositionAction`).
    """

    settle_steps: int = 0


class SettleRelativeJointPositionAction(RelativeJointPositionAction):
    """Relative action with a first-N-steps hold.

    ``compute_target_positions`` is identical to
    :class:`RelativeJointPositionAction` except that envs whose
    ``episode_length_buf < settle_steps`` have their target clamped
    to ``current_joint_pos``.
    """

    __name__ = "SettleRelativeJointPositionAction"

    def compute_target_positions(self) -> torch.Tensor:
        current_pos = self._manager._get_joint_pos()[:, self._joint_ids]
        target = current_pos + self._processed_actions

        settle_steps = self._cfg.settle_steps
        if settle_steps > 0:
            in_settle = (self._env.episode_length_buf < settle_steps).unsqueeze(-1)
            target = torch.where(in_settle, current_pos, target)

        return target


# ── Motion-residual (Any2Track-style) ───────────────────────────────


@dataclass
class MotionResidualJointPositionActionCfg(JointActionCfg):
    """Motion-anchored residual joint position action.

    Used by motion tracking tasks. Target each step is the reference
    motion's joint position plus a tanh-bounded, per-joint-scaled
    correction the policy outputs:

        target = motion_command.joint_pos + alpha * tanh(raw)

    ``raw = 0`` (the PPO Gaussian's mean at init) reduces to perfect
    motion playback, so the policy's bootstrapping baseline already
    tracks the motion. The policy only needs to learn corrections
    (balance, contact reaction, embodiment gap) on top, which is a
    much smaller function than learning motion + corrections together.
    Bounded ``tanh`` also prevents PD target spikes from outlier
    action samples during early training.

    Inspired by Any2Track [arXiv 2025] eq. (1):
    ``q_d = q_tilde_{t+1} + alpha * tanh(pi(a_t | s_t))``.

    The base ``scale`` and ``offset`` fields are unused — this term
    overrides ``process_actions`` so motion + alpha * tanh(raw) is
    the entire pipeline.

    Attributes:
        command_name: Name of the :class:`MotionCommand` term in the
            env's :class:`CommandManager`. The term reads
            ``env.command_manager.get_term(command_name).joint_pos``
            as the residual anchor each step. Joint positions in the
            command are assumed to be in the env's canonical
            actuated-joint order (which :class:`MotionLoader` enforces).
        alpha: Per-joint correction scale. ``float`` applies the same
            value to every joint; ``dict[regex -> float]`` resolves
            per-joint via the standard regex-on-name plumbing. Default
            0.5 — i.e. each joint can deviate up to ±0.5 rad from the
            motion reference per step.
    """

    command_name: str = "motion"
    alpha: float | dict[str, float] = 0.5


class MotionResidualJointPositionAction(JointAction):
    """Motion-anchored residual joint position action.

    target = motion_command.joint_pos[:, joint_ids] + alpha * tanh(raw)

    ``compute_target_positions`` reads the motion command's joint
    reference each control step and adds a per-joint, tanh-bounded
    correction from the policy. The base ``scale`` and ``offset``
    pipeline is bypassed; raw actions are stored unchanged in
    ``processed_actions`` for logging compatibility but not used in
    the target computation.
    """

    __name__ = "MotionResidualJointPositionAction"
    _cfg: MotionResidualJointPositionActionCfg

    def __init__(
        self,
        cfg: MotionResidualJointPositionActionCfg,
        env: World,
        manager: ActionManagerBase,
        tanh_squash: bool = False,
    ) -> None:
        super().__init__(cfg, env, manager)
        # Per-joint tanh scale (alpha). Replaces self._scale/_offset's role.
        self._alpha = self._resolve_float_field(cfg.alpha, default=0.5)
        self._tanh_squash = tanh_squash

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clip then store. ``processed_actions = raw`` (no scale/offset)."""
        if self._cfg.clip is not None:
            lo, hi = self._cfg.clip
            actions = torch.clamp(actions, lo, hi)
        self._raw_actions[:] = actions
        # Kept identical to raw so logging / diagnostics that read
        # processed_actions still see something sensible. Not used in
        # target computation.
        self._processed_actions[:] = actions

    def compute_target_positions(self) -> torch.Tensor:
        cmd = self._env.command_manager.get_term(self._cfg.command_name)
        motion_target = cmd.joint_pos[:, self._joint_ids]
        if self._tanh_squash:
            residual = self._alpha * torch.tanh(self._raw_actions)
        else:
            residual = self._alpha * self._raw_actions
        return motion_target + residual
