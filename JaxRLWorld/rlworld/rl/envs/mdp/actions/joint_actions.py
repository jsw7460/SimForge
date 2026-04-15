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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from rlworld.rl.utils import string as string_utils
from rlworld.rl.envs.mdp.actions.base import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World
    from rlworld.rl.envs.managers.common.action import ActionManagerBase


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
        env: "World",
        manager: "ActionManagerBase",
    ) -> None:
        super().__init__(cfg, env, manager)

        # Resolve joint_names → indices into manager.actuated_joint_names.
        all_names = manager.actuated_joint_names
        matched_indices, _ = string_utils.resolve_matching_names(
            cfg.joint_names, all_names, preserve_order=True
        )
        if not matched_indices:
            raise ValueError(
                f"ActionTerm {type(self).__name__} got joint_names={cfg.joint_names!r} "
                f"but no actuated joints matched. Available: {all_names}"
            )
        self._joint_ids = torch.tensor(
            matched_indices, device=env.device, dtype=torch.long
        )
        self._joint_names_local = [all_names[i] for i in matched_indices]

        n = len(matched_indices)
        self._raw_actions = torch.zeros(
            (env.num_envs, n), device=env.device, dtype=torch.float32
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Resolve scale dict → per-joint tensor (n,).
        self._scale = self._resolve_float_field(cfg.scale, default=1.0)
        # Resolve offset dict → per-joint tensor (n,).
        self._offset = self._resolve_float_field(cfg.offset, default=0.0)

    def _resolve_float_field(
        self,
        value: "float | dict[str, float]",
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
            indices, _, values = string_utils.resolve_matching_names_values(
                value, self._joint_names_local
            )
            out[indices] = torch.tensor(values, device=self._env.device)
        else:
            raise TypeError(
                f"{type(self).__name__}: expected float or dict for "
                f"scale/offset, got {type(value).__name__}"
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
        env: "World",
        manager: "ActionManagerBase",
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
        env: "World",
        manager: "ActionManagerBase",
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
            in_settle = (
                self._env.episode_length_buf < settle_steps
            ).unsqueeze(-1)
            target = torch.where(in_settle, current_pos, target)

        return target
