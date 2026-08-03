"""PD-based actuator models."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .actuator_base import ActuatorBase
from .actuator_cfg import (
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
)


class IdealPDActuator(ActuatorBase):
    r"""Ideal torque-controlled actuator with simple saturation.

    .. math::

        \tau_{PD} = K_p (q_{target} - q) + K_d (0 - \dot{q})

    With ``tau_scale`` (:math:`\kappa`) unset the output is the raw PD torque
    clipped to ``[-effort_limit, effort_limit]``. With ``tau_scale`` set a
    smooth actuator-saturation model is applied first,

    .. math::

        \tau = \kappa \tanh(\tau_{PD} / \kappa),

    so small torques pass ~linearly and large ones roll off toward
    :math:`\pm\kappa` (the torque decay real motors show under high load),
    followed by the same hard effort clip as a safety ceiling.
    """

    cfg: IdealPDActuatorCfg

    def __init__(
        self,
        cfg: IdealPDActuatorCfg,
        num_envs: int,
        num_joints: int,
        device: str,
        joint_names: list[str] | None = None,
    ) -> None:
        super().__init__(cfg, num_envs, num_joints, device, joint_names)

        self.stiffness = self._resolve_per_joint_param(cfg.stiffness, default=0.0)
        self.damping = self._resolve_per_joint_param(cfg.damping, default=0.0)

        # Optional tanh torque-saturation scale kappa. None ⇒ plain hard clip.
        self._use_tau_scale = cfg.tau_scale is not None
        if self._use_tau_scale:
            self.tau_scale = self._resolve_per_joint_param(cfg.tau_scale, default=float("inf"))
            if bool((self.tau_scale <= 0.0).any()):
                raise ValueError("tau_scale (kappa) must be > 0 for the tanh actuator model")

        # Optional piecewise-linear torque-speed (T-N) curve: full effort below
        # knee_point_velocity, ramp to zero at velocity_limit. Active only when
        # both are set; otherwise the plain box clip is used.
        self._use_tn = cfg.velocity_limit is not None and cfg.knee_point_velocity is not None
        if self._use_tn:
            self._vel_limit = self._resolve_per_joint_param(cfg.velocity_limit, default=float("inf"))
            self._knee_point = self._resolve_per_joint_param(cfg.knee_point_velocity, default=0.0)
            self._tn_denom = (self._vel_limit - self._knee_point).clamp(min=1e-6)

    def reset(self, env_ids: Sequence[int]) -> None:
        pass

    def compute(
        self,
        target_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        error_pos = target_pos - joint_pos
        # Raw PD torque (kept in computed_effort for logging/diagnostics).
        self.computed_effort = self.stiffness * error_pos - self.damping * joint_vel
        if self._use_tau_scale:
            effort = self.tau_scale * torch.tanh(self.computed_effort / self.tau_scale)
        else:
            effort = self.computed_effort
        if self._use_tn:
            self.applied_effort = self._clip_effort_tn(effort, joint_vel)
        else:
            self.applied_effort = self._clip_effort(effort)
        return self.applied_effort

    def _clip_effort_tn(self, effort: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear torque-speed clip (booster_train T-N curve): the
        deliverable torque is ``effort_limit`` for ``|vel| <= knee_point``, then
        ramps linearly to 0 at ``velocity_limit``."""
        tau_linear = self.effort_limit * (self._vel_limit - joint_vel.abs()) / self._tn_denom
        max_effort = torch.minimum(tau_linear.clamp(min=0.0), self.effort_limit)
        return torch.clip(effort, min=-max_effort, max=max_effort)


class DelayedPDActuator(IdealPDActuator):
    """Ideal PD actuator with delayed command application.

    A circular buffer stores recent position targets.  The target
    actually sent to the PD computation is lagged by a random number
    of physics steps sampled at each environment reset.
    """

    cfg: DelayedPDActuatorCfg

    def __init__(
        self,
        cfg: DelayedPDActuatorCfg,
        num_envs: int,
        num_joints: int,
        device: str,
        joint_names: list[str] | None = None,
    ) -> None:
        super().__init__(cfg, num_envs, num_joints, device, joint_names)

        max_delay = max(cfg.max_delay, 1)
        # Ring buffer: (max_delay, num_envs, num_joints)
        self._buffer = torch.zeros(max_delay, num_envs, num_joints, device=device)
        self._head = 0
        # Per-env delay in [min_delay, max_delay]
        self._delay = torch.randint(
            cfg.min_delay,
            cfg.max_delay + 1,
            (num_envs,),
            device=device,
            dtype=torch.long,
        )
        self._max_delay = max_delay

    def reset(self, env_ids: Sequence[int]) -> None:
        super().reset(env_ids)
        self._buffer[:, env_ids] = 0.0
        self._delay[env_ids] = torch.randint(
            self.cfg.min_delay,
            self.cfg.max_delay + 1,
            (len(env_ids),),
            device=self._device,
            dtype=torch.long,
        )

    def compute(
        self,
        target_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        # Push current target into the ring buffer
        self._buffer[self._head] = target_pos
        self._head = (self._head + 1) % self._max_delay

        # Read delayed targets: for each env, go back self._delay[i] steps
        read_idx = (self._head - 1 - self._delay) % self._max_delay  # (num_envs,)
        env_idx = torch.arange(self._num_envs, device=self._device)
        delayed_target = self._buffer[read_idx, env_idx]  # (num_envs, num_joints)
        return super().compute(delayed_target, joint_pos, joint_vel)


class DCMotor(IdealPDActuator):
    r"""DC motor actuator with a linear torque-speed saturation curve.

    The instantaneous torque limits depend on the current joint velocity:

    .. math::

        \tau_{max}(\dot{q}) = \text{clip}\!\bigl(
            \tau_{stall} (1 - \dot{q}/\dot{q}_{max}),\;
            -\infty,\; \tau_{continuous}\bigr)

    where :math:`\tau_{stall}` is :attr:`saturation_effort`,
    :math:`\dot{q}_{max}` is :attr:`velocity_limit`, and
    :math:`\tau_{continuous}` is :attr:`effort_limit`.
    """

    cfg: DCMotorCfg

    def __init__(
        self,
        cfg: DCMotorCfg,
        num_envs: int,
        num_joints: int,
        device: str,
        joint_names: list[str] | None = None,
    ) -> None:
        super().__init__(cfg, num_envs, num_joints, device, joint_names)

        if cfg.saturation_effort <= 0:
            raise ValueError("saturation_effort must be > 0 for DCMotor")
        if cfg.velocity_limit is None or cfg.velocity_limit <= 0:
            raise ValueError("velocity_limit must be > 0 for DCMotor")

        self._saturation_effort = cfg.saturation_effort
        self._velocity_limit = torch.full((num_envs, num_joints), cfg.velocity_limit, device=device)

    def compute(
        self,
        target_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        # Compute raw PD torque
        error_pos = target_pos - joint_pos
        self.computed_effort = self.stiffness * error_pos - self.damping * joint_vel

        # Torque-speed curve limits
        torque_speed_top = self._saturation_effort * (1.0 - joint_vel / self._velocity_limit)
        torque_speed_bottom = self._saturation_effort * (-1.0 - joint_vel / self._velocity_limit)

        max_effort = torch.clip(torque_speed_top, max=self.effort_limit)
        min_effort = torch.clip(torque_speed_bottom, min=-self.effort_limit)

        self.applied_effort = torch.clip(self.computed_effort, min=min_effort, max=max_effort)
        return self.applied_effort
