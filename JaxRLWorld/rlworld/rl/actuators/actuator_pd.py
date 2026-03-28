"""PD-based actuator models."""

from __future__ import annotations

from collections import deque
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

        \tau = K_p (q_{target} - q) + K_d (0 - \dot{q})

    The output is clipped to ``[-effort_limit, effort_limit]``.
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

    def reset(self, env_ids: Sequence[int]) -> None:
        pass

    def compute(
        self,
        target_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        error_pos = target_pos - joint_pos
        self.computed_effort = (
            self.stiffness * error_pos - self.damping * joint_vel
        )
        self.applied_effort = self._clip_effort(self.computed_effort)
        return self.applied_effort


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
        self._buffer = torch.zeros(
            max_delay, num_envs, num_joints, device=device
        )
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
        self._velocity_limit = torch.full(
            (num_envs, num_joints), cfg.velocity_limit, device=device
        )

    def compute(
        self,
        target_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        # Compute raw PD torque
        error_pos = target_pos - joint_pos
        self.computed_effort = (
            self.stiffness * error_pos - self.damping * joint_vel
        )

        # Torque-speed curve limits
        torque_speed_top = self._saturation_effort * (
            1.0 - joint_vel / self._velocity_limit
        )
        torque_speed_bottom = self._saturation_effort * (
            -1.0 - joint_vel / self._velocity_limit
        )

        max_effort = torch.clip(torque_speed_top, max=self.effort_limit)
        min_effort = torch.clip(torque_speed_bottom, min=-self.effort_limit)

        self.applied_effort = torch.clip(
            self.computed_effort, min=min_effort, max=max_effort
        )
        return self.applied_effort
