"""Configuration dataclasses for actuator models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ActuatorBaseCfg:
    """Base configuration shared by all actuator models.

    Attributes:
        effort_limit: Maximum torque the actuator can produce [N*m].
            If None, no clipping is applied.
        velocity_limit: Maximum joint velocity [rad/s].
            Used by velocity-dependent saturation models (e.g. DCMotor).
        stiffness: P-gain for PD-based actuators.
            Can be a single float (same for all joints) or a dict mapping
            joint name regex patterns to per-joint values.
        damping: D-gain for PD-based actuators.  Same format as stiffness.
    """

    effort_limit: float | None = None
    velocity_limit: float | None = None
    stiffness: float | dict[str, float] | None = None
    damping: float | dict[str, float] | None = None


@dataclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator.

    Computes: tau = Kp * (target - pos) + Kd * (0 - vel)
    """

    pass


@dataclass
class DelayedPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for a PD actuator with command delay.

    At each environment reset, a random delay (in physics steps) is
    sampled uniformly from [min_delay, max_delay] for each environment.

    Attributes:
        min_delay: Minimum delay in physics time-steps.
        max_delay: Maximum delay in physics time-steps.
    """

    min_delay: int = 0
    max_delay: int = 0


@dataclass
class DCMotorCfg(IdealPDActuatorCfg):
    """Configuration for a DC motor with velocity-dependent torque saturation.

    Uses a linear torque-speed curve to compute instantaneous torque
    limits based on the current joint velocity.

    Attributes:
        saturation_effort: Stall torque of the motor [N*m].
    """

    saturation_effort: float = 0.0


@dataclass
class ActuatorNetMLPCfg(ActuatorBaseCfg):
    """Configuration for an MLP-based learned actuator model.

    The network is loaded from a TorchScript (.pt) file and maps
    a history of (position_error, velocity) to output torque.

    Attributes:
        network_file: Path to the TorchScript JIT model.
        pos_scale: Scaling applied to position error inputs.
        vel_scale: Scaling applied to velocity inputs.
        torque_scale: Scaling applied to the network's torque output.
        input_order: Whether position or velocity comes first in the
            concatenated network input.
        input_idx: Indices into the history buffer to use as network
            input.  Index 0 is the current step; index n is n steps ago.
            The history buffer length is ``max(input_idx) + 1``.
    """

    network_file: str = ""
    pos_scale: float = 1.0
    vel_scale: float = 1.0
    torque_scale: float = 1.0
    input_order: Literal["pos_vel", "vel_pos"] = "pos_vel"
    input_idx: tuple[int, ...] = (0,)


@dataclass
class ActuatorNetLSTMCfg(ActuatorBaseCfg):
    """Configuration for an LSTM-based learned actuator model.

    The LSTM implicitly captures temporal dynamics through its hidden
    state, removing the need for an explicit history buffer.

    Attributes:
        network_file: Path to the TorchScript JIT model.
    """

    network_file: str = ""
