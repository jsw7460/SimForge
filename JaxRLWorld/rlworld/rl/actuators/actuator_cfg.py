"""Configuration dataclasses for actuator models.

Each actuator config specifies both **which joints** it drives
(``target_names_expr``) and **how** it drives them (gains, limits,
network files, etc.).  The actuator type determines the control mode:

- :class:`ImplicitActuatorCfg` — simulator's built-in PD controller.
- :class:`IdealPDActuatorCfg` — explicit PD torque computation.
- :class:`DelayedPDActuatorCfg` — explicit PD with command delay.
- :class:`DCMotorCfg` — explicit PD with velocity-dependent saturation.
- :class:`ActuatorNetMLPCfg` — pretrained MLP actuator model.
- :class:`ActuatorNetLSTMCfg` — pretrained LSTM actuator model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ActuatorBaseCfg:
    """Base configuration shared by all actuator models.

    Attributes:
        target_names_expr: Regex patterns matching joint names that
            this actuator drives.
        stiffness: P-gain for PD-based actuators [N*m/rad].
            Can be a single float or a dict mapping joint name regex
            patterns to per-joint values.
        damping: D-gain for PD-based actuators [N*m*s/rad].
            Same format as stiffness.
        effort_limit: Maximum torque [N*m].  Can be a single float (applied
            to all joints matched by ``target_names_expr``) or a dict mapping
            joint-name regex patterns to per-joint values.  ``None`` means no
            limit.
        velocity_limit: Maximum joint velocity [rad/s].
            Used by velocity-dependent saturation models (e.g. DCMotor).
        armature: Reflected rotor inertia added to the joint [kg*m^2].
        frictionloss: Static friction at the joint [N*m].
    """

    target_names_expr: tuple[str, ...] = ()
    stiffness: float | dict[str, float] | None = None
    damping: float | dict[str, float] | None = None
    effort_limit: float | dict[str, float] | None = None
    velocity_limit: float | None = None
    armature: float | dict[str, float] = 0.0
    frictionloss: float = 0.0


@dataclass
class ImplicitActuatorCfg(ActuatorBaseCfg):
    """Actuator handled by the simulator's built-in PD controller.

    No explicit torque computation is performed.  The simulator uses
    the configured stiffness and damping to compute PD torques
    internally at every physics substep.

    This is equivalent to the default behavior when no actuator model
    is specified.
    """

    pass


@dataclass
class IdealPDActuatorCfg(ActuatorBaseCfg):
    """Explicit ideal PD actuator.

    Computes: ``tau = Kp * (target - pos) + Kd * (0 - vel)``

    Torques are computed externally and applied as direct forces,
    bypassing the simulator's built-in PD.
    """

    pass


@dataclass
class DelayedPDActuatorCfg(IdealPDActuatorCfg):
    """Explicit PD actuator with random command delay.

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
    """Explicit PD with velocity-dependent torque saturation (DC motor curve).

    Attributes:
        saturation_effort: Stall torque of the motor [N*m].
    """

    saturation_effort: float = 0.0


@dataclass
class ActuatorNetMLPCfg(ActuatorBaseCfg):
    """MLP-based learned actuator model loaded from TorchScript.

    Attributes:
        network_file: Path to the TorchScript JIT model.
        pos_scale: Scaling applied to position error inputs.
        vel_scale: Scaling applied to velocity inputs.
        torque_scale: Scaling applied to the network's torque output.
        input_order: Whether position or velocity comes first in the
            concatenated network input.
        input_idx: Indices into the history buffer to use as network
            input.  Index 0 is the current step; index n is n steps ago.
    """

    network_file: str = ""
    pos_scale: float = 1.0
    vel_scale: float = 1.0
    torque_scale: float = 1.0
    input_order: Literal["pos_vel", "vel_pos"] = "pos_vel"
    input_idx: tuple[int, ...] = (0,)


@dataclass
class ActuatorNetLSTMCfg(ActuatorBaseCfg):
    """LSTM-based learned actuator model loaded from TorchScript.

    Attributes:
        network_file: Path to the TorchScript JIT model.
    """

    network_file: str = ""
