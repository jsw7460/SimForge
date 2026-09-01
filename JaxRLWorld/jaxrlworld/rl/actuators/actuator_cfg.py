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
        frictionloss: Static friction at the joint [N*m]. ``None`` (the
            default) leaves whatever the asset declares — a value forces
            it on every matched joint, zero included.
    """

    target_names_expr: tuple[str, ...] = ()

    class_type: type | None = None
    """Actuator class to instantiate for this config. ``None`` (default)
    uses the built-in cfg-type -> class mapping in the action manager;
    setting it lets external packages plug in custom actuator models
    without editing that map (same pattern as ``ActionTermCfg.class_type``).
    The class must accept the ``ActuatorBase`` constructor signature."""

    stiffness: float | dict[str, float] | None = None
    damping: float | dict[str, float] | None = None
    effort_limit: float | dict[str, float] | None = None
    velocity_limit: float | dict[str, float] | None = None
    armature: float | dict[str, float] = 0.0
    frictionloss: float | None = None


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

    Attributes:
        tau_scale: Optional per-joint saturation scale kappa. When set, the
            PD torque is passed through a smooth ``tau = kappa * tanh(tau_PD /
            kappa)`` before the hard effort clip, modeling the torque decay
            real actuators exhibit in the high-torque regime (small torques
            pass ~linearly; large ones roll off toward +/-kappa). ``None``
            (default) keeps the plain hard clip only. Same scalar-or-dict
            format as ``stiffness``.
    """

    tau_scale: float | dict[str, float] | None = None

    # Piecewise-linear torque-speed (T-N) curve. When BOTH ``velocity_limit``
    # and ``knee_point_velocity`` are set, the deliverable torque is full
    # ``effort_limit`` for |vel| <= knee_point, then ramps linearly to zero at
    # ``velocity_limit`` (booster_train BoosterDelayedPDActuator). Same
    # scalar-or-dict format as ``stiffness``. ``None`` ⇒ plain box clip.
    knee_point_velocity: float | dict[str, float] | None = None

    # First-order lag on the output torque: tau_out follows the PD torque
    # with time constant ``tau_lpf_time_constant`` [s], discretized with
    # ``physics_dt``. Models the motor/current-loop torque bandwidth the
    # real-robot logs show on the hips: slow (static) commands pass with
    # gain 1, walking-frequency content is attenuated. ``None`` disables;
    # 0.0 is an exact passthrough with the per-env buffer allocated (so
    # the value stays runtime-writable for DR / identification). Same
    # scalar-or-dict format as ``stiffness``.
    tau_lpf_time_constant: float | dict[str, float] | None = None
    physics_dt: float = 0.005

    # Velocity-gated transmission efficiency:
    #   tau_out *= 1 - (1 - dyn_gain) * tanh(|vel| / dyn_gain_velocity).
    # Models gear losses that vanish at standstill (full static torque)
    # but scale in during motion, as the real-robot knee logs show:
    # |vel| >> dyn_gain_velocity delivers only ``dyn_gain`` of the
    # commanded torque. ``None`` disables; 1.0 is an exact passthrough
    # with the buffer allocated. Same scalar-or-dict format as
    # ``stiffness``.
    dyn_gain: float | dict[str, float] | None = None
    dyn_gain_velocity: float | dict[str, float] | None = None


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
