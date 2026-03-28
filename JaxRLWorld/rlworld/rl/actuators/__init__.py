"""Actuator models for bridging RL actions to simulator joint torques.

Actuator models augment simulated joints with external drive dynamics.
They convert user-provided joint position targets into torques that are
applied directly to the simulation, bypassing the simulator's built-in
PD controller.  This enables more realistic motor modelling (delays,
saturation, learned dynamics) which is critical for sim-to-real transfer.
"""

from .actuator_base import ActuatorBase
from .actuator_cfg import (
    ActuatorBaseCfg,
    ImplicitActuatorCfg,
    IdealPDActuatorCfg,
    DelayedPDActuatorCfg,
    DCMotorCfg,
    ActuatorNetMLPCfg,
    ActuatorNetLSTMCfg,
)
from .actuator_pd import DCMotor, DelayedPDActuator, IdealPDActuator
from .actuator_net import ActuatorNetMLP, ActuatorNetLSTM

__all__ = [
    "ActuatorBase",
    "ActuatorBaseCfg",
    "ImplicitActuatorCfg",
    "IdealPDActuatorCfg",
    "DelayedPDActuatorCfg",
    "DCMotorCfg",
    "ActuatorNetMLPCfg",
    "ActuatorNetLSTMCfg",
    "IdealPDActuator",
    "DelayedPDActuator",
    "DCMotor",
    "ActuatorNetMLP",
    "ActuatorNetLSTM",
]
