"""MuJoCo/mjlab termination functions.

This module provides termination functions for MuJoCo-based environments,
ported from mjlab's MDP module.
"""

# MuJoCo-specific termination functions.
# nan_detection lives in terminations.common (sim-agnostic RobotData impl).
from .terminations import (
    bad_orientation,
    base_contact,
    illegal_contact,
    joint_limit_violation,
    roll_pitch_violation,
    root_height_below_minimum,
    time_out,
    velocity_limit_violation,
)

__all__ = [
    "time_out",
    "bad_orientation",
    "root_height_below_minimum",
    "roll_pitch_violation",
    "illegal_contact",
    "base_contact",
    "joint_limit_violation",
    "velocity_limit_violation",
]
