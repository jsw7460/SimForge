"""MuJoCo/mjlab termination functions.

This module provides termination functions for MuJoCo-based environments,
ported from mjlab's MDP module.
"""

# MuJoCo-specific termination functions
from .terminations import (
    time_out,
    bad_orientation,
    root_height_below_minimum,
    roll_pitch_violation,
    illegal_contact,
    base_contact,
    nan_detection,
    joint_limit_violation,
    velocity_limit_violation,
)

__all__ = [
    "time_out",
    "bad_orientation",
    "root_height_below_minimum",
    "roll_pitch_violation",
    "illegal_contact",
    "base_contact",
    "nan_detection",
    "joint_limit_violation",
    "velocity_limit_violation",
]
