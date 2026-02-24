"""MuJoCo/mjlab reward functions.

This module provides reward functions for MuJoCo-based environments,
ported from mjlab's MDP module.
"""

# MuJoCo-specific reward functions
from .reward_terms import (
    # Basic rewards
    is_alive,
    is_terminated,
    # Velocity tracking
    track_linear_velocity,
    track_angular_velocity,
    # Joint-based penalties
    joint_torques_l2,
    joint_vel_l2,
    joint_acc_l2,
    joint_pos_limits,
    # Action-based penalties
    action_rate_l2,
    # Orientation rewards
    flat_orientation_l2,
    flat_orientation,
    # Contact/feet rewards
    feet_air_time,
    feet_clearance,
    feet_slip,
    soft_landing,
    body_angular_velocity_penalty,
    # Utility
)

__all__ = [
    # Basic rewards
    "is_alive",
    "is_terminated",
    # Velocity tracking
    "track_linear_velocity",
    "track_angular_velocity",
    # Joint-based penalties
    "joint_torques_l2",
    "joint_vel_l2",
    "joint_acc_l2",
    "joint_pos_limits",
    # Action-based penalties
    "action_rate_l2",
    # Orientation rewards
    "flat_orientation_l2",
    "flat_orientation",
    # Contact/feet rewards
    "feet_air_time",
    "feet_clearance",
    "feet_slip",
    "soft_landing",
    "body_angular_velocity_penalty",
    # Utility
]
