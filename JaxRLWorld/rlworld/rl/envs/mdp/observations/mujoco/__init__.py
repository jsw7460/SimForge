"""MuJoCo/mjlab observation functions.

This module provides observation functions for MuJoCo-based environments,
ported from mjlab's MDP module.
"""

from .proprioception import (
    all_joint_pos,
    all_joint_vel,
    base_ang_vel,
    base_height,
    base_lin_vel,
    base_pos,
    base_quat,
    # Commands
    command_velocity,
    # Joint state
    dof_pos,
    dof_pos_nominal_difference,
    dof_vel,
    foot_air_time,
    foot_contact,
    foot_contact_forces,
    foot_contact_time,
    # Contact/feet
    foot_height,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    prev_processed_actions,
    processed_actions,
    # Root state
    projected_gravity,
    # Actions
    raw_actions,
)

__all__ = [
    # Root state
    "projected_gravity",
    "base_lin_vel",
    "base_ang_vel",
    "base_pos",
    "base_quat",
    "base_height",
    # Joint state
    "dof_pos",
    "dof_pos_nominal_difference",
    "dof_vel",
    "all_joint_pos",
    "all_joint_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    # Actions
    "raw_actions",
    "processed_actions",
    "prev_processed_actions",
    "last_action",
    # Commands
    "command_velocity",
    "generated_commands",
    # Contact/feet
    "foot_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
    "foot_contact_time",
]
