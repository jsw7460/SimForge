"""MuJoCo/mjlab termination functions.

These functions provide termination conditions for MuJoCo-based environments,
ported from mjlab's MDP module with adaptations for rlworld's interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.mdp.configs.terminations import TerminationResult
from rlworld.rl.envs.mdp.observations.mujoco import proprioception

if TYPE_CHECKING:
    from rlworld.rl.envs.mujoco import MujocoEnv


def time_out(env: "MujocoEnv") -> TerminationResult:
    """Terminate when the episode length exceeds maximum.

    Args:
        env: The MujocoEnv environment.

    Returns:
        TerminationResult indicating timeout.
    """
    terminated = env.termination_manager.episode_length_buf >= env.termination_manager.max_episode_length
    return TerminationResult(terminated, is_timeout=True)


def bad_orientation(
    env: "MujocoEnv",
    limit_angle: float = 1.0,
) -> TerminationResult:
    """Terminate when the robot's orientation exceeds the limit angle.

    The limit angle is computed from the projected gravity vector.

    Args:
        env: The MujocoEnv environment.
        limit_angle: Maximum allowed tilt angle in radians.

    Returns:
        TerminationResult for orientation violation.
    """
    robot_data = env.scene_manager.robot.data
    projected_gravity = robot_data.projected_gravity_b

    # acos(-z) gives angle from upright
    tilt_angle = torch.acos(-projected_gravity[:, 2]).abs()
    terminated = tilt_angle > limit_angle

    return TerminationResult(terminated)


def root_height_below_minimum(
    env: "MujocoEnv",
    minimum_height: float = 0.2,
) -> TerminationResult:
    """Terminate when the robot's root height falls below minimum.

    Args:
        env: The MujocoEnv environment.
        minimum_height: Minimum allowed root height in meters.

    Returns:
        TerminationResult for height violation.
    """
    robot_data = env.scene_manager.robot.data
    root_height = robot_data.root_link_pos_w[:, 2]
    terminated = root_height < minimum_height

    return TerminationResult(terminated)


def roll_pitch_violation(
    env: "MujocoEnv",
    roll_threshold: float = 0.5,
    pitch_threshold: float = 0.5,
) -> TerminationResult:
    """Terminate when roll or pitch exceeds thresholds.

    Uses projected gravity to compute approximate roll/pitch.

    Args:
        env: The MujocoEnv environment.
        roll_threshold: Maximum allowed roll in radians.
        pitch_threshold: Maximum allowed pitch in radians.

    Returns:
        TerminationResult for roll/pitch violation.
    """
    projected_gravity = proprioception.projected_gravity(env)

    # Approximate roll/pitch from projected gravity
    # For small angles: roll ≈ atan2(g_y, g_z), pitch ≈ atan2(g_x, g_z)
    roll = torch.atan2(projected_gravity[:, 1], -projected_gravity[:, 2])
    pitch = torch.atan2(projected_gravity[:, 0], -projected_gravity[:, 2])

    roll_violated = torch.abs(roll) > roll_threshold
    pitch_violated = torch.abs(pitch) > pitch_threshold

    return TerminationResult(roll_violated | pitch_violated)


def illegal_contact(
    env: "MujocoEnv",
    sensor_name: str | None = None,
    force_threshold: float = 10.0,
) -> TerminationResult:
    """Terminate when illegal contact is detected.

    When ``sensor_name`` is provided, uses the contact sensor directly.
    If the sensor has ``force_history`` (history_length > 0), checks whether
    any substep force exceeds ``force_threshold``. Otherwise falls back to the
    instantaneous ``found`` flag.

    When ``sensor_name`` is None, uses the legacy contact_manager-based path.

    Args:
        env: The MujocoEnv environment.
        sensor_name: Optional contact sensor name for sensor-based detection.
        force_threshold: Force magnitude threshold for history-based detection.

    Returns:
        TerminationResult for illegal contact.
    """
    if sensor_name is not None:
        sensor = env.scene_manager.get_sensor(sensor_name)
        data = sensor.data
        if data.force_history is not None:
            force_mag = torch.norm(data.force_history, dim=-1)
            return TerminationResult((force_mag > force_threshold).any(dim=-1).any(dim=-1))
        assert data.found is not None
        return TerminationResult(torch.any(data.found, dim=-1))

    # Legacy fallback: contact_manager-based
    contact_data = env.contact_manager
    if hasattr(contact_data, 'illegal_contact_detected'):
        return TerminationResult(contact_data.illegal_contact_detected)
    return TerminationResult(torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


def base_contact(env: "MujocoEnv") -> TerminationResult:
    """Terminate when the base/torso makes contact with ground.

    Args:
        env: The MujocoEnv environment.

    Returns:
        TerminationResult for base contact.
    """
    contact_data = env.contact_manager

    # Check for base contact (typically first body in contact sensor)
    if hasattr(contact_data, 'base_contact'):
        terminated = contact_data.base_contact
    else:
        terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    return TerminationResult(terminated)


def nan_detection(env: "MujocoEnv") -> TerminationResult:
    """Terminate environments that have NaN/Inf values in physics state.

    Args:
        env: The MujocoEnv environment.

    Returns:
        TerminationResult for NaN detection.
    """
    robot_data = env.scene_manager.robot.data

    # Check common state variables for NaN
    has_nan = (
        torch.any(torch.isnan(robot_data.joint_pos), dim=1) |
        torch.any(torch.isnan(robot_data.joint_vel), dim=1) |
        torch.any(torch.isnan(robot_data.root_link_pos_w), dim=1)
    )

    return TerminationResult(has_nan)


def joint_limit_violation(
    env: "MujocoEnv",
    margin: float = 0.0,
) -> TerminationResult:
    """Terminate when joint positions exceed limits.

    Args:
        env: The MujocoEnv environment.
        margin: Additional margin beyond soft limits.

    Returns:
        TerminationResult for joint limit violation.
    """
    robot_data = env.scene_manager.robot.data
    joint_ids = env.act_manager._joint_ids

    soft_limits = robot_data.soft_joint_pos_limits
    if soft_limits is None:
        return TerminationResult(
            torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )

    joint_pos = robot_data.joint_pos[:, joint_ids]
    lower_limits = soft_limits[:, joint_ids, 0] - margin
    upper_limits = soft_limits[:, joint_ids, 1] + margin

    below_lower = torch.any(joint_pos < lower_limits, dim=1)
    above_upper = torch.any(joint_pos > upper_limits, dim=1)

    return TerminationResult(below_lower | above_upper)


def velocity_limit_violation(
    env: "MujocoEnv",
    max_linear_velocity: float = 10.0,
    max_angular_velocity: float = 20.0,
) -> TerminationResult:
    """Terminate when base velocities exceed safe limits.

    Args:
        env: The MujocoEnv environment.
        max_linear_velocity: Maximum allowed linear velocity (m/s).
        max_angular_velocity: Maximum allowed angular velocity (rad/s).

    Returns:
        TerminationResult for velocity limit violation.
    """
    base_lin_vel = proprioception.base_lin_vel(env)
    base_ang_vel = proprioception.base_ang_vel(env)

    lin_vel_magnitude = torch.norm(base_lin_vel, dim=1)
    ang_vel_magnitude = torch.norm(base_ang_vel, dim=1)

    lin_exceeded = lin_vel_magnitude > max_linear_velocity
    ang_exceeded = ang_vel_magnitude > max_angular_velocity

    return TerminationResult(lin_exceeded | ang_exceeded)
