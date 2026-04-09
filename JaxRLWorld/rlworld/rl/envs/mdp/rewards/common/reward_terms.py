"""Unified reward terms using the RobotData interface.

All functions accept any ``World`` subclass and read state exclusively
through ``env.get_robot_data(entity_name)``, making them simulator-agnostic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


# ── Quadruped leg geometry helpers ───────────────────────────────────────

# Nominal x/y sign for each leg in body frame (x=forward, y=left).
_LEG_NOMINAL_SIGNS = {
      "FL": (+1.0, +1.0),   # Front-Left:  +x, +y (URDF: left = +y)
      "FR": (+1.0, -1.0),   # Front-Right: +x, -y (URDF: right = -y)
      "RL": (-1.0, +1.0),   # Rear-Left:   -x, +y
      "RR": (-1.0, -1.0),   # Rear-Right:  -x, -y
}


def get_leg_xy_signs(foot_names: tuple[str, ...] | list[str]) -> list[tuple[float, float]]:
    """Return (x_sign, y_sign) for each foot, matching foot_names order.

    Parses FL/FR/RL/RR substring from each name.
    """
    signs = []
    for name in foot_names:
        matched = [key for key in _LEG_NOMINAL_SIGNS if key in name]
        if len(matched) != 1:
            raise ValueError(
                f"Cannot identify leg from foot name '{name}'. "
                f"Expected exactly one of {list(_LEG_NOMINAL_SIGNS)} as substring."
            )
        signs.append(_LEG_NOMINAL_SIGNS[matched[0]])
    return signs


def track_lin_vel(
    env: World,
    std: float = 0.25,
    penalize_z: bool = False,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward for tracking commanded linear velocity in xy plane.

    Args:
        env: Any environment with ``get_robot_data``.
        std: Standard deviation for exponential kernel.
        penalize_z: If True, include z-velocity penalty (assumes commanded z is zero).
            Matches mjlab ``track_linear_velocity`` behavior.
        entity_name: Name of the entity to query.

    Returns:
        Tensor of shape (num_envs,).
    """
    target = torch.stack(
        [env.command_manager.lin_vel_x, env.command_manager.lin_vel_y], dim=1
    )
    actual = env.get_robot_data(entity_name).root_link_lin_vel_b
    xy_error = torch.sum(torch.square(target - actual[:, :2]), dim=1)
    if penalize_z:
        xy_error = xy_error + torch.square(actual[:, 2])
    return torch.exp(-xy_error / std ** 2)


def track_ang_vel(
    env: World,
    std: float = 0.25,
    penalize_xy: bool = False,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward for tracking commanded angular velocity (yaw).

    Args:
        env: Any environment with ``get_robot_data``.
        std: Standard deviation for exponential kernel.
        penalize_xy: If True, include xy angular velocity penalty (assumes
            commanded xy is zero). Matches mjlab ``track_angular_velocity``.
        entity_name: Name of the entity to query.

    Returns:
        Tensor of shape (num_envs,).
    """
    actual = env.get_robot_data(entity_name).root_link_ang_vel_b
    z_error = torch.square(env.command_manager.ang_vel - actual[:, 2])
    if penalize_xy:
        z_error = z_error + torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-z_error / std ** 2)


def action_rate_l2(env: World) -> torch.Tensor:
    """Penalty for sudden action changes (L2 squared).

    Returns:
        Tensor of shape (num_envs,).
    """
    return -torch.sum(
        torch.square(
            env.act_manager.prev_processed_actions
            - env.act_manager.processed_actions
        ),
        dim=1,
    )


def flat_orientation(
    env: World,
    std: float | None = None,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalty for non-flat orientation (roll/pitch deviation from upright).

    Args:
        env: Any environment with ``get_robot_data``.
        std: If provided, use exponential kernel ``exp(-xy² / std²)`` returning
            a positive reward (matches mjlab behavior). If None, return
            ``-sum(xy²)`` as a negative penalty.
        entity_name: Name of the entity to query.

    Returns:
        Tensor of shape (num_envs,).
    """
    gravity_b = env.get_robot_data(entity_name).projected_gravity_b
    xy_squared = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    if std is not None:
        return torch.exp(-xy_squared / (std ** 2))
    return -xy_squared



# ── Walk-These-Ways reward terms ─────────────────────────────────────────

def penalize_lin_vel_z(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Penalize z-axis base linear velocity. WTW: _reward_lin_vel_z."""
    vel_z = env.get_robot_data(entity_name).root_link_lin_vel_b[:, 2]
    return -torch.square(vel_z)


def penalize_ang_vel_xy(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Penalize xy-axis base angular velocity. WTW: _reward_ang_vel_xy."""
    ang_vel_xy = env.get_robot_data(entity_name).root_link_ang_vel_b[:, :2]
    return -torch.sum(torch.square(ang_vel_xy), dim=1)


def penalize_dof_vel(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Penalize joint velocities. WTW: _reward_dof_vel."""
    return -torch.sum(torch.square(env.get_robot_data(entity_name).joint_vel), dim=1)


def penalize_action_smoothness_1(env: World) -> torch.Tensor:
    """Penalize 1st-order action changes (processed). WTW: _reward_action_smoothness_1.

    Uses processed_action_history (joint position targets) and masks
    the first step where raw actions are still zero.
    """
    hist = env.act_manager.processed_action_history
    diff = torch.square(hist[0] - hist[1])
    mask = (env.act_manager.raw_action_history[1] != 0)
    return -torch.sum(diff * mask, dim=1)


def penalize_action_smoothness_2(env: World) -> torch.Tensor:
    """Penalize 2nd-order action changes (processed). WTW: _reward_action_smoothness_2.

    Second-order finite difference of joint position targets, masked
    for the first two steps.
    """
    hist = env.act_manager.processed_action_history
    diff = torch.square(hist[0] - 2.0 * hist[1] + hist[2])
    mask1 = (env.act_manager.raw_action_history[1] != 0)
    mask2 = (env.act_manager.raw_action_history[2] != 0)
    return -torch.sum(diff * mask1 * mask2, dim=1)


def penalize_orientation_control(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Penalize deviation from commanded body orientation. WTW: _reward_orientation_control.

    Constructs desired body quaternion from body_pitch and body_roll commands,
    computes desired projected gravity, and penalizes xy-deviation from actual.
    """
    from rlworld.rl.utils.quat_utils import (
        quat_from_angle_axis_wxyz,
        quat_mul_wxyz,
        quat_rotate_inverse_wxyz,
    )

    body_pitch = env.command_manager.body_pitch
    body_roll = env.command_manager.body_roll
    device = body_pitch.device

    # WTW: quat_roll = quat_from_angle_axis(-body_roll, [1,0,0])
    #      quat_pitch = quat_from_angle_axis(-body_pitch, [0,1,0])
    #      desired = quat_mul(quat_roll, quat_pitch)
    axis_x = torch.tensor([1.0, 0.0, 0.0], device=device)
    axis_y = torch.tensor([0.0, 1.0, 0.0], device=device)
    quat_roll = quat_from_angle_axis_wxyz(-body_roll, axis_x)
    quat_pitch = quat_from_angle_axis_wxyz(-body_pitch, axis_y)
    desired_quat = quat_mul_wxyz(quat_roll, quat_pitch)

    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=device).expand(len(body_pitch), -1)
    desired_gravity = quat_rotate_inverse_wxyz(desired_quat, gravity_vec)

    actual_gravity = env.get_robot_data(entity_name).projected_gravity_b
    return -torch.sum(torch.square(actual_gravity[:, :2] - desired_gravity[:, :2]), dim=1)


def reward_body_height_cmd(
    env: World,
    base_height_target: float = 0.30,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward for tracking commanded body height. WTW: _reward_jump.

    Target height = body_height command + base_height_target.
    """
    body_height = env.get_robot_data(entity_name).root_link_pos_w[:, 2]
    target = env.command_manager.body_height + base_height_target
    return -torch.square(body_height - target)


def similar_to_default(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Penalty for deviating from default joint positions.

    Returns:
        Tensor of shape (num_envs,).
    """
    return -torch.sum(
        torch.abs(env.get_robot_data(entity_name).joint_pos - env.act_manager.offset), dim=1
    )


def reward_alive(env: World) -> torch.Tensor:
    """Constant alive reward (1.0 per env).

    Returns:
        Tensor of shape (num_envs,) on the default device. Matches the
        original sim-specific implementations exactly: ``torch.ones((num_envs,))``.
    """
    return torch.ones((env.num_envs,))


def base_height_penalty(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Penalty for deviating from target base height.

    Returns negative squared error between actual base z and the desired
    height stored in ``env.command_manager.base_height``.

    Returns:
        Tensor of shape (num_envs,).
    """
    height_z = env.get_robot_data(entity_name).root_link_pos_w[:, 2]
    return -torch.square(height_z - env.command_manager.base_height)


def penalize_body_ang_vel_xy(
    env: World,
    body_name: str,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity of a single body (sim-agnostic).

    Reads world-frame angular velocity for the named body via
    ``RobotData.find_body_index`` and ``RobotData.body_ang_vel_w``, then
    returns ``-sum(square(ang_vel[:, :2]))``. The yaw component (index 2)
    is intentionally NOT penalized — only roll/pitch are.

    Matches the behavior of mjlab's ``body_angular_velocity_penalty``
    exactly, which the legacy sim-specific implementations also matched.

    Args:
        env: Any environment whose RobotData implements the body-level
            accessors (Newton, Genesis, MuJoCo).
        body_name: Name of the body. Format depends on the simulator's
            naming convention (Newton uses prefixed names like
            ``"g1_29dof/torso_link"``, Genesis and mjlab use bare names
            like ``"torso_link"``).
        entity_name: Name of the entity. Default ``"robot"``.

    Returns:
        Tensor of shape ``(num_envs,)``.
    """
    rd = env.get_robot_data(entity_name)
    body_idx = rd.find_body_index(body_name)
    ang_vel = rd.body_ang_vel_w(body_idx)
    return -torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def penalize_joint_pos_limits_l1(
    env: World,
    soft_limit_factor: float = 1.0,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize joint positions exceeding soft limits (L1, sim-agnostic).

    Matches the math of mjlab's ``joint_pos_limits`` exactly:

        out = max(lower - q, 0) + max(q - upper, 0)
        return -sum(out, dim=-1)

    Where ``lower``, ``upper`` are the *soft* limits, computed as
    ``hard_lower * soft_limit_factor`` and ``hard_upper * soft_limit_factor``.

    Reads ``RobotData.joint_pos`` and ``RobotData.joint_pos_limits``, both
    in canonical actuated joint order.

    Args:
        env: Any environment whose ``RobotData`` implements
            ``joint_pos_limits`` (Newton, Genesis). Note: not callable on
            MuJoCo, which uses its own ``joint_pos_limits`` reward function
            in ``mdp/rewards/mujoco/reward_terms.py``.
        soft_limit_factor: Multiplicative factor on the hard limits.
            ``1.0`` (the active default in current presets) means
            penalize whenever the joint exceeds its hard limit.
        entity_name: Name of the entity to query. Default ``"robot"``.

    Returns:
        Tensor of shape ``(num_envs,)`` — negative sum of soft-limit
        violations across joints.
    """
    rd = env.get_robot_data(entity_name)
    dof_pos = rd.joint_pos
    lower, upper = rd.joint_pos_limits
    lower = lower * soft_limit_factor
    upper = upper * soft_limit_factor
    out_of_limits = -(dof_pos - lower).clamp(max=0.0)
    out_of_limits += (dof_pos - upper).clamp(min=0.0)
    return -torch.sum(out_of_limits, dim=-1)
