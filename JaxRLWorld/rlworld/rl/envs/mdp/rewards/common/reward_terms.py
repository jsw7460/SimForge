"""Unified reward terms using the RobotData interface.

All functions accept any ``World`` subclass and read state exclusively
through ``env.get_robot_data(entity_name)``, making them simulator-agnostic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


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


def similar_to_default(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Penalty for deviating from default joint positions.

    Returns:
        Tensor of shape (num_envs,).
    """
    return -torch.sum(
        torch.abs(env.get_robot_data(entity_name).joint_pos - env.act_manager.offset), dim=1
    )
