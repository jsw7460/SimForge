"""Unified reward terms using the RobotData interface.

All functions accept any ``World`` subclass and read state exclusively
through ``env.robot_data``, making them simulator-agnostic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


def track_lin_vel(env: World, std: float = 0.25) -> torch.Tensor:
    """Reward for tracking commanded linear velocity in xy plane.

    Returns:
        Tensor of shape (num_envs,).
    """
    target = torch.stack(
        [env.command_manager.lin_vel_x, env.command_manager.lin_vel_y], dim=1
    )
    actual = env.robot_data.root_link_lin_vel_b[:, :2]
    error = torch.sum(torch.square(target - actual), dim=1)
    return torch.exp(-error / std ** 2)


def track_ang_vel(env: World, std: float = 0.25) -> torch.Tensor:
    """Reward for tracking commanded angular velocity (yaw).

    Returns:
        Tensor of shape (num_envs,).
    """
    actual_yaw_rate = env.robot_data.root_link_ang_vel_b[:, 2]
    error = torch.square(env.command_manager.ang_vel - actual_yaw_rate)
    return torch.exp(-error / std ** 2)


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


def flat_orientation(env: World) -> torch.Tensor:
    """Penalty for non-flat orientation (roll/pitch deviation from upright).

    Returns:
        Tensor of shape (num_envs,).
    """
    gravity_b = env.robot_data.projected_gravity_b
    return -torch.sum(torch.square(gravity_b[:, :2]), dim=1)


def similar_to_default(env: World) -> torch.Tensor:
    """Penalty for deviating from default joint positions.

    Returns:
        Tensor of shape (num_envs,).
    """
    return -torch.sum(
        torch.abs(env.robot_data.joint_pos - env.act_manager.offset), dim=1
    )
