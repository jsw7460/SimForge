"""Unified proprioception observations using the RobotData interface.

All functions accept any ``World`` subclass and read state exclusively
through ``env.get_robot_data(entity_name)``, making them simulator-agnostic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


def base_lin_vel(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Base linear velocity in body frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    return env.get_robot_data(entity_name).root_link_lin_vel_b


def base_ang_vel(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Base angular velocity in body frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    return env.get_robot_data(entity_name).root_link_ang_vel_b


def projected_gravity(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Gravity vector projected into the body frame.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    return env.get_robot_data(entity_name).projected_gravity_b


def dof_pos(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Actuated joint positions.

    Returns:
        Tensor of shape (num_envs, num_joints).
    """
    return env.get_robot_data(entity_name).joint_pos


def dof_vel(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Actuated joint velocities.

    Returns:
        Tensor of shape (num_envs, num_joints).
    """
    return env.get_robot_data(entity_name).joint_vel


def dof_pos_nominal_difference(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Joint positions relative to nominal (default) positions.

    Returns:
        Tensor of shape (num_envs, num_joints).
    """
    return env.get_robot_data(entity_name).joint_pos - env.act_manager.offset


def base_height(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Base height (z-coordinate) above world origin.

    Returns:
        Tensor of shape (num_envs, 1).
    """
    return env.get_robot_data(entity_name).root_link_pos_w[:, 2:3]
