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


def _actuated_joint_ids(env: World) -> torch.Tensor | None:
    """Return act_manager._joint_ids if it exists (MuJoCo needs reindexing)."""
    ids = getattr(env.act_manager, "_joint_ids", None)
    if ids is None:
        return None
    # Only return if it's actually a permutation (not identity).
    n = len(ids)
    if n > 0 and not torch.equal(ids, torch.arange(n, device=ids.device)):
        return ids
    return None


def dof_pos(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Actuated joint positions in act_manager order.

    Returns:
        Tensor of shape (num_envs, num_joints).
    """
    pos = env.get_robot_data(entity_name).joint_pos
    joint_ids = _actuated_joint_ids(env)
    if joint_ids is not None:
        return pos[:, joint_ids]
    return pos


def dof_vel(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Actuated joint velocities in act_manager order.

    Returns:
        Tensor of shape (num_envs, num_joints).
    """
    vel = env.get_robot_data(entity_name).joint_vel
    joint_ids = _actuated_joint_ids(env)
    if joint_ids is not None:
        return vel[:, joint_ids]
    return vel


def dof_pos_nominal_difference(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Joint positions relative to nominal (default) positions, in act_manager order.

    Returns:
        Tensor of shape (num_envs, num_joints).
    """
    return dof_pos(env, entity_name) - env.act_manager.offset


def base_height(env: World, entity_name: str = "robot") -> torch.Tensor:
    """Base height (z-coordinate) above world origin.

    Returns:
        Tensor of shape (num_envs, 1).
    """
    return env.get_robot_data(entity_name).root_link_pos_w[:, 2:3]


def prev_processed_actions(env: World) -> torch.Tensor:
    """Current step's processed actions (used as observation input).

    Note: Despite the name, this returns the *current* processed actions,
    matching the existing Newton/Genesis observation behavior.

    Returns:
        Tensor of shape (num_envs, num_actions).
    """
    return env.act_manager.processed_actions.clone()


def raw_actions(env: World) -> torch.Tensor:
    """Current step's raw (unprocessed) actions.

    Returns:
        Tensor of shape (num_envs, num_actions).
    """
    return env.act_manager.raw_actions
