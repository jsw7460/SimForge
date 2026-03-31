"""MuJoCo/mjlab proprioception observation functions.

These functions extract proprioceptive information from mjlab environments,
accessing data through the Entity.data interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.utils import EnvStepCache

if TYPE_CHECKING:
    from rlworld.rl.envs.mujoco import MujocoEnv


def _get_robot_data(env: "MujocoEnv"):
    """Get robot entity data from scene manager."""
    return env.scene_manager.robot.data


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse quaternion rotation to vector.

    Args:
        quat: Quaternion (w, x, y, z) of shape [..., 4]
        vec: Vector of shape [..., 3]

    Returns:
        Rotated vector of shape [..., 3]
    """
    # Quaternion conjugate (inverse for unit quaternion)
    quat_conj = quat.clone()
    quat_conj[..., 1:] = -quat_conj[..., 1:]

    # Apply rotation: q_conj * v * q
    # Using quaternion-vector multiplication formula
    w, x, y, z = quat_conj[..., 0], quat_conj[..., 1], quat_conj[..., 2], quat_conj[..., 3]
    vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]

    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    # result = v + w * t + cross(q.xyz, t)
    result_x = vx + w * tx + (y * tz - z * ty)
    result_y = vy + w * ty + (z * tx - x * tz)
    result_z = vz + w * tz + (x * ty - y * tx)

    return torch.stack([result_x, result_y, result_z], dim=-1)

# =============================================================================
# Root state observations
# =============================================================================

@EnvStepCache()
def projected_gravity(env: "MujocoEnv") -> torch.Tensor:
    """Get gravity vector projected to body frame.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    robot_data = _get_robot_data(env)
    return robot_data.projected_gravity_b


@EnvStepCache()
def base_lin_vel(env: "MujocoEnv") -> torch.Tensor:
    """Get base linear velocity in body frame.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    robot_data = _get_robot_data(env)
    return robot_data.root_link_lin_vel_b


@EnvStepCache()
def base_ang_vel(env: "MujocoEnv") -> torch.Tensor:
    """Get base angular velocity in body frame.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    robot_data = _get_robot_data(env)
    return robot_data.root_link_ang_vel_b


@EnvStepCache()
def base_pos(env: "MujocoEnv") -> torch.Tensor:
    """Get base position in world frame.

    Returns:
        Tensor of shape [num_envs, 3]
    """
    robot_data = _get_robot_data(env)
    return robot_data.root_link_pos_w


@EnvStepCache()
def base_quat(env: "MujocoEnv") -> torch.Tensor:
    """Get base quaternion orientation in world frame.

    Returns:
        Tensor of shape [num_envs, 4] (w, x, y, z)
    """
    robot_data = _get_robot_data(env)
    return robot_data.root_link_quat_w


@EnvStepCache()
def base_height(env: "MujocoEnv") -> torch.Tensor:
    """Get base height above ground.

    Returns:
        Tensor of shape [num_envs, 1]
    """
    robot_data = _get_robot_data(env)
    return robot_data.root_link_pos_w[:, 2:3]


# =============================================================================
# Joint state observations
# =============================================================================

@EnvStepCache()
def dof_pos(env: "MujocoEnv") -> torch.Tensor:
    """Get actuated joint positions.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    robot_data = _get_robot_data(env)
    joint_ids = env.act_manager._joint_ids
    return robot_data.joint_pos[:, joint_ids]


@EnvStepCache()
def dof_pos_nominal_difference(env: "MujocoEnv") -> torch.Tensor:
    """Get joint positions relative to nominal (default) positions.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return dof_pos(env) - env.act_manager.offset


@EnvStepCache()
def dof_vel(env: "MujocoEnv") -> torch.Tensor:
    """Get actuated joint velocities.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    robot_data = _get_robot_data(env)
    joint_ids = env.act_manager._joint_ids
    return robot_data.joint_vel[:, joint_ids]


@EnvStepCache()
def all_joint_pos(env: "MujocoEnv") -> torch.Tensor:
    """Get all joint positions (including non-actuated).

    Returns:
        Tensor of shape [num_envs, num_joints]
    """
    robot_data = _get_robot_data(env)
    return robot_data.joint_pos


@EnvStepCache()
def all_joint_vel(env: "MujocoEnv") -> torch.Tensor:
    """Get all joint velocities (including non-actuated).

    Returns:
        Tensor of shape [num_envs, num_joints]
    """
    robot_data = _get_robot_data(env)
    return robot_data.joint_vel


@EnvStepCache()
def joint_pos_rel(env: "MujocoEnv") -> torch.Tensor:
    """Get joint positions relative to default positions.

    Alias for dof_pos_nominal_difference.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return dof_pos_nominal_difference(env)


@EnvStepCache()
def joint_vel_rel(env: "MujocoEnv") -> torch.Tensor:
    """Get joint velocities (default velocity is assumed zero).

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return dof_vel(env)


# =============================================================================
# Action observations
# =============================================================================

@EnvStepCache()
def raw_actions(env: "MujocoEnv") -> torch.Tensor:
    """Get raw (unprocessed) actions from current step.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return env.act_manager.raw_actions


@EnvStepCache()
def processed_actions(env: "MujocoEnv") -> torch.Tensor:
    """Get processed actions from current step.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return env.act_manager.processed_actions


@EnvStepCache()
def prev_processed_actions(env: "MujocoEnv") -> torch.Tensor:
    """Get processed actions from previous step.

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return env.act_manager.prev_processed_actions.clone()


@EnvStepCache()
def last_action(env: "MujocoEnv") -> torch.Tensor:
    """Get last action (alias for processed_actions).

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return processed_actions(env)


@EnvStepCache()
def last_raw_action(env: "MujocoEnv") -> torch.Tensor:
    """Get last action (alias for processed_actions).

    Returns:
        Tensor of shape [num_envs, num_actions]
    """
    return env.act_manager.raw_actions


# =============================================================================
# Command observations
# =============================================================================

@EnvStepCache()
def command_velocity(env: "MujocoEnv") -> torch.Tensor:
    """Get velocity command.

    Returns:
        Tensor of shape [num_envs, 3] (lin_x, lin_y, ang_z)
    """
    return env.command_manager.get_commands_tensor()


@EnvStepCache()
def generated_commands(env: "MujocoEnv") -> torch.Tensor:
    """Get generated commands (alias for command_velocity).

    Returns:
        Tensor of shape [num_envs, 3]
    """
    return command_velocity(env)


# =============================================================================
# Contact/feet observations
# =============================================================================

@EnvStepCache()
def foot_height(
    env: "MujocoEnv",
    site_names: tuple[str, ...],
    asset_cfg_name: str = "robot",
) -> torch.Tensor:
    """Get height of specified feet sites above ground."""
    robot = env.scene_manager.get_entity(asset_cfg_name)
    site_ids, _ = robot.find_sites(site_names)
    return robot.data.site_pos_w[:, site_ids, 2]


@EnvStepCache()
def foot_air_time(env: "MujocoEnv") -> torch.Tensor:
    """Get current air time of feet.

    Returns:
        Tensor of shape [num_envs, num_feet]
    """
    contact_data = env.contact_manager
    return contact_data.current_air_time


@EnvStepCache()
def foot_contact(env: "MujocoEnv") -> torch.Tensor:
    """Get binary contact state of feet.

    Returns:
        Tensor of shape [num_envs, num_feet] with values 0.0 or 1.0
    """
    contact_data = env.contact_manager
    return contact_data.is_contact.float()


@EnvStepCache()
def foot_contact_forces(env: "MujocoEnv") -> torch.Tensor:
    """Get contact forces on feet (log-scaled).

    Returns log1p scaled forces to compress large force magnitudes.

    Returns:
        Tensor of shape [num_envs, num_feet * 3]
    """
    contact_data = env.contact_manager
    forces = contact_data.contact_force  # [B, N, 3]
    forces_flat = forces.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


@EnvStepCache()
def foot_contact_time(env: "MujocoEnv") -> torch.Tensor:
    """Get current contact time of feet.

    Returns:
        Tensor of shape [num_envs, num_feet]
    """
    contact_data = env.contact_manager
    return contact_data.current_contact_time


@EnvStepCache()
def relative_sites_pos(
    env: "MujocoEnv",
    base_name: str,
    sites: tuple[str, ...],
) -> torch.Tensor:
    """Get site positions relative to base body in base frame.

    Args:
        env: The MujocoEnv environment.
        base_name: Name of the base body.
        sites: Tuple of site names.

    Returns:
        Tensor of shape [num_envs, len(sites) * 3].
    """

    robot = env.scene_manager.robot
    robot_data = robot.data

    # Get base body id and data
    base_body_ids, _ = robot.find_bodies([base_name])
    base_body_id = base_body_ids[0]
    base_pos_w = robot_data.body_link_pos_w[:, base_body_id]  # [num_envs, 3]
    base_quat_w = robot_data.body_link_quat_w[:, base_body_id]  # [num_envs, 4]

    # Get site ids
    site_ids, _ = robot.find_sites(sites)  # list[int], list[str]
    site_pos_w = robot_data.site_pos_w[:, site_ids]  # [num_envs, num_sites, 3]

    # Compute relative position in world frame
    rel_pos_w = site_pos_w - base_pos_w.unsqueeze(1)  # [num_envs, num_sites, 3]

    # Rotate to base frame
    num_sites = len(site_ids)
    base_quat_expanded = base_quat_w.unsqueeze(1).expand(-1, num_sites, -1)
    rel_pos_b = quat_apply_inverse(base_quat_expanded, rel_pos_w)

    return rel_pos_b.reshape(env.num_envs, -1)  # [num_envs, num_sites * 3]