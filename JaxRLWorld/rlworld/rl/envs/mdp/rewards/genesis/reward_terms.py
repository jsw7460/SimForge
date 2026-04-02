from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesis import inv_quat, transform_by_quat, transform_quat_by_quat, quat_to_xyz
from genesis.engine.entities import RigidEntity
from rlworld.rl.envs.mdp.observations.genesis import proprioception, state
from rlworld.rl.envs.utils import EnvStepCache
from rlworld.rl.utils import entity_utils as eu
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv, GenesisLocomotionEnv


def tracking_lin_vel(env: GenesisEnv) -> torch.Tensor:
    """Reward for tracking commanded linear velocity in xy plane
       Returns exponential of negative squared error between commanded and actual velocity"""
    target_lin_vel = torch.stack([env.command_manager.lin_vel_x, env.command_manager.lin_vel_y], dim=1)
    base_lin_vel = state.base_lin_vel(env)
    lin_vel_error = torch.sum(
        torch.square(target_lin_vel - base_lin_vel[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / env.reward_cfg.tracking_sigma)


def tracking_ang_vel(env: GenesisEnv, entity_name: str = "robot", base_name: str = "base") -> torch.Tensor:
    """Reward for tracking commanded angular velocity (yaw)
       Returns exponential of negative squared error between commanded and actual yaw rate"""

    base_ang_vel = proprioception.imu_ang_vel(env, entity_name, base_name)
    ang_vel_error = torch.square(env.command_manager.ang_vel - base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / env.reward_cfg.tracking_sigma)


def lin_vel_z(env: GenesisEnv) -> torch.Tensor:
    """Penalty for vertical movement
       Returns negative squared vertical velocity to discourage unwanted up/down motion"""
    base_lin_vel = state.base_lin_vel(env)
    return -torch.square(base_lin_vel[:, 2])


def base_height(env: GenesisEnv) -> torch.Tensor:
    """Penalty for deviating from target base height
       Returns negative squared error from desired base height"""
    # Position w.r.t. standard basis

    base_pos = state.base_pos(env)
    return -torch.square(base_pos[:, 2] - env.command_manager.base_height)


def adaptive_base_height(env: GenesisEnv) -> torch.Tensor:
    """Penalty for deviating from target base height above terrain
           Returns negative squared error from desired base height above local terrain"""

    base_pos = state.base_pos(env)

    # Get terrain height at robot's current x, y position
    terrain: RigidEntity = env.scene_manager["base_entity"]
    terrain_geom = terrain.geoms[0]
    height_field = terrain_geom.metadata["height_field"]
    terrain_morph = terrain.morph

    # Convert to tensor if needed
    if not isinstance(height_field, torch.Tensor):
        height_field = torch.tensor(height_field, dtype=torch.float32, device=env.device)

    # Convert robot positions to heightfield indices
    horizontal_scale = terrain_morph.horizontal_scale
    vertical_scale = terrain_morph.vertical_scale

    indices_x = (base_pos[:, 0] / horizontal_scale).long()
    indices_y = (base_pos[:, 1] / horizontal_scale).long()

    # Clamp to valid range
    indices_x = torch.clamp(indices_x, 0, height_field.shape[0] - 1)
    indices_y = torch.clamp(indices_y, 0, height_field.shape[1] - 1)

    # Get terrain height at robot position
    terrain_heights = height_field[indices_x, indices_y] * vertical_scale

    # Calculate height above terrain
    height_above_terrain = base_pos[:, 2] - terrain_heights
    # Penalize deviation from desired height above terrain
    return -torch.square(height_above_terrain - env.command_manager.base_height)


def action_rate(env: GenesisEnv) -> torch.Tensor:
    """Penalty for sudden joint action changes
       Returns negative squared difference between consecutive joint actions"""
    return -torch.sum(
        torch.square(env.act_manager.processed_actions - env.act_manager.prev_processed_actions), dim=1
    )


def similar_to_default(env: GenesisEnv) -> torch.Tensor:
    """Penalty for deviating from default joint positions
       Returns negative absolute difference from default pose"""

    dof_pos = proprioception.dof_pos(env)
    return -torch.sum(torch.abs(dof_pos - env.act_manager.offset), dim=1)


def penalize_invalid_contact(
    env: GenesisEnv,
    contact_allowed_links: list[str],
    entity_name: str = "robot",
    exclude_self_contact: bool = True,
):
    entity = env.scene_manager[entity_name]

    # Get allowed link indices (global)
    allowed_ids, _ = eu.find_links(entity, contact_allowed_links, global_ids=True)
    allowed_ids_tensor = torch.tensor(allowed_ids, dtype=torch.int32, device=env.device)

    # Get all robot link indices
    all_robot_link_ids = torch.arange(
        entity.link_start,
        entity.link_end,
        dtype=torch.int32,
        device=env.device
    )

    # Get contact information
    contact_info = entity.get_contacts(exclude_self_contact=exclude_self_contact)

    valid_mask = contact_info["valid_mask"]
    link_a = contact_info["link_a"]
    link_b = contact_info["link_b"]

    # Check if links are robot links
    is_robot_link_a = torch.isin(link_a, all_robot_link_ids)
    is_robot_link_b = torch.isin(link_b, all_robot_link_ids)

    # Check if links are in allowed list
    link_a_allowed = torch.isin(link_a, allowed_ids_tensor)
    link_b_allowed = torch.isin(link_b, allowed_ids_tensor)

    # Invalid contact: valid AND robot link involved AND NOT allowed
    invalid_contact_a = valid_mask & is_robot_link_a & ~link_a_allowed
    invalid_contact_b = valid_mask & is_robot_link_b & ~link_b_allowed

    # Apply masks to contact forces
    contact_force_a = contact_info["force_a"] * invalid_contact_a.unsqueeze(-1)
    contact_force_b = contact_info["force_b"] * invalid_contact_b.unsqueeze(-1)

    # Count contacts over 1N threshold
    return - (torch.sum(torch.norm(contact_force_a, dim=-1) > 1.0, dim=-1)
           + torch.sum(torch.norm(contact_force_b, dim=-1) > 1.0, dim=-1))


def wtw_collision(
    env: GenesisEnv,
    contact_group: str = "body_ground_contact",
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """WTW collision: count non-foot bodies with contact force > threshold."""
    force = env.contact_manager.contact_force(contact_group)  # (num_envs, N, 3)
    force_mag = torch.norm(force, dim=-1)  # (num_envs, N)
    return -torch.sum((force_mag > force_threshold).float(), dim=-1)


def penalize_ang_vel_xy(
    env: GenesisEnv,
    entity_name: str = "robot",
    base_name: str = "base"
) -> torch.Tensor:
    """
    Penalize roll and pitch angular velocities in the body frame.

    Discourages the robot from tilting or rolling while allowing yaw rotation.
    Returns negative squared magnitude of xy angular velocities.

    Args:
        env: Locomotion environment
        entity_name: Name of the robot entity
        base_name: Name of the base link

    Returns:
        Penalty values for each environment (shape: [n_envs])
    """
    entity = env.scene_manager[entity_name]
    base_link = entity.get_link(base_name)

    # Transform angular velocity from world frame to body frame
    world_ang_vel = base_link.get_ang()
    base_quat = state.base_quat(env)
    body_ang_vel = transform_by_quat(world_ang_vel, inv_quat(base_quat))

    # Penalize roll (x) and pitch (y) rotations, allow yaw (z)
    roll_pitch_vel_squared = torch.sum(torch.square(body_ang_vel[:, :2]), dim=-1)

    return -roll_pitch_vel_squared


def penalize_nonflat_by_gravity(env: GenesisEnv):
    projected_gravity = proprioception.projected_gravity(env)
    return - torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)


def penalize_torques(env: GenesisEnv, entity_name: str = "robot"):
    entity = env.scene_manager[entity_name]
    torque = entity.get_dofs_control_force()
    return - torch.sum(torch.square(torque), dim=-1)


def penalize_dof_vel(env: GenesisEnv, entity_name: str = "robot"):
    dof_vel = proprioception.dof_vel(env, entity_name=entity_name)
    return - torch.sum(torch.square(dof_vel), dim=-1)

def penalize_dof_acc(env: GenesisEnv, entity_name: str = "robot"):
    """Todo"""


def penalize_base_acc(env: GenesisEnv, entity_name: str = "robot", base_name: str = "base"):
    entity = env.scene_manager[entity_name]
    base_idx_local = entity.get_link(base_name).idx_local
    base_acc = entity.get_links_acc(base_idx_local).squeeze(1)
    return - torch.sum(torch.square(base_acc), dim=-1)


def penalize_dof_pos_limits(env: GenesisEnv):
    """Todo"""


def penalize_dof_vel_limits(env: GenesisEnv):
    """Todo"""


def penalize_torque_limits(env: GenesisEnv, entity_name: str = "robot", soft_torque_limit: float = 1.0):
    """Maybe not required: Genesis internally clips torque limits."""


def penalize_power(env: GenesisEnv, entity_name: str = "robot"):
    entity = env.scene_manager[entity_name]
    torque = entity.get_dofs_control_force(env.act_manager.actuated_dof_ids)
    velocity = entity.get_dofs_velocity(env.act_manager.actuated_dof_ids)

    return -torch.sum((torque * velocity).clip(min=0.0), dim=-1)


@EnvStepCache()
def feet_rpy(
    env: GenesisEnv,
    feet_links: tuple[str, ...],
    entity_name: str = "robot"
):
    entity = env.scene_manager[entity_name]
    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False)

    # Get feet quaternions (world frame)
    feet_quat_world = entity.get_links_quat(links_idx_local=links_idx_local)  # [n_envs, n_feet, 4]

    # Get base quaternion
    base_quat = state.base_quat(env)  # [n_envs, 4]
    inv_base_quat = inv_quat(base_quat)  # [n_envs, 4]

    # Transform to body frame (vectorized!)
    # Broadcasting: [n_envs, 1, 4] * [n_envs, n_feet, 4] -> [n_envs, n_feet, 4]
    feet_quat_body = transform_quat_by_quat(
        inv_base_quat.unsqueeze(1),  # [n_envs, 1, 4]
        feet_quat_world  # [n_envs, n_feet, 4]
    )  # [n_envs, n_feet, 4]

    # Convert to RPY in body frame
    feet_rpy_body = quat_to_xyz(
        feet_quat_body,
        rpy=True,
        degrees=False
    )

    return feet_rpy_body


def penalize_feet_slip(
    env: GenesisEnv,
    feet_links: tuple[str, ...],
    entity_name: str = "robot"
) -> torch.Tensor:
    """
    Penalize foot velocities when feet are in contact with ground.

    Args:
        env: Locomotion environment
        feet_links: Names of foot links
        entity_name: Name of the robot entity

    Returns:
        Penalty values for each environment (shape: [n_envs])
    """
    entity = env.scene_manager[entity_name]

    # Get foot link indices and velocities
    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False)
    feet_vel = entity.get_links_vel(links_idx_local=links_idx_local)  # [num_envs, num_feet, 3]

    # Get contact indicator
    contact_indicator = state.contact_indicator(
        env, entity_name=entity_name, links=feet_links
    )  # [num_envs, num_feet]

    # Calculate squared velocity magnitude for each foot (xy only)
    vel_magnitude_sq = torch.sum(torch.square(feet_vel[..., :2]), dim=-1)  # [num_envs, num_feet]

    # Apply contact mask and sum over all feet
    penalty = torch.sum(vel_magnitude_sq * contact_indicator, dim=-1)  # [num_envs]

    # Skip first episode step if needed
    penalty = penalty * (env.termination_manager.episode_length_buf > 1).float()
    return -penalty


def penalize_feet_roll(
    env: GenesisEnv,
    feet_links: tuple[str, ...],
    entity_name: str = "robot"
) -> torch.Tensor:
    """
    Penalize roll angle of feet in body frame.
    Encourages feet to stay flat relative to robot's orientation.
    """
    feet_rpy_body_frame = feet_rpy(env, feet_links, entity_name)

    # Penalize roll component
    feet_roll = feet_rpy_body_frame[..., 0]  # [n_envs, n_feet]
    return -torch.sum(torch.square(feet_roll), dim=-1)


def penalize_feet_yaw_difference(
    env: GenesisEnv,
    feet_links: tuple[str, str],
    entity_name: str = "robot"
) -> torch.Tensor:
    """
    This is for humanoid bipedal robot.
    Penalize yaw difference between two feet.

    Args:
        env: Locomotion environment
        feet_links: Exactly 2 foot link names (e.g., ("FL_foot", "FR_foot"))
        entity_name: Name of the robot entity

    Returns:
        Penalty values for each environment (shape: [n_envs])
    """

    feet_rpy_body = feet_rpy(env, feet_links, entity_name)  # [num_envs, 2, 3]
    feet_yaw = feet_rpy_body[..., 2]  # [num_envs, 2]

    if feet_yaw.shape[-1] != 2:
        raise ValueError("Feet yaw difference between two feet")

    yaw0 = feet_yaw[:, 0]  # [num_envs]
    yaw1 = feet_yaw[:, 1]  # [num_envs]

    # Wrapped difference
    yaw_diff = (yaw0 - yaw1 + torch.pi) % (2 * torch.pi) - torch.pi

    return -torch.square(yaw_diff)


def penalize_feet_yaw_mean_deviation(env, feet_links: tuple[str, ...], entity_name="robot"):
    # This is calculated in world frame
    entity = env.scene_manager[entity_name]
    links_idx, _ = eu.find_links(entity, list(feet_links), global_ids=False)

    feet_quat = entity.get_links_quat(links_idx_local=links_idx)
    feet_yaw = quat_to_xyz(feet_quat, rpy=True, degrees=False)[..., 2]

    mean = feet_yaw.mean(-1) + torch.pi * (torch.abs(feet_yaw[:, 1] - feet_yaw[:, 0]) > torch.pi).float()
    base_yaw = quat_to_xyz(state.base_quat(env), rpy=True, degrees=False)[:, 2]
    error = (base_yaw - mean + torch.pi) % (2 * torch.pi) - torch.pi

    return -torch.square(error)


def penalize_feet_distance(
    env: GenesisEnv,
    feet_links: tuple[str, str],
    feet_distance_ref: float,
    entity_name: str = "robot"
) -> torch.Tensor:
    """
    Reward feet maintaining target lateral distance in body frame.

    Measures left-right distance between feet relative to robot's heading,
    ignoring forward-backward separation.
    """
    entity = env.scene_manager[entity_name]
    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False)

    # Get feet positions and transform to body frame
    feet_pos_world = entity.get_links_pos(links_idx_local=links_idx_local)
    base_quat = state.base_quat(env)

    # Transform: [num_envs, 2, 3] with broadcasting
    feet_pos_body = transform_by_quat(
        feet_pos_world,
        inv_quat(base_quat).unsqueeze(1)
    )

    # Lateral distance = y-component difference in body frame
    lateral_distance = torch.abs(feet_pos_body[:, 1, 1] - feet_pos_body[:, 0, 1])

    # Reward proximity to target distance
    return - torch.clamp(feet_distance_ref - lateral_distance, min=0.0, max=0.1)


def reward_gait_pattern(
    env: GenesisLocomotionEnv,
    entity_name: str = "robot"
) -> torch.Tensor:
    """Reward for matching desired gait pattern.

    Encourages feet to follow the gait phase:
    - Swing phase: foot should NOT be in contact
    - Stance phase: foot SHOULD be in contact

    Args:
        env: Locomotion environment with gait_manager.
        entity_name: Name of the robot entity.

    Returns:
        Reward tensor [num_envs], range [0, 1].
    """
    feet_links = tuple(env.gait_manager.foot_names)

    # Get contact state and desired gait phase
    is_contact = state.contact_indicator(env, entity_name=entity_name, links=feet_links).bool()
    swing_mask = env.gait_manager.get_swing_mask()  # [num_envs, num_feet]

    # Check correctness for both swing and stance
    correct_swing = ~is_contact & swing_mask  # Should be airborne
    correct_stance = is_contact & ~swing_mask  # Should be grounded

    # Count correct feet
    num_correct = torch.sum(correct_swing.float() + correct_stance.float(), dim=-1)

    # Normalize to [0, 1]
    num_feet = swing_mask.shape[-1]
    return num_correct / num_feet


def reward_feet_air_time(
    env: GenesisEnv,
    threshold: float = 0.1,
    command_threshold: float = 0.1,
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Reward for taking long steps.

    Encourages the robot to lift its feet off the ground for at least
    `threshold` seconds before landing. Only active when moving.

    Args:
        env: Locomotion environment with ContactManager.
        threshold: Minimum air time (seconds) to receive reward.
        command_threshold: Minimum command velocity magnitude to activate reward.
        contact_group: Name of the contact group to use.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    first_contact = env.contact_manager.compute_first_contact(contact_group)
    last_air_time = env.contact_manager.last_air_time(contact_group)

    reward = torch.sum((last_air_time - threshold) * first_contact, dim=-1)

    command_vel = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y
    ], dim=-1)
    is_moving = torch.norm(command_vel, dim=-1) > command_threshold

    return reward * is_moving


def reward_feet_height_exp(
    env: GenesisEnv,
    feet_links: str | list[str],
    target_height: float = 0.08,
    sigma: float = 0.01,
    entity_name: str = "robot",
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Reward feet reaching target height during swing (exponential kernel)."""
    entity = env.scene_manager[entity_name]

    if isinstance(feet_links, str):
        feet_links = [feet_links]

    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False, preserve_order=True)
    feet_pos = entity.get_links_pos(links_idx_local=links_idx_local)
    feet_height = feet_pos[..., 2]  # z pos

    is_contact = env.contact_manager.is_contact(contact_group)
    is_swing = ~is_contact

    # Exponential reward for being close to target height
    height_error = torch.square(feet_height - target_height)
    height_reward = torch.exp(-height_error / sigma)

    # Only count during swing
    reward = torch.sum(height_reward * is_swing.float(), dim=-1)

    return reward


def penalize_impact_force(
    env: GenesisEnv,
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Penalize contact force at the moment of landing."""
    forces_3d = env.contact_manager.contact_force(contact_group)  # (num_envs, num_links, 3)
    forces = torch.norm(forces_3d, dim=-1)  # (num_envs, num_links)
    first_contact = env.contact_manager.compute_first_contact(contact_group)

    return -torch.sum(forces * first_contact.float(), dim=-1)


def reward_alive(env: GenesisEnv) -> torch.Tensor:
    return torch.ones((env.num_envs,))


def penalize_joint_deviation_l1(
    env: GenesisEnv,
    joints: str | list[str],
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one (L1 norm).

    Args:
        env: Genesis environment.
        joints: Joint name(s) or regex pattern(s).

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    actuated_joint_names = env.act_manager._actuated_joint_names

    indices_in_actuated, _ = string_utils.resolve_matching_names(
        joints, actuated_joint_names
    )

    dof_pos = proprioception.dof_pos(env)
    default_pos = env.act_manager.offset

    deviation = dof_pos[:, indices_in_actuated] - default_pos[:, indices_in_actuated]

    return -torch.sum(torch.abs(deviation), dim=-1)



def penalize_hip_deviation(
    env: GenesisEnv,
    hip_joints: str | tuple[str, ...]
) -> torch.Tensor:
    """
    Penalize hip joint angles deviating from nominal pose.

    Args:
        env: GenesisEnv instance
        hip_joints: Joint name pattern (supports regex) or tuple of names
        entity_name: Name of the robot entity
    """
    # Find indices within actuated joints
    indices, _ = string_utils.resolve_matching_names(
        hip_joints,
        env.act_manager._actuated_joint_names
    )

    dof_pos = proprioception.dof_pos(env)[:, indices]
    nominal_pos = env.act_manager.offset[:, indices]

    return -torch.sum(torch.square(dof_pos - nominal_pos), dim=-1)


def penalize_hip_deviation_huber(
    env: GenesisEnv,
    hip_joints: str | tuple[str, ...],
    threshold: float = 0.3,
) -> torch.Tensor:
    """
    Penalize hip joint angles deviating from nominal pose (Huber loss).

    Below threshold: gentle quadratic penalty
    Above threshold: linear penalty

    Args:
        env: GenesisEnv instance
        hip_joints: Joint name pattern (supports regex) or tuple of names
        threshold: Deviation threshold for switching from quadratic to linear
    """
    indices, _ = string_utils.resolve_matching_names(
        hip_joints,
        env.act_manager._actuated_joint_names
    )

    dof_pos = proprioception.dof_pos(env)[:, indices]
    nominal_pos = env.act_manager.offset[:, indices]

    error = torch.abs(dof_pos - nominal_pos)
    penalty = torch.where(
        error < threshold,
        0.5 * torch.square(error) / threshold,
        error - 0.5 * threshold
    )

    return -torch.sum(penalty, dim=-1)


# ── Walk-These-Ways reward terms (Genesis) ───────────────────────────────

def wtw_feet_slip(
    env: GenesisLocomotionEnv,
    entity_name: str = "robot",
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """WTW feet slip: penalize foot xy velocity when in contact OR was in contact.

    Uses contact OR prev_contact (2-step filtering), matching WTW exactly.
    """
    feet_links = tuple(env.gait_manager.foot_names)
    entity = env.scene_manager[entity_name]
    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False)
    feet_vel = entity.get_links_vel(links_idx_local=links_idx_local)

    # Contact: current OR previous step
    contact = env.contact_manager.is_contact(contact_group, order=feet_links)
    prev_contact = env.contact_manager.prev_is_contact(contact_group)
    contact_filt = contact | prev_contact

    vel_sq = torch.sum(torch.square(feet_vel[..., :2]), dim=-1)
    return -torch.sum(contact_filt.float() * vel_sq, dim=-1)


def wtw_tracking_contacts_shaped_force(
    env: GenesisLocomotionEnv,
    gait_force_sigma: float = 100.0,
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """WTW: penalize foot contact force when foot should be in swing.

    reward = mean_over_feet[ -(1 - desired_contact) * (1 - exp(-force² / σ)) ]
    """
    foot_forces_3d = env.contact_manager.contact_force(contact_group)  # (num_envs, num_feet, 3)
    foot_forces = torch.norm(foot_forces_3d, dim=-1)  # (num_envs, num_feet)

    desired_contact = env.gait_manager.desired_contact_states

    reward = -(1.0 - desired_contact) * (1.0 - torch.exp(-foot_forces ** 2 / gait_force_sigma))
    return reward.mean(dim=-1)


def wtw_tracking_contacts_shaped_vel(
    env: GenesisLocomotionEnv,
    gait_vel_sigma: float = 10.0,
    entity_name: str = "robot",
) -> torch.Tensor:
    """WTW: penalize foot velocity when foot should be in stance.

    reward = mean_over_feet[ -(desired_contact) * (1 - exp(-vel² / σ)) ]
    """
    feet_links = tuple(env.gait_manager.foot_names)
    entity = env.scene_manager[entity_name]
    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False)
    feet_vel = entity.get_links_vel(links_idx_local=links_idx_local)

    foot_vel_norm = torch.norm(feet_vel, dim=-1)
    desired_contact = env.gait_manager.desired_contact_states

    reward = -(desired_contact * (1.0 - torch.exp(-foot_vel_norm ** 2 / gait_vel_sigma)))
    return reward.mean(dim=-1)


def wtw_feet_clearance_cmd_linear(
    env: GenesisLocomotionEnv,
    foot_radius: float = 0.02,
    entity_name: str = "robot",
) -> torch.Tensor:
    """WTW: penalize foot height error during swing, scaled by commanded footswing height.

    target_height = footswing_height_cmd * phase_triangle + foot_radius
    penalty = (target - actual)² * (1 - desired_contact)
    """
    feet_links = tuple(env.gait_manager.foot_names)
    entity = env.scene_manager[entity_name]
    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False)
    feet_pos = entity.get_links_pos(links_idx_local=links_idx_local)
    foot_height = feet_pos[..., 2]

    # Triangle wave from foot phases: peaks at mid-swing (phase=0.75)
    foot_phases = env.gait_manager.foot_phases
    phases = 1.0 - torch.abs(
        1.0 - torch.clip((foot_phases * 2.0) - 1.0, 0.0, 1.0) * 2.0
    )

    footswing_height = env.command_manager.footswing_height
    target_height = footswing_height.unsqueeze(1) * phases + foot_radius

    desired_contact = env.gait_manager.desired_contact_states
    clearance_error = torch.square(target_height - foot_height) * (1.0 - desired_contact)
    return -torch.sum(clearance_error, dim=-1)


def wtw_raibert_heuristic(
    env: GenesisLocomotionEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """WTW: penalize footstep placement error vs Raibert heuristic.

    Computes desired foot positions in body frame using commanded velocity,
    stance dimensions, and gait phase, then penalizes deviation.
    """
    from rlworld.rl.utils.quat_utils import quat_apply_yaw_wxyz, quat_conjugate_wxyz

    feet_links = tuple(env.gait_manager.foot_names)
    entity = env.scene_manager[entity_name]
    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False, preserve_order=True)

    foot_positions = entity.get_links_pos(links_idx_local=links_idx_local)
    base_pos = env.get_robot_data(entity_name).root_link_pos_w
    base_quat = env.get_robot_data(entity_name).root_link_quat_w

    num_feet = foot_positions.shape[1]
    cur_footsteps_translated = foot_positions - base_pos.unsqueeze(1)

    footsteps_in_body = torch.zeros_like(cur_footsteps_translated)
    for i in range(num_feet):
        footsteps_in_body[:, i, :] = quat_apply_yaw_wxyz(
            quat_conjugate_wxyz(base_quat), cur_footsteps_translated[:, i, :]
        )

    # Nominal positions from stance commands, order-independent via leg parsing
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import get_leg_xy_signs

    stance_width = env.command_manager.stance_width
    stance_length = env.command_manager.stance_length

    leg_signs = get_leg_xy_signs(feet_links)
    x_signs = torch.tensor([s[0] for s in leg_signs], device=env.device)
    y_signs = torch.tensor([s[1] for s in leg_signs], device=env.device)

    desired_xs = (stance_length.unsqueeze(1) / 2) * x_signs.unsqueeze(0)
    desired_ys = (stance_width.unsqueeze(1) / 2) * y_signs.unsqueeze(0)

    # Raibert offsets based on velocity and gait phase
    foot_phases = env.gait_manager.foot_phases
    phases = torch.abs(1.0 - (foot_phases * 2.0)) * 1.0 - 0.5
    freq = env.command_manager.gait_freq
    x_vel = env.command_manager.lin_vel_x.unsqueeze(1)
    yaw_vel = env.command_manager.ang_vel.unsqueeze(1)
    y_vel_des = yaw_vel * stance_length.unsqueeze(1) / 2

    desired_xs_offset = phases * x_vel * (0.5 / freq.unsqueeze(1))
    # y offset flips for rear legs (x_sign < 0)
    desired_ys_offset = phases * y_vel_des * (0.5 / freq.unsqueeze(1))
    desired_ys_offset = desired_ys_offset * x_signs.unsqueeze(0)

    desired_xs = desired_xs + desired_xs_offset
    desired_ys = desired_ys + desired_ys_offset

    desired_footsteps = torch.stack([desired_xs, desired_ys], dim=2)
    err = torch.abs(desired_footsteps - footsteps_in_body[:, :, 0:2])
    return -torch.sum(torch.square(err), dim=(1, 2))