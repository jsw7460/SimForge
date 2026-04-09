import torch
import warp as wp

from rlworld.rl.envs import NewtonEnv, NewtonLocomotionEnv
from rlworld.rl.envs.mdp.observations.newton import state, proprioception
from rlworld.rl.envs.mdp.observations.newton.body_utils import (
    get_bodies_height_with_contact,
    get_bodies_pos,
)
from rlworld.rl.envs.mdp.observations.newton.proprioception import projected_gravity
from rlworld.rl.envs.mdp.observations.newton.state import base_ang_vel
from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
    action_rate_l2,
    base_height_penalty as _common_base_height_penalty,
    get_leg_xy_signs,
    penalize_contact_force_count,
    penalize_lin_vel_z,
    similar_to_default as _common_sim_to_def,
)
from rlworld.rl.envs.utils.newton.body_cache import get_cache
from rlworld.rl.utils import string as string_utils
from rlworld.rl.utils.quat_utils import quat_apply_yaw_wxyz, quat_conjugate_wxyz


def tracking_lin_vel(env: "NewtonEnv", sigma: float = 0.25) -> torch.Tensor:
    """Reward for tracking commanded linear velocity in xy plane.

    Returns exponential of negative squared error between commanded and actual velocity.
    """
    target_lin_vel = torch.stack(
        [env.command_manager.lin_vel_x, env.command_manager.lin_vel_y], dim=1
    )
    lin_vel = state.base_lin_vel(env)
    lin_vel_error = torch.sum(
        torch.square(target_lin_vel - lin_vel[:, :2]), dim=1
    )

    return torch.exp(-lin_vel_error / sigma)


def tracking_ang_vel(env: "NewtonEnv", sigma: float = 0.25) -> torch.Tensor:
    """Reward for tracking commanded angular velocity (yaw).

    Returns exponential of negative squared error between commanded and actual yaw rate.
    """
    ang_vel = state.base_ang_vel(env)
    ang_vel_error = torch.square(env.command_manager.ang_vel - ang_vel[:, 2])
    return torch.exp(-ang_vel_error / sigma)


def lin_vel_z(env: "NewtonEnv") -> torch.Tensor:
    """Penalty for vertical movement.

    Delegates to ``common.penalize_lin_vel_z``. Both implementations apply
    the same ``q⁻¹ ⊗ v ⊗ q`` formula to the world-frame velocity, with the
    same operator order. The Newton-local helper uses xyzw indexing while
    common uses wxyz indexing, but the actual float operations are identical
    → bit-identical for Newton.
    """
    return penalize_lin_vel_z(env)


def base_height_penalty(env: "NewtonEnv") -> torch.Tensor:
    """Penalty for deviating from target base height.

    Delegates to ``common.base_height_penalty``. Bit-identical: both read
    base z from the same root position accessor (no quaternion rotation
    involved) and compute ``-(z - command_manager.base_height)²``.
    """
    return _common_base_height_penalty(env)


def action_rate(env: "NewtonEnv") -> torch.Tensor:
    """Penalty for sudden joint action changes.

    Returns negative squared difference between consecutive joint actions.

    Delegates to the simulator-agnostic ``common.action_rate_l2``. The body
    is bit-identical to the original Newton implementation: both compute
    ``-sum(square(prev - cur))``.
    """
    return action_rate_l2(env)


def similar_to_default(env: "NewtonEnv") -> torch.Tensor:
    """Penalty for deviating from default joint positions.

    Delegates to ``common.similar_to_default``. Bit-identical: both compute
    ``-sum(abs(joint_pos - act_manager.offset))`` from the same actuated
    joint indices.
    """
    return _common_sim_to_def(env)


def reward_feet_air_time(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    threshold: float = 0.1,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward for taking long steps.

    Encourages the robot to lift its feet off the ground for at least
    `threshold` seconds before landing. Only active when moving.

    Args:
        env: Newton environment with contact_manager.
        feet_bodies: Body name pattern(s) for feet (e.g., ".*_foot").
        threshold: Minimum air time (seconds) to receive reward.
        command_threshold: Minimum command velocity magnitude to activate reward.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    result = get_bodies_height_with_contact(env, feet_bodies)

    first_contact = env.contact_manager.compute_first_contact("foot_contact")[:, result.contact_indices]
    last_air_time = env.contact_manager.last_air_time("foot_contact")[:, result.contact_indices]

    reward = torch.sum((last_air_time - threshold) * first_contact, dim=-1)

    command_vel = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
    ], dim=-1)
    is_moving = torch.norm(command_vel, dim=-1) > command_threshold

    return reward * is_moving * (env.termination_manager.episode_length_buf > 5)


def reward_feet_height_exp(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    target_height: float = 0.08,
    sigma: float = 0.01,
) -> torch.Tensor:
    """Reward feet reaching target height during swing (exponential kernel).

    Args:
        env: Newton environment.
        feet_bodies: Body name pattern(s) for feet (e.g., ".*_foot").
        target_height: Target foot height during swing (meters).
        sigma: Exponential kernel width (smaller = stricter).

    Returns:
        Reward tensor of shape (num_envs,).
    """
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data  # (num_envs, num_feet)
    is_contact = env.contact_manager.is_contact("foot_contact")[:, result.contact_indices]
    is_swing = ~is_contact

    height_error = torch.square(feet_height - target_height)
    height_reward = torch.exp(-height_error / sigma)

    return torch.sum(height_reward * is_swing.float(), dim=-1)


def penalize_invalid_contact(
    env: "NewtonEnv",
    allowed_bodies: str | list[str],
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize contacts on non-allowed bodies.

    Args:
        env: Newton environment.
        allowed_bodies: Body name pattern(s) for allowed contacts (e.g., ".*_foot").
        force_threshold: Minimum force (N) to count as invalid contact.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    model = env.scene_manager.model

    contacts = env.scene_manager.contacts
    contact_count = wp.to_torch(contacts.rigid_contact_count).item()
    if contact_count == 0:
        return torch.zeros(env.num_envs, device=env.device)

    shape0 = wp.to_torch(contacts.rigid_contact_shape0)[:contact_count]
    shape1 = wp.to_torch(contacts.rigid_contact_shape1)[:contact_count]
    force = wp.to_torch(contacts.rigid_contact_force)[:contact_count]
    force_magnitude = torch.norm(force, dim=-1)

    shape_body = wp.to_torch(model.shape_body)

    cache = get_cache(env)
    allowed_body_indices = cache.get_body_indices(allowed_bodies)
    allowed_tensor = torch.tensor(allowed_body_indices, device=env.device, dtype=torch.long)

    body0 = shape_body[shape0]
    body1 = shape_body[shape1]

    # Ground plane has body_idx = -1, filter it out
    is_robot_body0 = (body0 != -1)
    is_robot_body1 = (body1 != -1)

    # Normalize to first env's body indices (only valid for robot bodies)
    body0_local = torch.where(is_robot_body0, body0 % cache.bodies_per_env, -1)
    body1_local = torch.where(is_robot_body1, body1 % cache.bodies_per_env, -1)

    # Check if robot body is in allowed list
    is_allowed_body0 = torch.isin(body0_local, allowed_tensor)
    is_allowed_body1 = torch.isin(body1_local, allowed_tensor)

    # Invalid: robot body AND not in allowed list
    invalid_body0 = is_robot_body0 & ~is_allowed_body0
    invalid_body1 = is_robot_body1 & ~is_allowed_body1

    # Contact is invalid if either body is invalid AND force > threshold
    is_invalid = (invalid_body0 | invalid_body1) & (force_magnitude > force_threshold)

    # env_idx from robot body (use body1 if body0 is ground)
    robot_body = torch.where(body0 != -1, body0, body1)
    env_idx = robot_body // cache.bodies_per_env

    # Scatter invalid contacts to each env
    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty.scatter_add_(0, env_idx.long(), is_invalid.float())
    return -penalty


def penalize_impact_force(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
) -> torch.Tensor:
    """Penalize contact force at the moment of landing.

    Args:
        env: Newton environment.
        feet_bodies: Body name pattern(s) for feet (e.g., ".*_foot").

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    result = get_bodies_height_with_contact(env, feet_bodies)

    # Get contact force magnitude for each foot
    contact_force = env.contact_manager.contact_force("foot_contact")[:, result.contact_indices]  # (num_envs, num_feet, 3)
    force_magnitude = torch.norm(contact_force, dim=-1)  # (num_envs, num_feet)

    # First contact mask
    first_contact = env.contact_manager.compute_first_contact("foot_contact")[:, result.contact_indices]
    return -torch.sum(force_magnitude * first_contact.float(), dim=-1)


def penalize_torques(env: "NewtonEnv") -> torch.Tensor:
    """Penalize joint torques."""
    mjw_data = env.scene_manager.solver.mjw_data

    # exclude base DOF
    qfrc_actuator = wp.to_torch(mjw_data.qfrc_actuator)[:, 6:]

    return -torch.sum(torch.square(qfrc_actuator), dim=-1)


def penalize_ang_vel_xy(env: "NewtonEnv") -> torch.Tensor:
    """Penalize roll and pitch angular velocities in body frame."""

    body_ang_vel = base_ang_vel(env)  # (num_envs, 3)
    roll_pitch_vel_squared = torch.sum(torch.square(body_ang_vel[:, :2]), dim=-1)

    return -roll_pitch_vel_squared


def penalize_nonflat_by_gravity(env: "NewtonEnv") -> torch.Tensor:
    """Penalize non-flat orientation using projected gravity."""

    proj_gravity = projected_gravity(env)  # (num_envs, 3)

    return -torch.sum(torch.square(proj_gravity[:, :2]), dim=-1)


def penalize_hip_deviation(
    env: "NewtonEnv",
    hip_joints: str | tuple[str, ...],
) -> torch.Tensor:
    """Penalize hip joint angles deviating from nominal pose."""

    indices, _ = string_utils.resolve_matching_names(
        hip_joints,
        env.act_manager._actuated_joint_names
    )

    dof = proprioception.dof_pos(env)[:, indices]
    nominal = env.act_manager.offset[:, indices]

    return -torch.sum(torch.square(dof - nominal), dim=-1)


def penalize_feet_slip(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    penalty_offset: float = 0.0
) -> torch.Tensor:
    """Penalize foot velocities when feet are in contact with ground.

    Args:
        env: Newton environment.
        feet_bodies: Body name pattern(s) for feet.
        penalty_offset: Will be added to the final penalty.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    cache = get_cache(env)
    state = env.scene_manager.state

    result = get_bodies_height_with_contact(env, feet_bodies)
    body_indices = result.body_indices

    # Get body velocities: (num_envs, bodies_per_env, 6)
    body_qd = wp.to_torch(state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)

    # Extract feet linear velocities (xy only)
    feet_vel_xy = body_qd[:, body_indices, :2]  # (num_envs, num_feet, 2)

    # Squared velocity magnitude (xy only)
    vel_magnitude_sq = torch.sum(torch.square(feet_vel_xy), dim=-1)  # (num_envs, num_feet)

    # Get contact states
    is_contact = env.contact_manager.is_contact("foot_contact")[:, result.contact_indices]  # (num_envs, num_feet)

    # Apply contact mask
    penalty = torch.sum(vel_magnitude_sq * is_contact.float(), dim=-1)  # (num_envs,)

    # Skip first episode step
    penalty = penalty * (env.termination_manager.episode_length_buf > 1).float()
    return -penalty + penalty_offset


# ── Walk-These-Ways reward terms (Newton) ────────────────────────────────

def wtw_feet_slip(env: "NewtonLocomotionEnv") -> torch.Tensor:
    """WTW feet slip: penalize foot xy velocity when in contact OR was in contact."""
    cache = get_cache(env)
    _state = env.scene_manager.state

    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    body_qd = wp.to_torch(_state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)
    feet_vel_xy = body_qd[:, result.body_indices, :2]
    vel_sq = torch.sum(torch.square(feet_vel_xy), dim=-1)

    contact = env.contact_manager.is_contact("foot_contact", order=result.body_names)
    prev_contact = env.contact_manager.prev_is_contact("foot_contact", order=result.body_names)
    contact_filt = contact | prev_contact
    return -torch.sum(contact_filt.float() * vel_sq, dim=-1)


def wtw_tracking_contacts_shaped_force(
    env: "NewtonLocomotionEnv",
    gait_force_sigma: float = 100.0,
) -> torch.Tensor:
    """WTW: penalize foot contact force when foot should be in swing."""
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)
    contact_force = env.contact_manager.contact_force("foot_contact")[:, result.contact_indices]
    foot_forces = torch.norm(contact_force, dim=-1)

    desired_contact = env.gait_manager.desired_contact_states

    reward = -(1.0 - desired_contact) * (1.0 - torch.exp(-foot_forces ** 2 / gait_force_sigma))
    return reward.mean(dim=-1)


def wtw_tracking_contacts_shaped_vel(
    env: "NewtonLocomotionEnv",
    gait_vel_sigma: float = 10.0,
) -> torch.Tensor:
    """WTW: penalize foot velocity when foot should be in stance."""
    cache = get_cache(env)
    _state = env.scene_manager.state

    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    body_qd = wp.to_torch(_state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)
    feet_vel = body_qd[:, result.body_indices, :3]
    foot_vel_norm = torch.norm(feet_vel, dim=-1)

    desired_contact = env.gait_manager.desired_contact_states

    reward = -(desired_contact * (1.0 - torch.exp(-foot_vel_norm ** 2 / gait_vel_sigma)))
    return reward.mean(dim=-1)


def wtw_feet_clearance_cmd_linear(
    env: "NewtonLocomotionEnv",
    foot_radius: float = 0.02,
) -> torch.Tensor:
    """WTW: penalize foot height error during swing, scaled by commanded footswing height."""
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)
    foot_height = result.data

    foot_phases = env.gait_manager.foot_phases
    phases = 1.0 - torch.abs(
        1.0 - torch.clip((foot_phases * 2.0) - 1.0, 0.0, 1.0) * 2.0
    )

    footswing_height = env.command_manager.footswing_height
    target_height = footswing_height.unsqueeze(1) * phases + foot_radius

    desired_contact = env.gait_manager.desired_contact_states
    clearance_error = torch.square(target_height - foot_height) * (1.0 - desired_contact)
    return -torch.sum(clearance_error, dim=-1)


def wtw_raibert_heuristic(env: "NewtonLocomotionEnv") -> torch.Tensor:
    """WTW: penalize footstep placement error vs Raibert heuristic."""
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_pos(env, feet_bodies)
    foot_positions = result.data  # (num_envs, num_feet, 3)

    base_pos = env.get_robot_data().root_link_pos_w
    base_quat = env.get_robot_data().root_link_quat_w

    num_feet = foot_positions.shape[1]
    cur_footsteps_translated = foot_positions - base_pos.unsqueeze(1)

    footsteps_in_body = torch.zeros_like(cur_footsteps_translated)
    for i in range(num_feet):
        footsteps_in_body[:, i, :] = quat_apply_yaw_wxyz(
            quat_conjugate_wxyz(base_quat), cur_footsteps_translated[:, i, :]
        )

    feet_names = env.gait_manager.foot_names
    stance_width = env.command_manager.stance_width
    stance_length = env.command_manager.stance_length

    leg_signs = get_leg_xy_signs(feet_names)
    x_signs = torch.tensor([s[0] for s in leg_signs], device=env.device)
    y_signs = torch.tensor([s[1] for s in leg_signs], device=env.device)

    desired_xs = (stance_length.unsqueeze(1) / 2) * x_signs.unsqueeze(0)
    desired_ys = (stance_width.unsqueeze(1) / 2) * y_signs.unsqueeze(0)

    foot_phases = env.gait_manager.foot_phases
    phases = torch.abs(1.0 - (foot_phases * 2.0)) * 1.0 - 0.5
    freq = env.command_manager.gait_freq
    x_vel = env.command_manager.lin_vel_x.unsqueeze(1)
    yaw_vel = env.command_manager.ang_vel.unsqueeze(1)
    y_vel_des = yaw_vel * stance_length.unsqueeze(1) / 2

    desired_xs_offset = phases * x_vel * (0.5 / freq.unsqueeze(1))
    desired_ys_offset = phases * y_vel_des * (0.5 / freq.unsqueeze(1))
    desired_ys_offset = desired_ys_offset * x_signs.unsqueeze(0)

    desired_xs = desired_xs + desired_xs_offset
    desired_ys = desired_ys + desired_ys_offset

    desired_footsteps = torch.stack([desired_xs, desired_ys], dim=2)
    err = torch.abs(desired_footsteps - footsteps_in_body[:, :, 0:2])
    return -torch.sum(torch.square(err), dim=(1, 2))


def wtw_collision(
    env: "NewtonLocomotionEnv",
    contact_group: str = "body_ground_contact",
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """Thin redirect to ``common.penalize_contact_force_count``.

    Bit-identical: Newton's contact_manager has no substep history, so
    the common helper falls through to the same instantaneous
    ``contact_force`` read this function used previously.
    """
    return penalize_contact_force_count(
        env, contact_group=contact_group, force_threshold=force_threshold
    )


