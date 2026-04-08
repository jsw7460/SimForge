import torch
import warp as wp

from genesis import quat_to_xyz
from rlworld.rl.envs import NewtonEnv, NewtonLocomotionEnv
from rlworld.rl.envs.mdp.observations.newton import state, proprioception
from rlworld.rl.envs.mdp.observations.newton.body_utils import (
    get_bodies_height_with_contact,
)
from rlworld.rl.envs.mdp.observations.newton.body_utils import get_bodies_pos, get_bodies_quat
from rlworld.rl.envs.mdp.observations.newton.proprioception import projected_gravity
from rlworld.rl.envs.mdp.observations.newton.state import base_ang_vel, _quat_rotate_inverse
from rlworld.rl.envs.utils.newton.body_cache import get_cache
from rlworld.rl.utils import string as string_utils


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
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import penalize_lin_vel_z
    return penalize_lin_vel_z(env)


def base_height_penalty(env: "NewtonEnv") -> torch.Tensor:
    """Penalty for deviating from target base height.

    Delegates to ``common.base_height_penalty``. Bit-identical: both read
    base z from the same root position accessor (no quaternion rotation
    involved) and compute ``-(z - command_manager.base_height)²``.
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import base_height_penalty as _common_base_height_penalty
    return _common_base_height_penalty(env)


def action_rate(env: "NewtonEnv") -> torch.Tensor:
    """Penalty for sudden joint action changes.

    Returns negative squared difference between consecutive joint actions.

    Delegates to the simulator-agnostic ``common.action_rate_l2``. The body
    is bit-identical to the original Newton implementation: both compute
    ``-sum(square(prev - cur))``.
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import action_rate_l2
    return action_rate_l2(env)


def similar_to_default(env: "NewtonEnv") -> torch.Tensor:
    """Penalty for deviating from default joint positions.

    Delegates to ``common.similar_to_default``. Bit-identical: both compute
    ``-sum(abs(joint_pos - act_manager.offset))`` from the same actuated
    joint indices.
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import similar_to_default as _common_sim_to_def
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


def penalize_feet_swing_height(
    env: "NewtonLocomotionEnv",
    max_height: float = 0.08,
    profile: str = "sine",
    foot_offset: float = 0.0,
) -> torch.Tensor:
    """Penalize feet height error during actual swing (not in contact).

    Args:
        env: Newton locomotion environment with gait_manager.
        max_height: Peak foot height during swing (meters).
        profile: Height profile ("sine" or "cosine").
        foot_offset: Distance from link origin to foot bottom (meters).

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data - foot_offset
    target_height = env.gait_manager.get_target_foot_height(max_height, profile)

    is_contact = env.contact_manager.is_contact("foot_contact")[:, result.contact_indices]
    is_swing = ~is_contact

    height_error = torch.square(feet_height - target_height) * is_swing.float()

    return -torch.sum(height_error, dim=-1)


def penalize_feet_swing_height_gait(
    env: "NewtonLocomotionEnv",
    max_height: float = 0.08,
    profile: str = "sine",
    foot_offset: float = 0.0,
    penalty_offset: float = 0.0
) -> torch.Tensor:
    """Penalize feet height error during commanded swing phase.

    Args:
        env: Newton locomotion environment with gait_manager.
        max_height: Peak foot height during swing (meters).
        profile: Height profile ("sine" or "cosine").
        foot_offset: Distance from link origin to foot bottom (meters).
        penalty_offset: Will be added to the final penalty.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data - foot_offset
    target_height = env.gait_manager.get_target_foot_height(max_height, profile)
    swing_mask = env.gait_manager.get_swing_mask()

    height_error = torch.square(feet_height - target_height) * swing_mask.float()

    return -torch.sum(height_error, dim=-1) + penalty_offset


def reward_feet_swing_height_gait_exp(
    env: "NewtonLocomotionEnv",
    max_height: float = 0.08,
    profile: str = "sine",
    foot_offset: float = 0.0,
    sigma: float = 0.01,
) -> torch.Tensor:
    """Reward feet height tracking during commanded swing phase (Gaussian kernel).

    Args:
        env: Newton locomotion environment with gait_manager.
        max_height: Peak foot height during swing (meters).
        profile: Height profile ("sine" or "cosine").
        foot_offset: Distance from link origin to foot bottom (meters).
        sigma: Gaussian kernel width (smaller = stricter).

    Returns:
        Reward tensor of shape (num_envs,).
    """
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data - foot_offset
    target_height = env.gait_manager.get_target_foot_height(max_height, profile)
    swing_mask = env.gait_manager.get_swing_mask()

    height_error_sq = torch.square(feet_height - target_height)

    # Per-foot Gaussian reward, averaged over swing feet
    per_foot_reward = torch.exp(-height_error_sq / sigma) * swing_mask.float()
    swing_count = swing_mask.float().sum(dim=-1).clamp(min=1)

    return per_foot_reward.sum(dim=-1) / swing_count


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
    from rlworld.rl.envs.mdp.observations.newton.body_utils import get_bodies_height_with_contact

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


def reward_flat_by_gravity_exp(env: "NewtonEnv", sigma: float = 0.1) -> torch.Tensor:
    """Reward flat orientation using projected gravity (Gaussian kernel)."""
    proj_gravity = projected_gravity(env)  # (num_envs, 3)

    error_sq = torch.sum(torch.square(proj_gravity[:, :2]), dim=-1)
    return torch.exp(-error_sq / sigma)


def penalize_nonflat_by_gravity_exp(env: "NewtonEnv") -> torch.Tensor:
    """Penalize non-flat orientation using projected gravity."""

    proj_gravity = projected_gravity(env)  # (num_envs, 3)

    return torch.exp(-torch.sum(torch.square(proj_gravity[:, :2]), dim=-1))


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

def penalize_joint_deviation_l1(
    env: "NewtonEnv",
    joints: str | tuple[str, ...],
    penalty_offset: float = 0.0
):
    """Penalize joint positions that deviate from the default one (L1 norm).

        Args:
            env: Genesis environment.
            joints: Joint name(s) or regex pattern(s).
            penalty_offset: Penalty offset.

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
    return -torch.sum(torch.abs(deviation), dim=-1) + penalty_offset


def penalize_dof_vel(env: "NewtonEnv") -> torch.Tensor:
    """Penalize joint velocities.

    Delegates to ``common.penalize_dof_vel``. Bit-identical: same data path
    (actuated joint velocities), same formula ``-sum(square(joint_vel))``.
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import penalize_dof_vel as _common_pen_dof_vel
    return _common_pen_dof_vel(env)


def penalize_dof_pos_limits(env: "NewtonEnv", soft_joint_pos_limit_factor: float = 1.0) -> torch.Tensor:
    """Penalize joint positions exceeding limits."""
    model = env.scene_manager.model
    num_worlds = model.num_worlds
    dofs_per_world = model.joint_dof_count // num_worlds

    dof_pos = proprioception.dof_pos(env)  # (num_envs, num_actuated)

    # Limits are indexed by qd (velocity DOF)
    lower_all = wp.to_torch(model.joint_limit_lower)[:dofs_per_world]
    upper_all = wp.to_torch(model.joint_limit_upper)[:dofs_per_world]

    lower = lower_all[env.act_manager.actuated_qd_indices] * soft_joint_pos_limit_factor
    upper = upper_all[env.act_manager.actuated_qd_indices] * soft_joint_pos_limit_factor

    # Violation amounts
    out_of_limits = -(dof_pos - lower).clamp(max=0.0)  # below lower
    out_of_limits += (dof_pos - upper).clamp(min=0.0)  # above upper

    return -torch.sum(out_of_limits, dim=-1)


def reward_gait_pattern(env: "NewtonLocomotionEnv") -> torch.Tensor:
    """Reward for matching desired gait pattern.

    Encourages feet to follow the gait phase:
    - Swing phase: foot should NOT be in contact
    - Stance phase: foot SHOULD be in contact

    Args:
        env: Newton locomotion environment with gait_manager.

    Returns:
        Reward tensor [num_envs], range [0, 1].
    """
    feet_bodies = env.gait_manager.foot_names

    result = get_bodies_height_with_contact(env, feet_bodies)

    is_contact = env.contact_manager.is_contact("foot_contact")[:, result.contact_indices]
    swing_mask = env.gait_manager.get_swing_mask()

    correct_swing = ~is_contact & swing_mask
    correct_stance = is_contact & ~swing_mask

    num_correct = torch.sum(correct_swing.float() + correct_stance.float(), dim=-1)
    num_feet = swing_mask.shape[-1]

    return num_correct / num_feet


def reward_alive(env: "NewtonEnv") -> torch.Tensor:
    """Constant alive reward. Delegates to ``common.reward_alive`` (bit-identical)."""
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import reward_alive as _common_reward_alive
    return _common_reward_alive(env)


def penalize_base_acc(
    env: "NewtonEnv",
    base_body: str = "base",
    penalize_offset: float = 0.0
) -> torch.Tensor:
    """Penalize base body acceleration.

    NOTE: Requires body_qdd to be requested via model.request_state_attributes("body_qdd").
          This is automatically done when IMU sensor is configured.

    Args:
        env: Newton environment.
        base_body: Name of the base body.
        penalize_offset: Will be added to the final penalty.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    state = env.scene_manager.state

    if state.body_qdd is None:
        raise RuntimeError(
            "body_qdd not available. Add IMU sensor or call "
            "model.request_state_attributes('body_qdd') before finalize."
        )

    cache = get_cache(env)
    body_indices = cache.get_body_indices(base_body)

    body_qdd = wp.to_torch(state.body_qdd).reshape(env.num_envs, cache.bodies_per_env, 6)
    base_acc = body_qdd[:, body_indices[0], :3]  # (num_envs, 3) [Linear acc]

    return -torch.sum(torch.square(base_acc), dim=-1) + penalize_offset


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


def penalize_feet_stance_height(
    env: "NewtonLocomotionEnv",
    threshold: float = 0.02,
) -> torch.Tensor:
    """Penalize feet height during commanded stance phase."""
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data
    stance_mask = ~env.gait_manager.get_swing_mask()

    height_violation = (feet_height - threshold).clamp(min=0.0)
    penalty = torch.square(height_violation) * stance_mask.float()

    return -torch.sum(penalty, dim=-1)


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
    from rlworld.rl.utils.quat_utils import quat_apply_yaw_wxyz, quat_conjugate_wxyz

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

    from rlworld.rl.envs.mdp.rewards.common.reward_terms import get_leg_xy_signs

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
    """WTW collision: count non-foot bodies with contact force > threshold.

    Uses contact_manager instead of raw contacts.rigid_contact_force
    (which is always zero in Newton without SensorContact).
    """
    force = env.contact_manager.contact_force(contact_group)  # (num_envs, N, 3)
    force_mag = torch.norm(force, dim=-1)  # (num_envs, N)
    return -torch.sum((force_mag > force_threshold).float(), dim=-1)


def penalize_feet_yaw_mean_deviation(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
) -> torch.Tensor:
    """Penalize deviation between base yaw and mean feet yaw.

    Args:
        env: Newton environment.
        feet_bodies: Body name pattern(s) for feet.

    Returns:
        Penalty tensor of shape (num_envs,).
    """

    result = get_bodies_quat(env, feet_bodies)

    feet_quat_xyzw = result.data  # (num_envs, num_feet, 4)
    feet_quat_wxyz = feet_quat_xyzw[..., [3, 0, 1, 2]]
    feet_yaw = quat_to_xyz(feet_quat_wxyz, rpy=True, degrees=False)[..., 2]  # (num_envs, num_feet)

    mean = feet_yaw.mean(-1) + torch.pi * (torch.abs(feet_yaw[:, 1] - feet_yaw[:, 0]) > torch.pi).float()

    base_quat_xyzw = state.base_quat(env)
    base_quat_wxyz = base_quat_xyzw[..., [3, 0, 1, 2]]
    base_yaw = quat_to_xyz(base_quat_wxyz, rpy=True, degrees=False)[:, 2]

    error = (base_yaw - mean + torch.pi) % (2 * torch.pi) - torch.pi
    return -torch.square(error)


def penalize_feet_yaw_difference(
    env: "NewtonEnv",
    feet_bodies: tuple[str, str],
) -> torch.Tensor:
    """Penalize yaw difference between two feet.

    For bipedal robots - encourages feet to point in the same direction.

    Args:
        env: Newton environment.
        feet_bodies: Exactly 2 foot body names.

    Returns:
        Penalty tensor of shape (num_envs,).
    """

    if len(feet_bodies) != 2:
        raise ValueError("feet_bodies must have exactly 2 elements")

    result = get_bodies_quat(env, list(feet_bodies))
    feet_quat_xyzw = result.data  # (num_envs, 2, 4)
    feet_quat_wxyz = feet_quat_xyzw[..., [3, 0, 1, 2]]

    feet_yaw = quat_to_xyz(feet_quat_wxyz, rpy=True, degrees=False)[..., 2]  # (num_envs, 2)

    yaw0 = feet_yaw[:, 0]
    yaw1 = feet_yaw[:, 1]

    # Wrapped difference
    yaw_diff = (yaw0 - yaw1 + torch.pi) % (2 * torch.pi) - torch.pi

    return -torch.square(yaw_diff)


def penalize_feet_distance(
    env: "NewtonEnv",
    feet_bodies: tuple[str, str] | list[str],
    feet_distance_ref: float,
) -> torch.Tensor:
    """Penalize feet lateral distance deviating from target.

    Measures left-right distance between feet relative to robot's heading,
    ignoring forward-backward separation.

    Args:
        env: Newton environment.
        feet_bodies: Exactly 2 foot body names.
        feet_distance_ref: Target lateral distance (meters).

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    result = get_bodies_pos(env, list(feet_bodies))
    feet_pos_world = result.data  # (num_envs, 2, 3)

    base_quat = state.base_quat(env)  # (num_envs, 4)

    # Transform to body frame
    feet_pos_body = _quat_rotate_inverse(
        base_quat.unsqueeze(1),  # (num_envs, 1, 4)
        feet_pos_world  # (num_envs, 2, 3)
    )  # (num_envs, 2, 3)

    # Lateral distance = y-component difference in body frame
    lateral_distance = torch.abs(feet_pos_body[:, 1, 1] - feet_pos_body[:, 0, 1])

    return -torch.clamp(feet_distance_ref - lateral_distance, min=0.0, max=0.1)


def penalize_swing_height_by_velocity(
    env: "NewtonLocomotionEnv",
    max_height: float = 0.1,
    profile: str = "sine",
) -> torch.Tensor:
    """Penalize feet height error weighted by horizontal velocity during swing.

    Feet moving fast horizontally should maintain target height.
    Feet moving slow (stance-like) receive less penalty.

    Args:
        env: Newton locomotion environment with gait_manager.
        max_height: Peak foot height during swing (meters).
        profile: Height profile ("sine" or "cosine").

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    cache = get_cache(env)
    state = env.scene_manager.state

    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    # Feet height and target
    feet_height = result.data  # (num_envs, num_feet)
    target_height = env.gait_manager.get_target_foot_height(max_height, profile)

    # Feet xy velocity magnitude
    body_qd = wp.to_torch(state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)
    feet_vel_xy = body_qd[:, result.body_indices, :2]  # (num_envs, num_feet, 2)
    vel_norm = torch.sqrt(torch.sum(torch.square(feet_vel_xy), dim=-1))  # (num_envs, num_feet)

    # Height error weighted by velocity
    height_error = torch.abs(feet_height - target_height)
    penalty = height_error * vel_norm

    return -torch.sum(penalty, dim=-1)