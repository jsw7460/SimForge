import jax
import jax.numpy as jnp

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
from rlworld.rl.envs.utils.warp_jax_utils import wp_to_jax
from rlworld.rl.utils import string as string_utils


def tracking_lin_vel(env: "NewtonEnv", sigma: float = 0.25) -> jax.Array:
    """Reward for tracking commanded linear velocity in xy plane."""
    target_lin_vel = jnp.stack(
        [env.command_manager.lin_vel_x, env.command_manager.lin_vel_y], axis=1
    )
    lin_vel = state.base_lin_vel(env)
    lin_vel_error = jnp.sum(
        jnp.square(target_lin_vel - lin_vel[:, :2]), axis=1
    )
    return jnp.exp(-lin_vel_error / sigma)


def tracking_ang_vel(env: "NewtonEnv", sigma: float = 0.25) -> jax.Array:
    """Reward for tracking commanded angular velocity (yaw)."""
    ang_vel = state.base_ang_vel(env)
    ang_vel_error = jnp.square(env.command_manager.ang_vel - ang_vel[:, 2])
    return jnp.exp(-ang_vel_error / sigma)


def lin_vel_z(env: "NewtonEnv") -> jax.Array:
    """Penalty for vertical movement."""
    lin_vel = state.base_lin_vel(env)
    return -jnp.square(lin_vel[:, 2])


def base_height_penalty(env: "NewtonEnv") -> jax.Array:
    """Penalty for deviating from target base height."""
    height = state.base_height(env)
    return -jnp.square(height[:, 0] - env.command_manager.base_height)


def action_rate(env: "NewtonEnv") -> jax.Array:
    """Penalty for sudden joint action changes."""
    return -jnp.sum(
        jnp.square(env.act_manager.prev_processed_actions - env.act_manager.processed_actions),
        axis=1
    )


def similar_to_default(env: "NewtonEnv") -> jax.Array:
    """Penalty for deviating from default joint positions."""
    pos = proprioception.dof_pos(env)
    return -jnp.sum(jnp.abs(pos - env.act_manager.offset), axis=1)


def reward_feet_air_time(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    threshold: float = 0.1,
    command_threshold: float = 0.1,
) -> jax.Array:
    """Reward for taking long steps."""
    result = get_bodies_height_with_contact(env, feet_bodies)

    first_contact = env.contact_manager.compute_first_contact()[:, result.contact_indices]
    last_air_time = env.contact_manager.last_air_time[:, result.contact_indices]

    reward = jnp.sum((last_air_time - threshold) * first_contact, axis=-1)

    command_vel = jnp.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
    ], axis=-1)
    is_moving = jnp.linalg.norm(command_vel, axis=-1) > command_threshold

    return reward * is_moving * (env.termination_manager.episode_length_buf > 5)


def penalize_feet_swing_height(
    env: "NewtonLocomotionEnv",
    max_height: float = 0.08,
    profile: str = "sine",
    foot_offset: float = 0.0,
) -> jax.Array:
    """Penalize feet height error during actual swing (not in contact)."""
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data - foot_offset
    target_height = env.gait_manager.get_target_foot_height(max_height, profile)

    is_contact = env.contact_manager.is_contact[:, result.contact_indices]
    is_swing = ~is_contact

    height_error = jnp.square(feet_height - target_height) * is_swing.astype(jnp.float32)

    return -jnp.sum(height_error, axis=-1)


def penalize_feet_swing_height_gait(
    env: "NewtonLocomotionEnv",
    max_height: float = 0.08,
    profile: str = "sine",
    foot_offset: float = 0.0,
    penalty_offset: float = 0.0
) -> jax.Array:
    """Penalize feet height error during commanded swing phase."""
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data - foot_offset
    target_height = env.gait_manager.get_target_foot_height(max_height, profile)
    swing_mask = env.gait_manager.get_swing_mask()

    height_error = jnp.square(feet_height - target_height) * swing_mask.astype(jnp.float32)

    return -jnp.sum(height_error, axis=-1) + penalty_offset


def reward_feet_swing_height_gait_exp(
    env: "NewtonLocomotionEnv",
    max_height: float = 0.08,
    profile: str = "sine",
    foot_offset: float = 0.0,
    sigma: float = 0.01,
) -> jax.Array:
    """Reward feet height tracking during commanded swing phase (Gaussian kernel)."""
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data - foot_offset
    target_height = env.gait_manager.get_target_foot_height(max_height, profile)
    swing_mask = env.gait_manager.get_swing_mask()

    height_error_sq = jnp.square(feet_height - target_height)

    per_foot_reward = jnp.exp(-height_error_sq / sigma) * swing_mask.astype(jnp.float32)
    swing_count = jnp.clip(swing_mask.astype(jnp.float32).sum(axis=-1), a_min=1)

    return per_foot_reward.sum(axis=-1) / swing_count


def reward_feet_height_exp(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    target_height: float = 0.08,
    sigma: float = 0.01,
) -> jax.Array:
    """Reward feet reaching target height during swing (exponential kernel)."""
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data
    is_contact = env.contact_manager.is_contact[:, result.contact_indices]
    is_swing = ~is_contact

    height_error = jnp.square(feet_height - target_height)
    height_reward = jnp.exp(-height_error / sigma)

    return jnp.sum(height_reward * is_swing.astype(jnp.float32), axis=-1)


def penalize_invalid_contact(
    env: "NewtonEnv",
    allowed_bodies: str | list[str],
    force_threshold: float = 1.0,
) -> jax.Array:
    """Penalize contacts on non-allowed bodies."""
    model = env.scene_manager.model

    contacts = env.scene_manager.contacts

    contact_count = int(wp_to_jax(contacts.rigid_contact_count).item())
    if contact_count == 0:
        return jnp.zeros(env.num_envs)

    shape0 = wp_to_jax(contacts.rigid_contact_shape0)[:contact_count]
    shape1 = wp_to_jax(contacts.rigid_contact_shape1)[:contact_count]
    force = wp_to_jax(contacts.rigid_contact_force)[:contact_count]
    force_magnitude = jnp.linalg.norm(force, axis=-1)

    shape_body = wp_to_jax(model.shape_body)

    cache = get_cache(env)
    allowed_body_indices = cache.get_body_indices(allowed_bodies)
    allowed_arr = jnp.array(allowed_body_indices, dtype=jnp.int32)

    body0 = shape_body[shape0]
    body1 = shape_body[shape1]

    is_robot_body0 = (body0 != -1)
    is_robot_body1 = (body1 != -1)

    body0_local = jnp.where(is_robot_body0, body0 % cache.bodies_per_env, -1)
    body1_local = jnp.where(is_robot_body1, body1 % cache.bodies_per_env, -1)

    is_allowed_body0 = jnp.isin(body0_local, allowed_arr)
    is_allowed_body1 = jnp.isin(body1_local, allowed_arr)

    invalid_body0 = is_robot_body0 & ~is_allowed_body0
    invalid_body1 = is_robot_body1 & ~is_allowed_body1

    is_invalid = (invalid_body0 | invalid_body1) & (force_magnitude > force_threshold)

    robot_body = jnp.where(body0 != -1, body0, body1)
    env_idx = robot_body // cache.bodies_per_env

    penalty = jnp.zeros(env.num_envs)
    penalty = penalty.at[env_idx].add(is_invalid.astype(jnp.float32))
    return -penalty


def penalize_impact_force(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
) -> jax.Array:
    """Penalize contact force at the moment of landing."""
    from rlworld.rl.envs.mdp.observations.newton.body_utils import get_bodies_height_with_contact

    result = get_bodies_height_with_contact(env, feet_bodies)

    contact_force = env.contact_manager.contact_force[:, result.contact_indices]
    force_magnitude = jnp.linalg.norm(contact_force, axis=-1)

    first_contact = env.contact_manager.compute_first_contact()[:, result.contact_indices]
    return -jnp.sum(force_magnitude * first_contact.astype(jnp.float32), axis=-1)


def penalize_torques(env: "NewtonEnv") -> jax.Array:
    """Penalize joint torques."""
    mjw_data = env.scene_manager.solver.mjw_data

    qfrc_actuator = wp_to_jax(mjw_data.qfrc_actuator)[:, 6:]

    return -jnp.sum(jnp.square(qfrc_actuator), axis=-1)


def penalize_ang_vel_xy(env: "NewtonEnv") -> jax.Array:
    """Penalize roll and pitch angular velocities in body frame."""
    body_ang_vel = base_ang_vel(env)
    roll_pitch_vel_squared = jnp.sum(jnp.square(body_ang_vel[:, :2]), axis=-1)
    return -roll_pitch_vel_squared


def penalize_nonflat_by_gravity(env: "NewtonEnv") -> jax.Array:
    """Penalize non-flat orientation using projected gravity."""
    proj_gravity = projected_gravity(env)
    return -jnp.sum(jnp.square(proj_gravity[:, :2]), axis=-1)


def reward_flat_by_gravity_exp(env: "NewtonEnv", sigma: float = 0.1) -> jax.Array:
    """Reward flat orientation using projected gravity (Gaussian kernel)."""
    proj_gravity = projected_gravity(env)
    error_sq = jnp.sum(jnp.square(proj_gravity[:, :2]), axis=-1)
    return jnp.exp(-error_sq / sigma)


def penalize_nonflat_by_gravity_exp(env: "NewtonEnv") -> jax.Array:
    """Penalize non-flat orientation using projected gravity."""
    proj_gravity = projected_gravity(env)
    return jnp.exp(-jnp.sum(jnp.square(proj_gravity[:, :2]), axis=-1))


def penalize_hip_deviation(
    env: "NewtonEnv",
    hip_joints: str | tuple[str, ...],
) -> jax.Array:
    """Penalize hip joint angles deviating from nominal pose."""
    indices, _ = string_utils.resolve_matching_names(
        hip_joints,
        env.act_manager._actuated_joint_names
    )

    dof = proprioception.dof_pos(env)[:, indices]
    nominal = env.act_manager.offset[:, indices]

    return -jnp.sum(jnp.square(dof - nominal), axis=-1)


def penalize_joint_deviation_l1(
    env: "NewtonEnv",
    joints: str | tuple[str, ...],
    penalty_offset: float = 0.0
):
    """Penalize joint positions that deviate from the default one (L1 norm)."""
    actuated_joint_names = env.act_manager._actuated_joint_names

    indices_in_actuated, _ = string_utils.resolve_matching_names(
        joints, actuated_joint_names
    )

    dof_pos = proprioception.dof_pos(env)
    default_pos = env.act_manager.offset

    deviation = dof_pos[:, indices_in_actuated] - default_pos[:, indices_in_actuated]
    return -jnp.sum(jnp.abs(deviation), axis=-1) + penalty_offset


def penalize_dof_vel(env: "NewtonEnv") -> jax.Array:
    """Penalize joint velocities."""
    vel = proprioception.dof_vel(env)
    return -jnp.sum(jnp.square(vel), axis=-1)


def penalize_dof_pos_limits(env: "NewtonEnv", soft_joint_pos_limit_factor: float = 1.0) -> jax.Array:
    """Penalize joint positions exceeding limits."""
    model = env.scene_manager.model
    num_worlds = model.num_worlds
    dofs_per_world = model.joint_dof_count // num_worlds

    dof_pos = proprioception.dof_pos(env)

    lower_all = wp_to_jax(model.joint_limit_lower)[:dofs_per_world]
    upper_all = wp_to_jax(model.joint_limit_upper)[:dofs_per_world]

    lower = lower_all[env.act_manager.actuated_qd_indices] * soft_joint_pos_limit_factor
    upper = upper_all[env.act_manager.actuated_qd_indices] * soft_joint_pos_limit_factor

    out_of_limits = -jnp.clip(dof_pos - lower, a_max=0.0)
    out_of_limits = out_of_limits + jnp.clip(dof_pos - upper, a_min=0.0)

    return -jnp.sum(out_of_limits, axis=-1)


def reward_gait_pattern(env: "NewtonLocomotionEnv") -> jax.Array:
    """Reward for matching desired gait pattern."""
    feet_bodies = env.gait_manager.foot_names

    result = get_bodies_height_with_contact(env, feet_bodies)

    is_contact = env.contact_manager.is_contact[:, result.contact_indices]
    swing_mask = env.gait_manager.get_swing_mask()

    correct_swing = ~is_contact & swing_mask
    correct_stance = is_contact & ~swing_mask

    num_correct = jnp.sum(correct_swing.astype(jnp.float32) + correct_stance.astype(jnp.float32), axis=-1)
    num_feet = swing_mask.shape[-1]

    return num_correct / num_feet


def reward_alive(env: "NewtonEnv") -> jax.Array:
    return jnp.ones((env.num_envs,))


def penalize_base_acc(
    env: "NewtonEnv",
    base_body: str = "base",
    penalize_offset: float = 0.0
) -> jax.Array:
    """Penalize base body acceleration."""
    state = env.scene_manager.state

    if state.body_qdd is None:
        raise RuntimeError(
            "body_qdd not available. Add IMU sensor or call "
            "model.request_state_attributes('body_qdd') before finalize."
        )

    cache = get_cache(env)
    body_indices = cache.get_body_indices(base_body)

    body_qdd = wp_to_jax(state.body_qdd).reshape(env.num_envs, cache.bodies_per_env, 6)
    base_acc = body_qdd[:, body_indices[0], :3]

    return -jnp.sum(jnp.square(base_acc), axis=-1) + penalize_offset


def penalize_feet_slip(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    penalty_offset: float = 0.0
) -> jax.Array:
    """Penalize foot velocities when feet are in contact with ground."""
    cache = get_cache(env)
    state = env.scene_manager.state

    result = get_bodies_height_with_contact(env, feet_bodies)
    body_indices = result.body_indices

    body_qd = wp_to_jax(state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)

    feet_vel_xy = body_qd[:, body_indices, :2]

    vel_magnitude_sq = jnp.sum(jnp.square(feet_vel_xy), axis=-1)

    is_contact = env.contact_manager.is_contact[:, result.contact_indices]

    penalty = jnp.sum(vel_magnitude_sq * is_contact.astype(jnp.float32), axis=-1)

    penalty = penalty * (env.termination_manager.episode_length_buf > 1).astype(jnp.float32)
    return -penalty + penalty_offset


def penalize_feet_stance_height(
    env: "NewtonLocomotionEnv",
    threshold: float = 0.02,
) -> jax.Array:
    """Penalize feet height during commanded stance phase."""
    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data
    stance_mask = ~env.gait_manager.get_swing_mask()

    height_violation = jnp.clip(feet_height - threshold, a_min=0.0)
    penalty = jnp.square(height_violation) * stance_mask.astype(jnp.float32)

    return -jnp.sum(penalty, axis=-1)


def penalize_feet_yaw_mean_deviation(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
) -> jax.Array:
    """Penalize deviation between base yaw and mean feet yaw."""
    result = get_bodies_quat(env, feet_bodies)

    feet_quat_xyzw = result.data
    feet_quat_wxyz = feet_quat_xyzw[..., jnp.array([3, 0, 1, 2])]
    feet_yaw = quat_to_xyz(feet_quat_wxyz, rpy=True, degrees=False)[..., 2]

    mean = feet_yaw.mean(-1) + jnp.pi * (jnp.abs(feet_yaw[:, 1] - feet_yaw[:, 0]) > jnp.pi).astype(jnp.float32)

    base_quat_xyzw = state.base_quat(env)
    base_quat_wxyz = base_quat_xyzw[..., jnp.array([3, 0, 1, 2])]
    base_yaw = quat_to_xyz(base_quat_wxyz, rpy=True, degrees=False)[:, 2]

    error = (base_yaw - mean + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return -jnp.square(error)


def penalize_feet_yaw_difference(
    env: "NewtonEnv",
    feet_bodies: tuple[str, str],
) -> jax.Array:
    """Penalize yaw difference between two feet."""
    if len(feet_bodies) != 2:
        raise ValueError("feet_bodies must have exactly 2 elements")

    result = get_bodies_quat(env, list(feet_bodies))
    feet_quat_xyzw = result.data
    feet_quat_wxyz = feet_quat_xyzw[..., jnp.array([3, 0, 1, 2])]

    feet_yaw = quat_to_xyz(feet_quat_wxyz, rpy=True, degrees=False)[..., 2]

    yaw0 = feet_yaw[:, 0]
    yaw1 = feet_yaw[:, 1]

    yaw_diff = (yaw0 - yaw1 + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return -jnp.square(yaw_diff)


def penalize_feet_distance(
    env: "NewtonEnv",
    feet_bodies: tuple[str, str] | list[str],
    feet_distance_ref: float,
) -> jax.Array:
    """Penalize feet lateral distance deviating from target."""
    result = get_bodies_pos(env, list(feet_bodies))
    feet_pos_world = result.data

    base_quat_val = state.base_quat(env)

    feet_pos_body = _quat_rotate_inverse(
        jnp.expand_dims(base_quat_val, 1),
        feet_pos_world
    )

    lateral_distance = jnp.abs(feet_pos_body[:, 1, 1] - feet_pos_body[:, 0, 1])

    return -jnp.clip(feet_distance_ref - lateral_distance, a_min=0.0, a_max=0.1)


def penalize_swing_height_by_velocity(
    env: "NewtonLocomotionEnv",
    max_height: float = 0.1,
    profile: str = "sine",
) -> jax.Array:
    """Penalize feet height error weighted by horizontal velocity during swing."""
    cache = get_cache(env)
    state = env.scene_manager.state

    feet_bodies = env.gait_manager.foot_names
    result = get_bodies_height_with_contact(env, feet_bodies)

    feet_height = result.data
    target_height = env.gait_manager.get_target_foot_height(max_height, profile)

    body_qd = wp_to_jax(state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)
    feet_vel_xy = body_qd[:, result.body_indices, :2]
    vel_norm = jnp.sqrt(jnp.sum(jnp.square(feet_vel_xy), axis=-1))

    height_error = jnp.abs(feet_height - target_height)
    penalty = height_error * vel_norm

    return -jnp.sum(penalty, axis=-1)