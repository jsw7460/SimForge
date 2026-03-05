"""mjlab-compatible reward functions for Newton environments.

These functions produce identical outputs to mjlab rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from rlworld.rl.envs.mdp.observations.newton.body_utils import (
    get_bodies_height_with_contact,
    get_bodies_quat,
)
from rlworld.rl.envs.mdp.observations.newton.state import (
    _quat_rotate,
    _quat_rotate_inverse,
    base_lin_vel,
    base_ang_vel,
    base_quat,
)
from rlworld.rl.envs.mdp.observations.newton import proprioception
from rlworld.rl.envs.utils.newton.body_cache import get_cache
from rlworld.rl.envs.utils.warp_jax_utils import wp_to_jax
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


# ============================================================
# track_lin_vel_mjlab
# ============================================================

def track_lin_vel_mjlab(
    env: "NewtonEnv",
    std: float,
) -> jax.Array:
    """Reward for tracking commanded base linear velocity."""
    command = jnp.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
    ], axis=1)

    actual = base_lin_vel(env)

    xy_error = jnp.sum(jnp.square(command - actual[:, :2]), axis=1)
    z_error = jnp.square(actual[:, 2])
    lin_vel_error = xy_error + z_error

    return jnp.exp(-lin_vel_error / (std ** 2))


# ============================================================
# track_ang_vel_mjlab
# ============================================================

def track_ang_vel_mjlab(
    env: "NewtonEnv",
    std: float,
) -> jax.Array:
    """Reward for tracking commanded angular velocity."""
    command_z = env.command_manager.ang_vel

    actual = base_ang_vel(env)

    z_error = jnp.square(command_z - actual[:, 2])
    xy_error = jnp.sum(jnp.square(actual[:, :2]), axis=1)
    ang_vel_error = z_error + xy_error

    return jnp.exp(-ang_vel_error / (std ** 2))


# ============================================================
# flat_orientation_mjlab
# ============================================================

def flat_orientation_mjlab(
    env: "NewtonEnv",
    std: float,
    body_name: str | None = None,
) -> jax.Array:
    if body_name is not None:
        result = get_bodies_quat(env, body_name)
        body_quat_xyzw = result.data[:, 0, :]

        if not hasattr(env, '_gravity_normalized_cache_jax'):
            env._gravity_normalized_cache_jax = jnp.broadcast_to(
                jnp.array([[0.0, 0.0, -1.0]]),
                (env.num_envs, 3),
            )

        projected_gravity_b = _quat_rotate_inverse(body_quat_xyzw, env._gravity_normalized_cache_jax)
        xy_squared = jnp.sum(jnp.square(projected_gravity_b[:, :2]), axis=1)
    else:
        projected_gravity_b = proprioception.projected_gravity(env)
        xy_squared = jnp.sum(jnp.square(projected_gravity_b[:, :2]), axis=1)

    return jnp.exp(-xy_squared / (std ** 2))


# ============================================================
# body_ang_vel_penalty_mjlab
# ============================================================

def body_ang_vel_penalty_mjlab(
    env: "NewtonEnv",
    body_name: str,
) -> jax.Array:
    """Penalize excessive body angular velocities (xy only)."""
    cache = get_cache(env)
    state = env.scene_manager.state

    body_indices = cache.get_body_indices(body_name)
    body_idx = body_indices[0]

    body_qd = wp_to_jax(state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)
    ang_vel_w = body_qd[:, body_idx, 3:6]

    ang_vel_xy = ang_vel_w[:, :2]

    return -jnp.sum(jnp.square(ang_vel_xy), axis=1)


# ============================================================
# feet_air_time_mjlab
# ============================================================

def feet_air_time_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_threshold: float = 0.5,
) -> jax.Array:
    """Reward feet air time."""
    result = get_bodies_height_with_contact(env, feet_bodies)
    contact_indices = result.contact_indices

    current_air_time = env.contact_manager.current_air_time[:, contact_indices]

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = jnp.sum(in_range.astype(jnp.float32), axis=1)

    command = jnp.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
        env.command_manager.ang_vel,
    ], axis=1)

    linear_norm = jnp.linalg.norm(command[:, :2], axis=1)
    angular_norm = jnp.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    scale = (total_command > command_threshold).astype(jnp.float32)
    return reward * scale


# ============================================================
# feet_clearance_mjlab
# ============================================================

def feet_clearance_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    target_height: float,
    command_threshold: float = 0.01,
) -> jax.Array:
    """Penalize deviation from target clearance height, weighted by foot velocity."""
    cache = get_cache(env)
    state = env.scene_manager.state

    result = get_bodies_height_with_contact(env, feet_bodies)
    body_indices = result.body_indices

    foot_z = result.data

    body_qd = wp_to_jax(state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)
    foot_vel = body_qd[:, body_indices, :3]
    foot_vel_xy = foot_vel[:, :, :2]
    vel_norm = jnp.linalg.norm(foot_vel_xy, axis=-1)

    delta = jnp.abs(foot_z - target_height)
    cost = jnp.sum(delta * vel_norm, axis=1)

    command = jnp.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
        env.command_manager.ang_vel,
    ], axis=1)

    linear_norm = jnp.linalg.norm(command[:, :2], axis=1)
    angular_norm = jnp.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    active = (total_command > command_threshold).astype(jnp.float32)

    return -cost * active


# ============================================================
# feet_slip_mjlab
# ============================================================

def feet_slip_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    command_threshold: float = 0.05,
) -> jax.Array:
    """Penalize foot sliding (xy velocity while in contact)."""
    cache = get_cache(env)
    state = env.scene_manager.state

    result = get_bodies_height_with_contact(env, feet_bodies)
    body_indices = result.body_indices
    contact_indices = result.contact_indices

    body_qd = wp_to_jax(state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)
    foot_vel_xy = body_qd[:, body_indices, :2]
    vel_xy_norm_sq = jnp.sum(jnp.square(foot_vel_xy), axis=-1)

    is_contact = env.contact_manager.is_contact[:, contact_indices]

    command = jnp.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
        env.command_manager.ang_vel,
    ], axis=1)

    linear_norm = jnp.linalg.norm(command[:, :2], axis=1)
    angular_norm = jnp.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    active = (total_command > command_threshold).astype(jnp.float32)

    cost = jnp.sum(vel_xy_norm_sq * is_contact.astype(jnp.float32), axis=1) * active
    return -cost


# ============================================================
# soft_landing_mjlab
# ============================================================

def soft_landing_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    command_threshold: float = 0.05,
) -> jax.Array:
    """Penalize high impact forces at landing."""
    result = get_bodies_height_with_contact(env, feet_bodies)
    contact_indices = result.contact_indices

    contact_force = env.contact_manager.contact_force[:, contact_indices]
    forces = jnp.linalg.norm(contact_force, axis=-1)

    first_contact = env.contact_manager.compute_first_contact()[:, contact_indices]

    landing_impact = forces * first_contact.astype(jnp.float32)
    cost = jnp.sum(landing_impact, axis=1)

    command = jnp.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
        env.command_manager.ang_vel,
    ], axis=1)

    linear_norm = jnp.linalg.norm(command[:, :2], axis=1)
    angular_norm = jnp.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    active = (total_command > command_threshold).astype(jnp.float32)
    return -cost * active


# ============================================================
# joint_pos_limits_mjlab
# ============================================================

def joint_pos_limits_mjlab(
    env: "NewtonEnv",
    soft_limit_factor: float = 1.0,
) -> jax.Array:
    """Penalize joint positions exceeding soft limits."""
    model = env.scene_manager.model
    dofs_per_world = model.joint_dof_count // model.world_count

    dof_pos = proprioception.dof_pos(env)

    lower_all = wp_to_jax(model.joint_limit_lower)[:dofs_per_world]
    upper_all = wp_to_jax(model.joint_limit_upper)[:dofs_per_world]

    lower = lower_all[env.act_manager.actuated_qd_indices] * soft_limit_factor
    upper = upper_all[env.act_manager.actuated_qd_indices] * soft_limit_factor

    out_of_limits = -jnp.clip(dof_pos - lower, a_max=0.0)
    out_of_limits = out_of_limits + jnp.clip(dof_pos - upper, a_min=0.0)

    return -jnp.sum(out_of_limits, axis=-1)


# ============================================================
# action_rate_l2_mjlab
# ============================================================

def raw_action_rate_l2_mjlab(env: "NewtonEnv") -> jax.Array:
    """Penalize the rate of change of actions using L2 squared kernel."""
    return -jnp.sum(
        jnp.square(env.act_manager.raw_actions - env.act_manager.prev_raw_actions),
        axis=1
    )

def processed_action_rate_l2_mjlab(env: "NewtonEnv") -> jax.Array:
    """Penalize the rate of change of actions using L2 squared kernel."""
    return -jnp.sum(
        jnp.square(env.act_manager.processed_actions - env.act_manager.prev_processed_actions),
        axis=1
    )


# ============================================================
# variable_posture
# ============================================================

class variable_posture:
    """Penalize deviation from default pose with speed-dependent tolerance."""

    __name__ = "variable_posture"

    def __init__(
        self,
        env: "NewtonEnv",
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ):
        self.env = env
        self.walking_threshold = walking_threshold
        self.running_threshold = running_threshold

        joint_names = env.act_manager.actuated_joint_names

        _, _, std_standing_values = string_utils.resolve_matching_names_values(
            std_standing, joint_names
        )
        self.std_standing = jnp.array(std_standing_values, dtype=jnp.float32)

        _, _, std_walking_values = string_utils.resolve_matching_names_values(
            std_walking, joint_names
        )
        self.std_walking = jnp.array(std_walking_values, dtype=jnp.float32)

        _, _, std_running_values = string_utils.resolve_matching_names_values(
            std_running, joint_names
        )
        self.std_running = jnp.array(std_running_values, dtype=jnp.float32)

        self.default_joint_pos = env.act_manager.offset

    def __call__(self, env: "NewtonEnv") -> jax.Array:
        lin_vel_x = env.command_manager.lin_vel_x
        lin_vel_y = env.command_manager.lin_vel_y
        ang_vel = env.command_manager.ang_vel

        command = jnp.stack([lin_vel_x, lin_vel_y, ang_vel], axis=1)

        linear_speed = jnp.linalg.norm(command[:, :2], axis=1)
        angular_speed = jnp.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < self.walking_threshold).astype(jnp.float32)
        walking_mask = (
            (total_speed >= self.walking_threshold) & (total_speed < self.running_threshold)
        ).astype(jnp.float32)
        running_mask = (total_speed >= self.running_threshold).astype(jnp.float32)

        std = (
            self.std_standing * jnp.expand_dims(standing_mask, 1)
            + self.std_walking * jnp.expand_dims(walking_mask, 1)
            + self.std_running * jnp.expand_dims(running_mask, 1)
        )

        current_joint_pos = proprioception.dof_pos(env)
        error_squared = jnp.square(current_joint_pos - self.default_joint_pos)

        return jnp.exp(-jnp.mean(error_squared / (std ** 2), axis=1))

    def reset(self, env_ids) -> None:
        pass


# ============================================================
# feet_swing_height_mjlab
# ============================================================

class feet_swing_height_mjlab:
    """Penalize deviation from target swing height, evaluated at landing."""

    __name__ = "feet_swing_height_mjlab"

    def __init__(
        self,
        env: "NewtonEnv",
        feet_bodies: str | list[str],
        target_height: float,
        command_threshold: float = 0.05,
    ):
        self.env = env
        self.feet_bodies = feet_bodies
        self.target_height = target_height
        self.command_threshold = command_threshold

        result = get_bodies_height_with_contact(env, feet_bodies)
        self.num_feet = len(result.body_indices)
        self.contact_indices = result.contact_indices

        self.peak_heights = jnp.zeros(
            (env.num_envs, self.num_feet), dtype=jnp.float32
        )

    def __call__(self, env: "NewtonEnv") -> jax.Array:
        result = get_bodies_height_with_contact(env, self.feet_bodies)
        foot_heights = result.data

        contact_found = env.contact_manager.is_contact[:, self.contact_indices]
        in_air = ~contact_found

        self.peak_heights = jnp.where(
            in_air,
            jnp.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )

        first_contact = env.contact_manager.compute_first_contact()[:, self.contact_indices]

        lin_vel_x = env.command_manager.lin_vel_x
        lin_vel_y = env.command_manager.lin_vel_y
        ang_vel = env.command_manager.ang_vel

        linear_norm = jnp.linalg.norm(
            jnp.stack([lin_vel_x, lin_vel_y], axis=1), axis=1
        )
        angular_norm = jnp.abs(ang_vel)
        total_command = linear_norm + angular_norm

        active = (total_command > self.command_threshold).astype(jnp.float32)

        error = self.peak_heights / self.target_height - 1.0
        cost = jnp.sum(jnp.square(error) * first_contact.astype(jnp.float32), axis=1) * active

        self.peak_heights = jnp.where(
            first_contact,
            jnp.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return -cost

    def reset(self, env_ids) -> None:
        result = get_bodies_height_with_contact(self.env, self.feet_bodies)
        self.peak_heights = self.peak_heights.at[env_ids].set(result.data[env_ids])


# ============================================================
# feet_swing_height (alias for backward compatibility)
# ============================================================

class feet_swing_height(feet_swing_height_mjlab):
    """Alias for feet_swing_height_mjlab for backward compatibility."""

    __name__ = "feet_swing_height"

    def __init__(
        self,
        env: "NewtonEnv",
        feet_bodies: str | list[str],
        target_height: float,
        command_threshold: float = 0.05,
    ):
        super().__init__(env, feet_bodies, target_height, command_threshold)

    def __call__(self, env: "NewtonEnv") -> jax.Array:
        result = get_bodies_height_with_contact(env, self.feet_bodies)
        foot_heights = result.data

        contact_found = env.contact_manager.is_contact[:, self.contact_indices]
        in_air = ~contact_found

        self.peak_heights = jnp.where(
            in_air,
            jnp.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )

        first_contact = env.contact_manager.compute_first_contact()[:, self.contact_indices]

        lin_vel_x = env.command_manager.lin_vel_x
        lin_vel_y = env.command_manager.lin_vel_y
        ang_vel = env.command_manager.ang_vel

        linear_norm = jnp.linalg.norm(
            jnp.stack([lin_vel_x, lin_vel_y], axis=1), axis=1
        )
        angular_norm = jnp.abs(ang_vel)
        total_command = linear_norm + angular_norm

        active = (total_command > self.command_threshold).astype(jnp.float32)

        error = self.peak_heights / self.target_height - 1.0
        cost = jnp.sum(jnp.abs(error) * first_contact.astype(jnp.float32), axis=1) * active

        self.peak_heights = jnp.where(
            first_contact,
            jnp.zeros_like(self.peak_heights),
            self.peak_heights,
        )

        return -cost


# ============================================================
# angular_momentum_penalty
# ============================================================

def angular_momentum_penalty(
    env: "NewtonEnv",
) -> jax.Array:
    """Penalize whole-body angular momentum."""
    model = env.scene_manager.model
    state = env.scene_manager.state
    cache = get_cache(env)

    body_inertia = wp_to_jax(model.body_inertia).reshape(
        env.num_envs, cache.bodies_per_env, 3, 3
    )
    body_qd = wp_to_jax(state.body_qd).reshape(
        env.num_envs, cache.bodies_per_env, 6
    )
    body_q = wp_to_jax(state.body_q).reshape(
        env.num_envs, cache.bodies_per_env, 7
    )

    ang_vel_world = body_qd[:, :, 3:6]
    body_quat = body_q[:, :, 3:7]

    ang_vel_body = _quat_rotate_inverse(body_quat, ang_vel_world)
    ang_momentum_body = jnp.einsum('nbij,nbj->nbi', body_inertia, ang_vel_body)
    ang_momentum_world = _quat_rotate(body_quat, ang_momentum_body)

    total_ang_momentum = jnp.sum(ang_momentum_world, axis=1)
    ang_momentum_magnitude_sq = jnp.sum(jnp.square(total_ang_momentum), axis=-1)

    return -ang_momentum_magnitude_sq
