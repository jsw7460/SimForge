"""mjlab-compatible reward functions for Newton environments.

These functions produce identical outputs to mjlab rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from rlworld.rl.envs.mdp.observations.newton.body_utils import (
    get_bodies_height_with_contact,
    get_bodies_quat,
)
from rlworld.rl.envs.mdp.observations.newton.state import (
    _quat_rotate,
    _quat_rotate_inverse,
    base_quat,
)
from rlworld.rl.envs.utils.newton.body_cache import get_cache
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


# ============================================================
# track_lin_vel_mjlab
# ============================================================

def track_lin_vel_mjlab(
    env: "NewtonEnv",
    std: float,
) -> torch.Tensor:
    """Reward for tracking commanded base linear velocity.

    Matches mjlab.tasks.velocity.mdp.track_linear_velocity exactly.
    Includes z velocity penalty (commanded z is assumed to be zero).

    Args:
        env: Newton environment.
        std: Standard deviation for exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
    ], dim=1)  # (num_envs, 2)

    actual = env.robot_data.root_link_lin_vel_b  # (num_envs, 3)

    xy_error = torch.sum(torch.square(command - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error

    return torch.exp(-lin_vel_error / (std ** 2))


# ============================================================
# track_ang_vel_mjlab
# ============================================================

def track_ang_vel_mjlab(
    env: "NewtonEnv",
    std: float,
) -> torch.Tensor:
    """Reward for tracking commanded angular velocity.

    Matches mjlab.tasks.velocity.mdp.track_angular_velocity exactly.
    Includes xy angular velocity penalty (commanded xy is assumed to be zero).

    Args:
        env: Newton environment.
        std: Standard deviation for exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command_z = env.command_manager.ang_vel  # (num_envs,)

    actual = env.robot_data.root_link_ang_vel_b  # (num_envs, 3)

    z_error = torch.square(command_z - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error

    return torch.exp(-ang_vel_error / (std ** 2))


# ============================================================
# flat_orientation_mjlab
# ============================================================

def flat_orientation_mjlab(
    env: "NewtonEnv",
    std: float,
    body_name: str | None = None,
) -> torch.Tensor:
    if body_name is not None:
        result = get_bodies_quat(env, body_name)
        body_quat_xyzw = result.data[:, 0, :]  # (num_envs, 4)

        # Cache normalized gravity vector
        if not hasattr(env, '_gravity_normalized_cache'):
            env._gravity_normalized_cache = torch.tensor(
                [[0.0, 0.0, -1.0]],
                device=env.device,
                dtype=torch.float32
            ).expand(env.num_envs, -1).contiguous()

        projected_gravity_b = _quat_rotate_inverse(body_quat_xyzw, env._gravity_normalized_cache)
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        projected_gravity_b = env.robot_data.projected_gravity_b  # (num_envs, 3)
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / (std ** 2))


# ============================================================
# body_ang_vel_penalty_mjlab
# ============================================================

def body_ang_vel_penalty_mjlab(
    env: "NewtonEnv",
    body_name: str,
) -> torch.Tensor:
    """Penalize excessive body angular velocities (xy only).

    Delegates to ``common.penalize_body_ang_vel_xy``. The Newton
    implementation of ``RobotData.find_body_index`` calls the same
    ``body_cache.get_body_indices(body_name)`` that the legacy code
    used, and ``RobotData.body_ang_vel_w`` reads the same
    ``state.body_qd[:, body_idx, 3:6]`` slice — so the result is
    bit-identical to the legacy direct-access path.

    Args:
        env: Newton environment.
        body_name: Name of the body to penalize.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
        penalize_body_ang_vel_xy as _common_fn,
    )
    return _common_fn(env, body_name=body_name)


# ============================================================
# feet_air_time_mjlab
# ============================================================

def feet_air_time_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward feet air time.

    Matches mjlab.tasks.velocity.mdp.feet_air_time exactly.

    Args:
        env: Newton environment.
        feet_bodies: Foot body name pattern(s).
        threshold_min: Minimum air time to receive reward.
        threshold_max: Maximum air time to receive reward.
        command_threshold: Minimum command velocity to activate reward.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    result = get_bodies_height_with_contact(env, feet_bodies)
    contact_indices = result.contact_indices

    current_air_time = env.contact_manager.current_air_time("foot_contact")[:, contact_indices]

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    command = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
        env.command_manager.ang_vel,
    ], dim=1)

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    scale = (total_command > command_threshold).float()
    return reward * scale


# ============================================================
# feet_clearance_mjlab
# ============================================================

def feet_clearance_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    target_height: float,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity.

    Matches mjlab.tasks.velocity.mdp.feet_clearance exactly.

    Args:
        env: Newton environment.
        feet_bodies: Foot body name pattern(s).
        target_height: Target foot clearance height.
        command_threshold: Minimum command velocity to activate penalty.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    cache = get_cache(env)
    state = env.scene_manager.state

    result = get_bodies_height_with_contact(env, feet_bodies)
    body_indices = result.body_indices

    foot_z = result.data  # (num_envs, num_feet)

    body_qd = wp.to_torch(state.body_qd).view(env.num_envs, cache.bodies_per_env, 6)
    foot_vel = body_qd[:, body_indices, :3]  # (num_envs, num_feet, 3)
    foot_vel_xy = foot_vel[:, :, :2]  # (num_envs, num_feet, 2)
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # (num_envs, num_feet)

    delta = torch.abs(foot_z - target_height)
    cost = torch.sum(delta * vel_norm, dim=1)

    command = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
        env.command_manager.ang_vel,
    ], dim=1)

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    active = (total_command > command_threshold).float()

    return -cost * active


# ============================================================
# feet_slip_mjlab
# ============================================================

def feet_slip_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact).

    Matches mjlab.tasks.velocity.mdp.feet_slip exactly.

    Args:
        env: Newton environment.
        feet_bodies: Foot body name pattern(s).
        command_threshold: Minimum command velocity to activate penalty.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    cache = get_cache(env)
    state = env.scene_manager.state

    result = get_bodies_height_with_contact(env, feet_bodies)
    body_indices = result.body_indices
    contact_indices = result.contact_indices

    body_qd = wp.to_torch(state.body_qd).view(env.num_envs, cache.bodies_per_env, 6)
    foot_vel_xy = body_qd[:, body_indices, :2]  # (num_envs, num_feet, 2)
    vel_xy_norm_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)  # (num_envs, num_feet)

    is_contact = env.contact_manager.is_contact("foot_contact")[:, contact_indices]  # (num_envs, num_feet)

    command = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
        env.command_manager.ang_vel,
    ], dim=1)

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    active = (total_command > command_threshold).float()

    cost = torch.sum(vel_xy_norm_sq * is_contact.float(), dim=1) * active
    return -cost


# ============================================================
# soft_landing_mjlab
# ============================================================

def soft_landing_mjlab(
    env: "NewtonEnv",
    feet_bodies: str | list[str],
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing.

    Matches mjlab.tasks.velocity.mdp.soft_landing exactly.

    Args:
        env: Newton environment.
        feet_bodies: Foot body name pattern(s).
        command_threshold: Minimum command velocity to activate penalty.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    result = get_bodies_height_with_contact(env, feet_bodies)
    contact_indices = result.contact_indices

    contact_force = env.contact_manager.contact_force("foot_contact")[:, contact_indices]  # (num_envs, num_feet, 3)
    forces = torch.norm(contact_force, dim=-1)  # (num_envs, num_feet)

    first_contact = env.contact_manager.compute_first_contact("foot_contact")[:, contact_indices]

    landing_impact = forces * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)

    command = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
        env.command_manager.ang_vel,
    ], dim=1)

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm

    active = (total_command > command_threshold).float()
    return -cost * active


# ============================================================
# joint_pos_limits_mjlab
# ============================================================

def joint_pos_limits_mjlab(
    env: "NewtonEnv",
    soft_limit_factor: float = 1.0,
) -> torch.Tensor:
    """Penalize joint positions exceeding soft limits.

    Delegates to ``common.penalize_joint_pos_limits_l1`` which reads
    ``RobotData.joint_pos`` and ``RobotData.joint_pos_limits``. The
    Newton implementation of ``joint_pos_limits`` reads the same
    ``model.joint_limit_lower/upper`` arrays indexed by
    ``newton_qd_indices`` that the legacy code accessed, so the result
    is bit-identical to the legacy direct-access path.

    Args:
        env: Newton environment.
        soft_limit_factor: Factor to scale hard limits to get soft limits.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
        penalize_joint_pos_limits_l1 as _common_fn,
    )
    return _common_fn(env, soft_limit_factor=soft_limit_factor)


# ============================================================
# action_rate_l2_mjlab
# ============================================================

def raw_action_rate_l2_mjlab(env: "NewtonEnv") -> torch.Tensor:
    """Penalize the rate of change of raw actions (L2 squared).

    Delegates to ``common.raw_action_rate_l2``. Bit-identical: same
    pure-Python ``act_manager`` arithmetic, no scene-state involved.
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import raw_action_rate_l2
    return raw_action_rate_l2(env)

def processed_action_rate_l2_mjlab(env: "NewtonEnv") -> torch.Tensor:
    """Penalize the rate of change of actions using L2 squared kernel.

    Matches mjlab.envs.mdp.action_rate_l2 exactly.

    Args:
        env: Newton environment.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    return -torch.sum(
        torch.square(env.act_manager.processed_actions - env.act_manager.prev_processed_actions),
        dim=1
    )


# ============================================================
# variable_posture
# ============================================================

class variable_posture:
    """Penalize deviation from default pose with speed-dependent tolerance.

    Uses per-joint standard deviations to control how much each joint can deviate
    from default pose. Smaller std = stricter (less deviation allowed), larger
    std = more forgiving. The reward is: exp(-mean(error² / std²))

    Three speed regimes (based on linear + angular command velocity):
      - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
      - std_walking (walking_threshold <= speed < running_threshold): Moderate.
      - std_running (speed >= running_threshold): Loose tolerance for large motion.

    Matches mjlab.tasks.velocity.mdp.variable_posture exactly.
    """

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
        self.std_standing = torch.tensor(
            std_standing_values, device=env.device, dtype=torch.float32
        )

        _, _, std_walking_values = string_utils.resolve_matching_names_values(
            std_walking, joint_names
        )
        self.std_walking = torch.tensor(
            std_walking_values, device=env.device, dtype=torch.float32
        )

        _, _, std_running_values = string_utils.resolve_matching_names_values(
            std_running, joint_names
        )
        self.std_running = torch.tensor(
            std_running_values, device=env.device, dtype=torch.float32
        )

        self.default_joint_pos = env.act_manager.offset

    def __call__(self, env: "NewtonEnv") -> torch.Tensor:
        lin_vel_x = env.command_manager.lin_vel_x
        lin_vel_y = env.command_manager.lin_vel_y
        ang_vel = env.command_manager.ang_vel

        command = torch.stack([lin_vel_x, lin_vel_y, ang_vel], dim=1)

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < self.walking_threshold).float()
        walking_mask = (
            (total_speed >= self.walking_threshold) & (total_speed < self.running_threshold)
        ).float()
        running_mask = (total_speed >= self.running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = env.robot_data.joint_pos
        error_squared = torch.square(current_joint_pos - self.default_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std ** 2), dim=1))

    def reset(self, env_ids: torch.Tensor) -> None:
        pass


# ============================================================
# feet_swing_height_mjlab
# ============================================================

class feet_swing_height_mjlab:
    """Penalize deviation from target swing height, evaluated at landing.

    Tracks peak foot height during swing phase and evaluates error at first contact.

    Matches mjlab.tasks.velocity.mdp.feet_swing_height exactly.
    """

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

        self.peak_heights = torch.zeros(
            (env.num_envs, self.num_feet), device=env.device, dtype=torch.float32
        )

    def __call__(self, env: "NewtonEnv") -> torch.Tensor:
        result = get_bodies_height_with_contact(env, self.feet_bodies)
        foot_heights = result.data

        contact_found = env.contact_manager.is_contact("foot_contact")[:, self.contact_indices]
        in_air = ~contact_found

        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )

        first_contact = env.contact_manager.compute_first_contact("foot_contact")[:, self.contact_indices]

        lin_vel_x = env.command_manager.lin_vel_x
        lin_vel_y = env.command_manager.lin_vel_y
        ang_vel = env.command_manager.ang_vel

        linear_norm = torch.norm(
            torch.stack([lin_vel_x, lin_vel_y], dim=1), dim=1
        )
        angular_norm = torch.abs(ang_vel)
        total_command = linear_norm + angular_norm

        active = (total_command > self.command_threshold).float()

        # mjlab uses squared error
        error = self.peak_heights / self.target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active

        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return -cost

    def reset(self, env_ids: torch.Tensor) -> None:
        result = get_bodies_height_with_contact(self.env, self.feet_bodies)
        self.peak_heights[env_ids] = result.data[env_ids]


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

    def __call__(self, env: "NewtonEnv") -> torch.Tensor:
        # Original uses abs(error), not squared
        result = get_bodies_height_with_contact(env, self.feet_bodies)
        foot_heights = result.data

        contact_found = env.contact_manager.is_contact("foot_contact")[:, self.contact_indices]
        in_air = ~contact_found

        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )

        first_contact = env.contact_manager.compute_first_contact("foot_contact")[:, self.contact_indices]

        lin_vel_x = env.command_manager.lin_vel_x
        lin_vel_y = env.command_manager.lin_vel_y
        ang_vel = env.command_manager.ang_vel

        linear_norm = torch.norm(
            torch.stack([lin_vel_x, lin_vel_y], dim=1), dim=1
        )
        angular_norm = torch.abs(ang_vel)
        total_command = linear_norm + angular_norm

        active = (total_command > self.command_threshold).float()

        error = self.peak_heights / self.target_height - 1.0
        cost = torch.sum(torch.abs(error) * first_contact.float(), dim=1) * active

        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )

        return -cost


# ============================================================
# angular_momentum_penalty
# ============================================================

def angular_momentum_penalty(
    env: "NewtonEnv",
) -> torch.Tensor:
    """Penalize whole-body angular momentum.

    Delegates to ``common.penalize_angular_momentum_l2``. Bit-identical
    to the legacy implementation: ``RobotData.angular_momentum_w`` for
    Newton uses the same ``model.body_inertia`` warp array, the same
    ``state.body_q/qd`` reshape, and the same xyzw quaternion rotation
    helpers (``_quat_rotate_inverse`` / ``_quat_rotate``) that this
    function previously called inline.
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
        penalize_angular_momentum_l2 as _common_fn,
    )
    return _common_fn(env)