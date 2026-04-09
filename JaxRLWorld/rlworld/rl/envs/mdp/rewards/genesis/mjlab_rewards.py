"""mjlab-compatible reward functions for Genesis environments.

These functions produce identical outputs to mjlab rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesis.utils.geom import transform_by_quat, inv_quat
from rlworld.rl.envs.mdp.observations.genesis import proprioception
from rlworld.rl.utils import entity_utils as eu
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


# ============================================================
# track_lin_vel_mjlab
# ============================================================

def track_lin_vel_mjlab(
    env: "GenesisEnv",
    std: float,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward for tracking commanded base linear velocity.

    Matches mjlab.tasks.velocity.mdp.track_linear_velocity exactly.
    Includes z velocity penalty (commanded z is assumed to be zero).

    Args:
        env: Genesis environment.
        std: Standard deviation for exponential kernel.
        entity_name: Name of the robot entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get commanded velocity (xy only)
    command = torch.stack([
        env.command_manager.lin_vel_x,
        env.command_manager.lin_vel_y,
    ], dim=1)  # (num_envs, 2)

    # Get actual velocity in body frame
    actual = env.robot_data.root_link_lin_vel_b  # (num_envs, 3)

    # xy error + z error (z command assumed zero)
    xy_error = torch.sum(torch.square(command - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error

    return torch.exp(-lin_vel_error / (std ** 2))


# ============================================================
# track_ang_vel_mjlab
# ============================================================

def track_ang_vel_mjlab(
    env: "GenesisEnv",
    std: float,
    entity_name: str = "robot",
    base_name: str = "base",
) -> torch.Tensor:
    """Reward for tracking commanded angular velocity.

    Matches mjlab.tasks.velocity.mdp.track_angular_velocity exactly.
    Includes xy angular velocity penalty (commanded xy is assumed to be zero).

    Args:
        env: Genesis environment.
        std: Standard deviation for exponential kernel.
        entity_name: Name of the robot entity.
        base_name: Name of the base link.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get commanded angular velocity (z only)
    command_z = env.command_manager.ang_vel  # (num_envs,)

    # Get actual angular velocity in body frame
    actual = env.get_robot_data(entity_name).root_link_ang_vel_b  # (num_envs, 3)

    # z error + xy error (xy command assumed zero)
    z_error = torch.square(command_z - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error

    return torch.exp(-ang_vel_error / (std ** 2))


# ============================================================
# flat_orientation_mjlab
# ============================================================

def flat_orientation_mjlab(
    env: "GenesisEnv",
    std: float,
    body_name: str | None = None,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward for flat base orientation (robot being upright).

    Matches mjlab.tasks.velocity.mdp.flat_orientation exactly.
    If body_name is specified, computes projected gravity for that specific body.
    Otherwise, uses the root link projected gravity.

    Args:
        env: Genesis environment.
        std: Standard deviation for exponential kernel.
        body_name: Name of the body to compute projected gravity for. If None, uses root.
        entity_name: Name of the robot entity.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    if body_name is not None:
        # Get specific body quaternion in world frame
        entity = env.scene_manager[entity_name]
        link = entity.get_link(name=body_name)
        body_quat_w = entity.get_links_quat(links_idx_local=[link.idx_local])  # (num_envs, 1, 4)
        body_quat_w = body_quat_w.squeeze(1)  # (num_envs, 4)

        # Compute projected gravity for that body
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device)
        projected_gravity_b = transform_by_quat(gravity_w, inv_quat(body_quat_w))
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        # Use root link projected gravity (unit gravity vector via robot_data)
        projected_gravity_b = env.get_robot_data(entity_name).projected_gravity_b  # (num_envs, 3)
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / (std ** 2))


# ============================================================
# variable_posture
# ============================================================

class variable_posture:
    """Penalize deviation from default pose with speed-dependent tolerance.

    Matches mjlab.tasks.velocity.mdp.variable_posture exactly.

    Uses per-joint standard deviations to control how much each joint can deviate
    from default pose. Smaller std = stricter (less deviation allowed), larger
    std = more forgiving. The reward is: exp(-mean(error² / std²))

    Three speed regimes (based on linear + angular command velocity):
      - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
      - std_walking (walking_threshold <= speed < running_threshold): Moderate.
      - std_running (speed >= running_threshold): Loose tolerance for large motion.
    """

    __name__ = "variable_posture"

    def __init__(
        self,
        env: "GenesisEnv",
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ):
        self.walking_threshold = walking_threshold
        self.running_threshold = running_threshold

        joint_names = env.act_manager._actuated_joint_names

        # Resolve std values per joint using regex matching
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

        # Default joint positions
        self.default_joint_pos = env.act_manager.offset

    def __call__(self, env: "GenesisEnv") -> torch.Tensor:
        # Get command velocities
        command = torch.stack([
            env.command_manager.lin_vel_x,
            env.command_manager.lin_vel_y,
            env.command_manager.ang_vel,
        ], dim=1)

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        # Speed regime masks
        standing_mask = (total_speed < self.walking_threshold).float()
        walking_mask = (
            (total_speed >= self.walking_threshold) & (total_speed < self.running_threshold)
        ).float()
        running_mask = (total_speed >= self.running_threshold).float()

        # Select std based on speed regime
        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        # Compute pose error
        current_joint_pos = env.robot_data.joint_pos
        error_squared = torch.square(current_joint_pos - self.default_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std ** 2), dim=1))

    def reset(self, env_ids: torch.Tensor) -> None:
        pass


# ============================================================
# body_ang_vel_penalty_mjlab
# ============================================================

def body_ang_vel_penalty_mjlab(
    env: "GenesisEnv",
    body_name: str,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize excessive body angular velocities (xy only).

    Delegates to ``common.penalize_body_ang_vel_xy``. The Genesis
    implementation of ``RobotData.find_body_index`` calls the same
    ``entity.get_link(name).idx_local`` that the legacy code used, and
    ``RobotData.body_ang_vel_w`` calls the same
    ``entity.get_links_ang(links_idx_local=[idx]).squeeze(1)`` — so the
    result is bit-identical to the legacy direct-access path.

    Args:
        env: Genesis environment.
        body_name: Name of the body to penalize.
        entity_name: Name of the robot entity.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
        penalize_body_ang_vel_xy as _common_fn,
    )
    return _common_fn(env, body_name=body_name, entity_name=entity_name)


# ============================================================
# feet_air_time_mjlab
# ============================================================

def feet_air_time_mjlab(
    env: "GenesisEnv",
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_threshold: float = 0.5,
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Reward feet air time.

    Matches mjlab.tasks.velocity.mdp.feet_air_time exactly.

    Args:
        env: Genesis environment.
        threshold_min: Minimum air time to receive reward.
        threshold_max: Maximum air time to receive reward.
        command_threshold: Minimum command velocity to activate reward.
        contact_group: Name of the contact group to use.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    current_air_time = env.contact_manager.current_air_time(contact_group)

    # Reward if air time is in valid range
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    # Scale by command magnitude
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
    env: "GenesisEnv",
    feet_links: str | list[str],
    target_height: float,
    command_threshold: float = 0.01,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity.

    Matches mjlab.tasks.velocity.mdp.feet_clearance exactly.

    Args:
        env: Genesis environment.
        feet_links: Foot link name pattern(s).
        target_height: Target foot clearance height.
        command_threshold: Minimum command velocity to activate penalty.
        entity_name: Name of the robot entity.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    entity = env.scene_manager[entity_name]

    if isinstance(feet_links, str):
        feet_links = [feet_links]

    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False, preserve_order=True)

    # Get foot positions and velocities
    foot_pos = entity.get_links_pos(links_idx_local=links_idx_local)  # (num_envs, num_feet, 3)
    foot_vel = entity.get_links_vel(links_idx_local=links_idx_local)  # (num_envs, num_feet, 3)

    foot_z = foot_pos[:, :, 2]  # (num_envs, num_feet)
    foot_vel_xy = foot_vel[:, :, :2]  # (num_envs, num_feet, 2)
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # (num_envs, num_feet)

    # Height error weighted by velocity
    delta = torch.abs(foot_z - target_height)
    cost = torch.sum(delta * vel_norm, dim=1)

    # Scale by command magnitude
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
# feet_swing_height_mjlab
# ============================================================

class feet_swing_height_mjlab:
    """Penalize deviation from target swing height, evaluated at landing."""

    __name__ = "feet_swing_height_mjlab"

    def __init__(
        self,
        env: "GenesisEnv",
        feet_links: str | list[str],
        target_height: float,
        command_threshold: float = 0.05,
        entity_name: str = "robot",
        contact_group: str = "feet_ground_contact",
    ):
        self.target_height = target_height
        self.command_threshold = command_threshold
        self.entity_name = entity_name
        self.contact_group = contact_group

        if isinstance(feet_links, str):
            feet_links = [feet_links]
        self.feet_links = feet_links

        # Lazy initialization - will be set on first call
        self._initialized = False
        self.links_idx_local = None
        self.num_feet = None
        self.peak_heights = None

    def _lazy_init(self, env: "GenesisEnv") -> None:
        """Initialize indices on first call when contact_manager is ready."""
        entity = env.scene_manager[self.entity_name]
        self.links_idx_local, _ = eu.find_links(
            entity, list(self.feet_links), global_ids=False, preserve_order=True
        )
        self.num_feet = len(self.links_idx_local)
        self.peak_heights = torch.zeros(
            (env.num_envs, self.num_feet), device=env.device, dtype=torch.float32
        )
        self._initialized = True

    def __call__(self, env: "GenesisEnv") -> torch.Tensor:
        if not self._initialized:
            self._lazy_init(env)

        entity = env.scene_manager[self.entity_name]

        # Get foot heights
        foot_pos = entity.get_links_pos(links_idx_local=self.links_idx_local)
        foot_heights = foot_pos[:, :, 2]

        # Get contact states (order=feet_links to match foot_heights column order)
        feet_order = list(self.feet_links)
        is_contact = env.contact_manager.is_contact(self.contact_group, order=feet_order)
        in_air = ~is_contact

        # Update peak heights during swing
        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )

        # Detect first contact
        first_contact = env.contact_manager.compute_first_contact(self.contact_group, order=feet_order)

        # Get command velocity
        command = torch.stack([
            env.command_manager.lin_vel_x,
            env.command_manager.lin_vel_y,
            env.command_manager.ang_vel,
        ], dim=1)

        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm

        active = (total_command > self.command_threshold).float()

        # Compute cost at landing (squared error like mjlab)
        error = self.peak_heights / self.target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active

        # Reset peak heights after landing
        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )

        return -cost

    def reset(self, env_ids: torch.Tensor) -> None:
        if self.peak_heights is not None:
            self.peak_heights[env_ids] = 0.0


# ============================================================
# feet_slip_mjlab
# ============================================================

def feet_slip_mjlab(
    env: "GenesisEnv",
    feet_links: str | list[str],
    command_threshold: float = 0.05,
    entity_name: str = "robot",
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact).

    Matches mjlab.tasks.velocity.mdp.feet_slip exactly.

    Args:
        env: Genesis environment.
        feet_links: Foot link name pattern(s).
        command_threshold: Minimum command velocity to activate penalty.
        entity_name: Name of the robot entity.
        contact_group: Name of the contact group to use.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    entity = env.scene_manager[entity_name]

    if isinstance(feet_links, str):
        feet_links = [feet_links]

    links_idx_local, _ = eu.find_links(entity, list(feet_links), global_ids=False, preserve_order=True)

    # Get foot velocities
    foot_vel = entity.get_links_vel(links_idx_local=links_idx_local)  # (num_envs, num_feet, 3)
    foot_vel_xy = foot_vel[:, :, :2]  # (num_envs, num_feet, 2)
    vel_xy_norm_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)  # (num_envs, num_feet)

    # Get contact states
    is_contact = env.contact_manager.is_contact(contact_group, order=feet_links)  # (num_envs, num_feet)

    # Command scaling
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
    env: "GenesisEnv",
    command_threshold: float = 0.05,
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Penalize high impact forces at landing.

    Matches mjlab.tasks.velocity.mdp.soft_landing exactly.

    Args:
        env: Genesis environment.
        command_threshold: Minimum command velocity to activate penalty.
        contact_group: Name of the contact group to use.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    # Get contact force magnitudes
    forces_3d = env.contact_manager.contact_force(contact_group)  # (num_envs, num_feet, 3)
    forces = torch.norm(forces_3d, dim=-1)  # (num_envs, num_feet)

    # Detect first contact
    first_contact = env.contact_manager.compute_first_contact(contact_group)

    # Landing impact
    landing_impact = forces * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)

    # Command scaling
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
    env: "GenesisEnv",
    soft_limit_factor: float = 1.0,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize joint positions exceeding soft limits.

    Delegates to ``common.penalize_joint_pos_limits_l1`` which reads
    ``RobotData.joint_pos`` and ``RobotData.joint_pos_limits``. The
    Genesis implementation of ``joint_pos_limits`` calls the same
    ``entity.get_dofs_limit(actuated_dof_ids)`` that the legacy code
    used (with a ``squeeze(0)`` to drop the leading dim), so the
    resulting penalty is bit-identical to the legacy direct-access path.

    Args:
        env: Genesis environment.
        soft_limit_factor: Factor to scale hard limits to get soft limits.
        entity_name: Name of the robot entity.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
        penalize_joint_pos_limits_l1 as _common_fn,
    )
    return _common_fn(env, soft_limit_factor=soft_limit_factor, entity_name=entity_name)


# ============================================================
# action_rate_l2_mjlab
# ============================================================

def raw_action_rate_l2_mjlab(env: "GenesisEnv") -> torch.Tensor:
    """Penalize the rate of change of raw actions (L2 squared).

    Delegates to ``common.raw_action_rate_l2``. Bit-identical: same
    pure-Python ``act_manager`` arithmetic, no scene-state involved.
    """
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import raw_action_rate_l2
    return raw_action_rate_l2(env)

def processed_action_rate_l2_mjlab(env: "GenesisEnv") -> torch.Tensor:
    """Penalize the rate of change of processed actions using L2 squared kernel.

    Matches mjlab.envs.mdp.action_rate_l2 exactly.

    Args:
        env: Genesis environment.

    Returns:
        Penalty tensor of shape (num_envs,).
    """

    return -torch.sum(
        torch.square(env.act_manager.processed_actions - env.act_manager.prev_processed_actions),
        dim=1
    )