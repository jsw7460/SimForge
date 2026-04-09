"""mjlab-compatible reward functions for Genesis environments.

These functions produce identical outputs to mjlab rewards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesis.utils.geom import transform_by_quat, inv_quat
from rlworld.rl.envs.mdp.observations.genesis import proprioception
from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
    FeetSwingHeightTracker,
    VariablePostureTracker,
    penalize_body_ang_vel_xy,
    penalize_contact_force_count,
    penalize_feet_clearance,
    penalize_feet_slip,
    penalize_joint_pos_limits_l1,
    penalize_soft_landing,
    raw_action_rate_l2,
)

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
    """Thin wrapper around ``common.VariablePostureTracker``.

    Bit-identical to the legacy Genesis implementation. The legacy code
    read ``env.act_manager._actuated_joint_names`` (private name), but
    the public ``actuated_joint_names`` property exposes the same list
    so we use that here.
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
        self._impl = VariablePostureTracker(
            env=env,
            joint_names=list(env.act_manager.actuated_joint_names),
            std_standing=std_standing,
            std_walking=std_walking,
            std_running=std_running,
            get_current_joint_pos=lambda e: e.robot_data.joint_pos,
            default_joint_pos=env.act_manager.offset,
            walking_threshold=walking_threshold,
            running_threshold=running_threshold,
        )

    def __call__(self, env: "GenesisEnv") -> torch.Tensor:
        return self._impl(env)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._impl.reset(env_ids)


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
    return penalize_body_ang_vel_xy(env, body_name=body_name, entity_name=entity_name)


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
    """Thin redirect to ``common.penalize_feet_clearance``.

    Bit-identical: ``RobotData.body_pos_w/body_lin_vel_w`` for Genesis
    call ``entity.get_links_pos/get_links_vel`` with the same
    ``links_idx_local`` the legacy code resolved via ``eu.find_links``.
    """
    names = [feet_links] if isinstance(feet_links, str) else list(feet_links)
    return penalize_feet_clearance(
        env,
        target_height=target_height,
        command_threshold=command_threshold,
        body_names=names,
        entity_name=entity_name,
    )


# ============================================================
# feet_swing_height_mjlab
# ============================================================

class feet_swing_height_mjlab:
    """Thin wrapper around ``common.FeetSwingHeightTracker`` (Genesis legacy reset).

    Preserves bit-identity by setting ``reset_mode="zero"`` (Genesis's
    original behavior).
    """

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
        names = [feet_links] if isinstance(feet_links, str) else list(feet_links)
        self._impl = FeetSwingHeightTracker(
            env=env,
            contact_group=contact_group,
            target_height=target_height,
            command_threshold=command_threshold,
            body_names=names,
            entity_name=entity_name,
            use_squared_error=True,
            reset_mode="zero",
        )

    def __call__(self, env: "GenesisEnv") -> torch.Tensor:
        return self._impl(env)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._impl.reset(env_ids)


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
    """Thin redirect to ``common.penalize_feet_slip``.

    Bit-identical: the legacy code already passed ``order=feet_links``
    to ``contact_manager.is_contact``, which is exactly what the common
    helper does (defaulting ``contact_order`` to ``body_names``).
    """
    names = [feet_links] if isinstance(feet_links, str) else list(feet_links)
    return penalize_feet_slip(
        env,
        contact_group=contact_group,
        command_threshold=command_threshold,
        body_names=names,
        entity_name=entity_name,
    )


# ============================================================
# soft_landing_mjlab
# ============================================================

def soft_landing_mjlab(
    env: "GenesisEnv",
    command_threshold: float = 0.05,
    contact_group: str = "feet_ground_contact",
) -> torch.Tensor:
    """Thin redirect to ``common.penalize_soft_landing``.

    Bit-identical: legacy code summed forces over the natural group
    order; the common helper does the same when ``contact_order=None``.
    """
    return penalize_soft_landing(
        env,
        contact_group=contact_group,
        command_threshold=command_threshold,
    )


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
    return penalize_joint_pos_limits_l1(
        env, soft_limit_factor=soft_limit_factor, entity_name=entity_name
    )


# ============================================================
# action_rate_l2_mjlab
# ============================================================

def raw_action_rate_l2_mjlab(env: "GenesisEnv") -> torch.Tensor:
    """Penalize the rate of change of raw actions (L2 squared).

    Delegates to ``common.raw_action_rate_l2``. Bit-identical: same
    pure-Python ``act_manager`` arithmetic, no scene-state involved.
    """
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