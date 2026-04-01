from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from rlworld.rl.envs.mdp.observations.mujoco.proprioception import quat_apply_inverse
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs.mujoco import MujocoEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def is_alive(env: "MujocoEnv") -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.dones).float()


def is_terminated(env: "MujocoEnv") -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.dones.float()


def track_linear_velocity(
    env: "MujocoEnv",
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.

    The commanded z velocity is assumed to be zero.
    """
    robot = env.scene_manager.get_entity(asset_cfg.name)
    command = env.command_manager.get_commands_tensor()
    actual = robot.data.root_link_lin_vel_b

    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error

    return torch.exp(-lin_vel_error / std ** 2)


def track_angular_velocity(
    env: "MujocoEnv",
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base angular velocity.

    The commanded xy angular velocities are assumed to be zero.
    """
    robot = env.scene_manager.get_entity(asset_cfg.name)
    command = env.command_manager.get_commands_tensor()
    actual = robot.data.root_link_ang_vel_b

    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error

    return torch.exp(-ang_vel_error / std ** 2)


# =============================================================================
# Joint-based rewards/penalties
# =============================================================================

def joint_torques_l2(
    env: "MujocoEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    return torch.sum(torch.square(robot.data.actuator_force), dim=1)


def joint_vel_l2(
    env: "MujocoEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    return torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
    env: "MujocoEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    return torch.sum(torch.square(robot.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: "MujocoEnv") -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return -torch.sum(
        torch.square(env.act_manager.processed_actions - env.act_manager.prev_processed_actions),
        dim=1
    )


def raw_action_rate_l2(env: "MujocoEnv") -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return -torch.sum(
        torch.square(env.act_manager.raw_actions - env.act_manager.prev_raw_actions),
        dim=1
    )


def joint_pos_limits(
    env: "MujocoEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    soft_joint_pos_limits = robot.data.soft_joint_pos_limits

    if soft_joint_pos_limits is None:
        return torch.zeros(env.num_envs, device=env.device)

    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids else slice(None)
    joint_pos = robot.data.joint_pos[:, joint_ids]

    out_of_limits = -(joint_pos - soft_joint_pos_limits[:, joint_ids, 0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_joint_pos_limits[:, joint_ids, 1]).clip(min=0.0)

    return -torch.sum(out_of_limits, dim=1)


def flat_orientation_l2(
    env: "MujocoEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    return -torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)


def flat_orientation(
    env: "MujocoEnv",
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward flat base orientation (robot being upright).

    If asset_cfg has body_ids specified, computes the projected gravity
    for that specific body. Otherwise, uses the root link projected gravity.
    """
    robot = env.scene_manager.get_entity(asset_cfg.name)

    # Check if body_ids is a valid list/tuple (not None, not slice)
    if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice):
        body_quat_w = robot.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]  # [num_envs, 4]
        gravity_w = robot.data.gravity_vec_w  # [num_envs, 3]
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [num_envs, 3]
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        xy_squared = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std ** 2)


# =============================================================================
# Body penalties
# =============================================================================

def body_angular_velocity_penalty(
    env: "MujocoEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive body angular velocities."""
    robot = env.scene_manager.get_entity(asset_cfg.name)

    if asset_cfg.body_ids is not None and not isinstance(asset_cfg.body_ids, slice):
        ang_vel = robot.data.body_link_ang_vel_w[:, asset_cfg.body_ids[0], :]
    else:
        ang_vel = robot.data.root_link_ang_vel_w

    ang_vel_xy = ang_vel[:, :2]
    return -torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
    env: "MujocoEnv",
    sensor_name: str,
) -> torch.Tensor:
    """Penalize whole-body angular momentum to encourage natural arm swing."""
    angmom_sensor = env.scene_manager.get_sensor(sensor_name)
    angmom = angmom_sensor.data
    angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
    return -angmom_magnitude_sq


def self_collision_cost(
    env: "MujocoEnv",
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collisions.

    When the sensor has force_history (history_length > 0), detects collisions
    across decimation substeps using force magnitude thresholding. Otherwise,
    falls back to the instantaneous ``found`` flag.
    """
    sensor = env.scene_manager.get_sensor(sensor_name)
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)       # [B, H]
        return -hit.sum(dim=-1).float()                      # [B]
    return -data.found.squeeze(-1).float()


def wtw_collision(
    env: "MujocoEnv",
    sensor_name: str,
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """WTW collision: count non-foot bodies with contact force > threshold.

    Matches WTW _reward_collision: counts bodies where net contact force
    magnitude exceeds threshold, regardless of contact source (ground + self).
    """
    sensor = env.scene_manager.get_sensor(sensor_name)
    data = sensor.data

    if data.force is not None:
        force_mag = torch.norm(data.force, dim=-1)  # [B, N]
        return -torch.sum((force_mag > force_threshold).float(), dim=-1)

    if data.found is not None:
        found = data.found
        if found.dim() == 3:
            found = found.squeeze(-1)
        return -torch.sum((found > 0).float(), dim=-1)

    return torch.zeros(env.num_envs, device=env.device)


def feet_air_time(
    env: "MujocoEnv",
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward feet air time."""
    sensor = env.scene_manager.get_sensor(sensor_name)
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    command = env.command_manager.get_commands_tensor()
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    scale = (total_command > command_threshold).float()

    return reward * scale


def feet_clearance(
    env: "MujocoEnv",
    target_height: float,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity."""
    robot = env.scene_manager.get_entity(asset_cfg.name)

    foot_z = robot.data.site_pos_w[:, asset_cfg.site_ids, 2]
    foot_vel_xy = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)

    delta = torch.abs(foot_z - target_height)
    cost = torch.sum(delta * vel_norm, dim=1)

    command = env.command_manager.get_commands_tensor()
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()

    return -cost * active


def feet_slip(
    env: "MujocoEnv",
    sensor_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact)."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    contact_sensor = env.scene_manager.get_sensor(sensor_name)
    command = env.command_manager.get_commands_tensor()

    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()

    data = contact_sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        in_contact = (force_mag > 1e-6).any(dim=-1).float()  # [B, N]
    else:
        in_contact = (data.found > 0).float()
    foot_vel_xy = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    vel_xy_norm_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)

    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active

    return -cost


def soft_landing(
    env: "MujocoEnv",
    sensor_name: str,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor = env.scene_manager.get_sensor(sensor_name)
    sensor_data = contact_sensor.data

    forces = sensor_data.force
    force_magnitude = torch.norm(forces, dim=-1)

    first_contact = contact_sensor.compute_first_contact(dt=env.control_dt)

    landing_impact = force_magnitude * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)

    command = env.command_manager.get_commands_tensor()
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()

    return -cost * active


def alive_bonus(env: "MujocoEnv") -> torch.Tensor:
    """Constant reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def lin_vel_z_penalty(env: "MujocoEnv") -> torch.Tensor:
    """Penalize vertical velocity to discourage bouncing."""
    robot = env.scene_manager.get_entity("robot")
    base_lin_vel = robot.data.root_link_lin_vel_b
    return -torch.square(base_lin_vel[:, 2])


class variable_posture:
    """Penalize deviation from default pose with speed-dependent tolerance."""

    __name__ = "variable_posture"

    def __init__(
        self,
        env: "MujocoEnv",
        asset_cfg: SceneEntityCfg,
        std_standing: Dict[str, float],
        std_walking: Dict[str, float],
        std_running: Dict[str, float],
        walking_threshold: float = 0.05,
        running_threshold: float = 1.5,
    ):
        self._env = env
        self._asset_cfg = asset_cfg
        self._walking_threshold = walking_threshold
        self._running_threshold = running_threshold

        robot = env.scene_manager.get_entity(asset_cfg.name)
        default_joint_pos = robot.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = robot.find_joints(asset_cfg.joint_names)

        _, _, std_standing_vals = string_utils.resolve_matching_names_values(
            data=std_standing,
            list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(
            std_standing_vals, device=env.device, dtype=torch.float32
        )

        _, _, std_walking_vals = string_utils.resolve_matching_names_values(
            data=std_walking,
            list_of_strings=joint_names,
        )
        self.std_walking = torch.tensor(
            std_walking_vals, device=env.device, dtype=torch.float32
        )

        _, _, std_running_vals = string_utils.resolve_matching_names_values(
            data=std_running,
            list_of_strings=joint_names,
        )
        self.std_running = torch.tensor(
            std_running_vals, device=env.device, dtype=torch.float32
        )

    def __call__(self, env: "MujocoEnv", **kwargs) -> torch.Tensor:
        robot = env.scene_manager.get_entity(self._asset_cfg.name)
        command = env.command_manager.get_commands_tensor()

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < self._walking_threshold).float()
        walking_mask = (
            (total_speed >= self._walking_threshold) & (total_speed < self._running_threshold)
        ).float()
        running_mask = (total_speed >= self._running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = robot.data.joint_pos[:, self._asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, self._asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std ** 2), dim=1))


class feet_swing_height:
    """Penalize deviation from target swing height, evaluated at landing."""

    __name__ = "feet_swing_height"

    def __init__(
        self,
        env: "MujocoEnv",
        sensor_name: str,
        target_height: float,
        command_threshold: float,
        asset_cfg: SceneEntityCfg,
    ):
        self._sensor_name = sensor_name
        self._target_height = target_height
        self._command_threshold = command_threshold
        self._asset_cfg = asset_cfg

        num_sites = len(asset_cfg.site_names)
        self.peak_heights = torch.zeros(
            (env.num_envs, num_sites), device=env.device, dtype=torch.float32
        )
        self.control_dt = env.control_dt

    def __call__(self, env: "MujocoEnv", **kwargs) -> torch.Tensor:
        robot = env.scene_manager.get_entity(self._asset_cfg.name)
        contact_sensor = env.scene_manager.get_sensor(self._sensor_name)
        command = env.command_manager.get_commands_tensor()

        foot_heights = robot.data.site_pos_w[:, self._asset_cfg.site_ids, 2]
        in_air = contact_sensor.data.found == 0

        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )

        first_contact = contact_sensor.compute_first_contact(dt=self.control_dt)

        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > self._command_threshold).float()

        error = self.peak_heights / self._target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active

        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )

        return -cost


class posture:
    """Penalize the deviation of the joint positions from the default positions."""

    __name__ = "posture"

    def __init__(
        self,
        env: "MujocoEnv",
        std: float | Dict[str, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ):
        robot = env.scene_manager.get_entity(asset_cfg.name)
        default_joint_pos = robot.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None and not isinstance(asset_cfg.joint_ids, slice) else slice(None)
        self._joint_ids = joint_ids

        if isinstance(std, dict):
            _, joint_names = robot.find_joints(asset_cfg.joint_names)
            _, _, std_vals = string_utils.resolve_matching_names_values(
                data=std,
                list_of_strings=joint_names,
            )
            self.std = torch.tensor(std_vals, device=env.device, dtype=torch.float32)
        else:
            num_joints = robot.data.joint_pos.shape[1] if isinstance(joint_ids, slice) else len(joint_ids)
            self.std = torch.full((num_joints,), std, device=env.device, dtype=torch.float32)

    def __call__(self, env: "MujocoEnv", **kwargs) -> torch.Tensor:
        robot = env.scene_manager.get_entity("robot")
        current_joint_pos = robot.data.joint_pos[:, self._joint_ids]
        desired_joint_pos = self.default_joint_pos[:, self._joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


# ── Walk-These-Ways reward terms (MuJoCo) ────────────────────────────────

def wtw_feet_slip(
    env: "MujocoEnv",
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WTW feet slip: penalize foot xy velocity when in contact OR was in contact."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    contact_sensor = env.scene_manager.get_sensor(sensor_name)

    data = contact_sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        in_contact = (force_mag > 1e-6).any(dim=-1)
    else:
        in_contact = (data.found > 0).squeeze(-1).bool()

    # prev_is_contact via contact_manager
    link_indices = list(range(in_contact.shape[1]))
    prev_contact = env.contact_manager.prev_is_contact[:, link_indices]
    contact_filt = (in_contact | prev_contact).float()

    foot_vel_xy = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    vel_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)
    return -torch.sum(contact_filt * vel_sq, dim=-1)


def wtw_tracking_contacts_shaped_force(
    env: "MujocoEnv",
    sensor_name: str,
    gait_force_sigma: float = 100.0,
) -> torch.Tensor:
    """WTW: penalize foot contact force when foot should be in swing."""
    contact_sensor = env.scene_manager.get_sensor(sensor_name)
    forces = contact_sensor.data.force
    foot_forces = torch.norm(forces, dim=-1)

    desired_contact = env.gait_manager.desired_contact_states
    num_feet = desired_contact.shape[1]

    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(num_feet):
        reward += -(1.0 - desired_contact[:, i]) * (
            1.0 - torch.exp(-foot_forces[:, i] ** 2 / gait_force_sigma)
        )
    return reward / num_feet


def wtw_tracking_contacts_shaped_vel(
    env: "MujocoEnv",
    gait_vel_sigma: float = 10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WTW: penalize foot velocity when foot should be in stance."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    foot_vel = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
    foot_vel_norm = torch.norm(foot_vel, dim=-1)

    desired_contact = env.gait_manager.desired_contact_states
    num_feet = desired_contact.shape[1]

    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(num_feet):
        reward += -(desired_contact[:, i] * (
            1.0 - torch.exp(-foot_vel_norm[:, i] ** 2 / gait_vel_sigma)
        ))
    return reward / num_feet


def wtw_feet_clearance_cmd_linear(
    env: "MujocoEnv",
    foot_radius: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WTW: penalize foot height error during swing, scaled by commanded footswing height."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    foot_height = robot.data.site_pos_w[:, asset_cfg.site_ids, 2]

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
    env: "MujocoEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WTW: penalize footstep placement error vs Raibert heuristic."""
    from rlworld.rl.utils.quat_utils import quat_apply_yaw_wxyz, quat_conjugate_wxyz

    robot = env.scene_manager.get_entity(asset_cfg.name)
    foot_positions = robot.data.site_pos_w[:, asset_cfg.site_ids, :]
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
