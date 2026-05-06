from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from rlworld.rl.envs.mdp.observations.mujoco.proprioception import quat_apply_inverse  # used by flat_orientation
from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
    FeetSwingHeightTracker,
    VariablePostureTracker,
    flat_orientation as flat_orientation_l2_common,
    get_leg_xy_signs,
    penalize_angular_momentum_l2,
    penalize_body_ang_vel_xy,
    penalize_contact_force_count,
    penalize_feet_clearance,
    penalize_feet_slip,
    penalize_lin_vel_z,
    penalize_soft_landing,
)
from rlworld.rl.utils import string as string_utils
from rlworld.rl.utils.quat_utils import quat_apply_yaw_wxyz, quat_conjugate_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs.mujoco import MujocoEnv, MujocoLocomotionEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def is_alive(env: MujocoEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.dones).float()


def is_terminated(env: MujocoEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.dones.float()


def track_linear_velocity(
    env: MujocoEnv,
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

    return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
    env: MujocoEnv,
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

    return torch.exp(-ang_vel_error / std**2)


# =============================================================================
# Joint-based rewards/penalties
# =============================================================================


def joint_torques_l2(
    env: MujocoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    return torch.sum(torch.square(robot.data.actuator_force), dim=1)


def joint_vel_l2(
    env: MujocoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    return torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
    env: MujocoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    robot = env.scene_manager.get_entity(asset_cfg.name)
    return torch.sum(torch.square(robot.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
    env: MujocoEnv,
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
    env: MujocoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize non-flat base orientation.

    Delegates to ``common.flat_orientation(std=None)`` which computes
    the same ``-sum(projected_gravity_xy²)`` penalty.
    """
    return flat_orientation_l2_common(env, entity_name=asset_cfg.name)


def flat_orientation(
    env: MujocoEnv,
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

    return torch.exp(-xy_squared / std**2)


# =============================================================================
# Body penalties
# =============================================================================


def body_angular_velocity_penalty(
    env: MujocoEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive body angular velocities (xy only).

    Thin wrapper that translates the mjlab ``asset_cfg`` parameter
    convention into the sim-agnostic ``body_name`` convention used by
    ``common.penalize_body_ang_vel_xy``. When ``asset_cfg.body_ids`` is
    a concrete index list (the active-preset path), we extract the
    first body name from ``asset_cfg.body_names`` and delegate. The
    legacy fallback path that uses ``robot.data.root_link_ang_vel_w``
    when ``body_ids is None`` is preserved here as a direct accessor
    call to keep mjlab's API surface intact.
    """
    if asset_cfg.body_ids is None or isinstance(asset_cfg.body_ids, slice):
        # Legacy fallback: penalize root angular velocity directly.
        robot = env.scene_manager.get_entity(asset_cfg.name)
        ang_vel = robot.data.root_link_ang_vel_w
        ang_vel_xy = ang_vel[:, :2]
        return -torch.sum(torch.square(ang_vel_xy), dim=1)

    # Active path: a concrete body was specified. Delegate to common.
    body_name = asset_cfg.body_names[0]
    return penalize_body_ang_vel_xy(env, body_name=body_name, entity_name=asset_cfg.name)


def angular_momentum_penalty(
    env: MujocoEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize whole-body angular momentum to encourage natural arm swing.

    Delegates to ``common.penalize_angular_momentum_l2`` which calls
    ``RobotData.angular_momentum_w(sensor_name)``. For mjlab this
    in turn calls ``env.scene_manager.get_sensor(sensor_name)`` —
    bit-identical to the legacy direct sensor read.
    """
    return penalize_angular_momentum_l2(env, sensor_name=sensor_name)


def self_collision_cost(
    env: MujocoEnv,
    contact_group: str = "body_ground_contact",
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Thin redirect to ``common.penalize_contact_force_count``.

    Bit-identical to the legacy implementation: the common helper uses
    ``contact_manager.contact_force_history`` first (mjlab returns the
    substep history when registered with ``history_length > 0``), then
    falls back to instantaneous ``contact_force`` — exactly the legacy
    branching.
    """
    return penalize_contact_force_count(env, contact_group=contact_group, force_threshold=force_threshold)


def wtw_collision(
    env: MujocoEnv,
    contact_group: str = "body_ground_contact",
    force_threshold: float = 0.1,
) -> torch.Tensor:
    """Thin redirect to ``common.penalize_contact_force_count``."""
    return penalize_contact_force_count(env, contact_group=contact_group, force_threshold=force_threshold)


def feet_air_time(
    env: MujocoEnv,
    contact_group: str = "feet_ground_contact",
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward feet air time."""
    current_air_time = env.contact_manager.current_air_time(contact_group)

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    command = env.command_manager.get_commands_tensor()
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    scale = (total_command > command_threshold).float()

    return reward * scale


def feet_clearance(
    env: MujocoEnv,
    target_height: float,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Thin redirect to ``common.penalize_feet_clearance``.

    Bit-identical: ``RobotData.site_pos_w/site_lin_vel_w`` for MuJoCo
    call ``entity.find_sites`` and read the same ``data.site_pos_w /
    data.site_lin_vel_w`` arrays the legacy code accessed via
    ``asset_cfg.site_ids``. The site name list is taken straight from
    ``asset_cfg.site_names`` to preserve the same column ordering.
    """
    return penalize_feet_clearance(
        env,
        target_height=target_height,
        command_threshold=command_threshold,
        site_names=list(asset_cfg.site_names),
        entity_name=asset_cfg.name,
    )


def feet_slip(
    env: MujocoEnv,
    contact_group: str = "feet_ground_contact",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Thin redirect to ``common.penalize_feet_slip``.

    Bit-identical: site velocities pulled via ``RobotData.site_lin_vel_w``
    (which uses the same ``data.site_lin_vel_w`` array indexed by the
    same site_ids), and contact tensor reads the natural group order
    (``contact_order=None``) — matching the legacy MuJoCo path.
    """
    return penalize_feet_slip(
        env,
        contact_group=contact_group,
        command_threshold=command_threshold,
        site_names=list(asset_cfg.site_names),
        entity_name=asset_cfg.name,
    )


def soft_landing(
    env: MujocoEnv,
    contact_group: str = "feet_ground_contact",
    command_threshold: float = 0.05,
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


def alive_bonus(env: MujocoEnv) -> torch.Tensor:
    """Constant reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def lin_vel_z_penalty(env: MujocoEnv) -> torch.Tensor:
    """Penalize vertical velocity to discourage bouncing.

    Delegates to ``common.penalize_lin_vel_z``.
    """
    return penalize_lin_vel_z(env)


class variable_posture:
    """Thin wrapper around ``common.VariablePostureTracker``.

    Bit-identical to the legacy MuJoCo implementation: pre-resolves the
    asset_cfg joint subset at construction time, slices
    ``robot.data.default_joint_pos`` by ``joint_ids`` for the default
    pose tensor, and provides a per-step closure that returns
    ``robot.data.joint_pos[:, joint_ids]``. The closure binds
    ``asset_cfg.name`` and ``joint_ids`` so subsequent calls do not need
    to re-resolve them.
    """

    __name__ = "variable_posture"

    def __init__(
        self,
        env: MujocoEnv,
        asset_cfg: SceneEntityCfg,
        std_standing: Dict[str, float],
        std_walking: Dict[str, float],
        std_running: Dict[str, float],
        walking_threshold: float = 0.05,
        running_threshold: float = 1.5,
    ):
        robot = env.scene_manager.get_entity(asset_cfg.name)
        default_joint_pos = robot.data.default_joint_pos
        assert default_joint_pos is not None

        _, joint_names = robot.find_joints(asset_cfg.joint_names)
        joint_ids = asset_cfg.joint_ids
        entity_name = asset_cfg.name

        sliced_default = default_joint_pos[:, joint_ids]

        def _get_current(e, _name=entity_name, _ids=joint_ids):
            return e.scene_manager.get_entity(_name).data.joint_pos[:, _ids]

        self._impl = VariablePostureTracker(
            env=env,
            joint_names=joint_names,
            std_standing=std_standing,
            std_walking=std_walking,
            std_running=std_running,
            get_current_joint_pos=_get_current,
            default_joint_pos=sliced_default,
            walking_threshold=walking_threshold,
            running_threshold=running_threshold,
        )

    def __call__(self, env: MujocoEnv, **kwargs) -> torch.Tensor:
        return self._impl(env)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._impl.reset(env_ids)


class feet_swing_height:
    """Thin wrapper around ``common.FeetSwingHeightTracker`` (MuJoCo legacy).

    Preserves bit-identity by setting ``reset_mode="none"`` — the
    original MuJoCo class had no ``reset`` method, so peak heights
    persisted across episode resets and were only zeroed naturally on
    landing. ``contact_order=None`` because legacy MuJoCo relies on the
    natural contact-group order matching site order.
    """

    __name__ = "feet_swing_height"

    def __init__(
        self,
        env: MujocoEnv,
        contact_group: str = "feet_ground_contact",
        target_height: float = 0.08,
        command_threshold: float = 0.01,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ):
        self._impl = FeetSwingHeightTracker(
            env=env,
            contact_group=contact_group,
            target_height=target_height,
            command_threshold=command_threshold,
            site_names=list(asset_cfg.site_names),
            entity_name=asset_cfg.name,
            use_squared_error=True,
            reset_mode="none",
        )

    def __call__(self, env: MujocoEnv, **kwargs) -> torch.Tensor:
        return self._impl(env)


class posture:
    """Penalize the deviation of the joint positions from the default positions."""

    __name__ = "posture"

    def __init__(
        self,
        env: MujocoEnv,
        std: float | Dict[str, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ):
        robot = env.scene_manager.get_entity(asset_cfg.name)
        default_joint_pos = robot.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        joint_ids = (
            asset_cfg.joint_ids
            if asset_cfg.joint_ids is not None and not isinstance(asset_cfg.joint_ids, slice)
            else slice(None)
        )
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

    def __call__(self, env: MujocoEnv, **kwargs) -> torch.Tensor:
        robot = env.scene_manager.get_entity("robot")
        current_joint_pos = robot.data.joint_pos[:, self._joint_ids]
        desired_joint_pos = self.default_joint_pos[:, self._joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))


# ── Walk-These-Ways reward terms (MuJoCo) ────────────────────────────────


def _contact_order_matching_gait(env: MujocoLocomotionEnv, contact_group: str) -> list[str]:
    """Permute a contact group's tracked_names so column ``i`` is the
    same foot as column ``i`` of ``env.gait_manager.foot_names``.

    Matches each gait foot name to the unique contact tracked name that
    contains it as a substring (e.g. ``"FR_foot"`` ↔
    ``"FR_foot_collision"``). Raises if the match is missing or
    ambiguous so any future naming drift surfaces immediately.
    """
    gait_order = list(env.gait_manager.foot_names)
    tracked = list(env.contact_manager.tracked_names(contact_group))
    out: list[str] = []
    for g in gait_order:
        candidates = [t for t in tracked if g in t]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot map gait foot {g!r} to contact group "
                f"{contact_group!r}: candidates {candidates}, "
                f"tracked {tracked}."
            )
        out.append(candidates[0])
    return out


def _site_ids_matching_gait(env: MujocoLocomotionEnv, asset_cfg: SceneEntityCfg) -> list[int]:
    """Permute ``asset_cfg.site_ids`` so column ``i`` is the same foot
    as column ``i`` of ``env.gait_manager.foot_names``. Matches site
    names to gait foot names by substring (either direction), so
    ``("FR","FL","RR","RL")`` matches ``("FR_foot","FL_foot","RR_foot","RL_foot")``.

    Independent of the preset's ``preserve_order`` setting: even if
    SceneEntityCfg.resolve() reordered ``site_ids`` away from the
    user-supplied tuple, this helper maps it back to gait order.
    """
    gait_order = list(env.gait_manager.foot_names)
    site_pairs = list(zip(asset_cfg.site_names, asset_cfg.site_ids))
    out: list[int] = []
    for g in gait_order:
        candidates = [i for sn, i in site_pairs if sn in g or g in sn]
        if len(candidates) != 1:
            raise ValueError(f"Cannot map gait foot {g!r} to asset_cfg sites {site_pairs}: candidates {candidates}.")
        out.append(candidates[0])
    return out


def wtw_feet_slip(
    env: MujocoLocomotionEnv,
    contact_group: str = "feet_ground_contact",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WTW feet slip: penalize foot xy velocity when in contact OR was in contact.

    Both contact and site arrays are read in ``gait_manager.foot_names``
    order so the elementwise multiply pairs each foot's contact state
    with its own velocity.
    """
    robot = env.scene_manager.get_entity(asset_cfg.name)
    contact_order = _contact_order_matching_gait(env, contact_group)
    site_ids = _site_ids_matching_gait(env, asset_cfg)

    in_contact = env.contact_manager.is_contact(contact_group, order=contact_order)
    prev_contact = env.contact_manager.prev_is_contact(contact_group, order=contact_order)
    contact_filt = (in_contact | prev_contact).float()

    foot_vel_xy = robot.data.site_lin_vel_w[:, site_ids, :2]
    vel_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)
    return -torch.sum(contact_filt * vel_sq, dim=-1)


def wtw_tracking_contacts_shaped_force(
    env: MujocoLocomotionEnv,
    contact_group: str = "feet_ground_contact",
    gait_force_sigma: float = 100.0,
) -> torch.Tensor:
    """WTW: penalize foot contact force when foot should be in swing.

    Contact forces are reordered to ``gait_manager.foot_names`` so that
    column ``i`` of ``foot_forces`` is the same foot as column ``i`` of
    ``desired_contact_states``.
    """
    contact_order = _contact_order_matching_gait(env, contact_group)
    forces = env.contact_manager.contact_force(contact_group, order=contact_order)
    foot_forces = torch.norm(forces, dim=-1)

    desired_contact = env.gait_manager.desired_contact_states
    reward = -(1.0 - desired_contact) * (1.0 - torch.exp(-(foot_forces**2) / gait_force_sigma))
    return reward.mean(dim=-1)


def wtw_tracking_contacts_shaped_vel(
    env: MujocoLocomotionEnv,
    gait_vel_sigma: float = 10.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WTW: penalize foot velocity when foot should be in stance.

    Site velocities are reordered to ``gait_manager.foot_names`` so the
    elementwise multiply with ``desired_contact_states`` lines up.
    """
    robot = env.scene_manager.get_entity(asset_cfg.name)
    site_ids = _site_ids_matching_gait(env, asset_cfg)
    foot_vel = robot.data.site_lin_vel_w[:, site_ids, :]
    foot_vel_norm = torch.norm(foot_vel, dim=-1)

    desired_contact = env.gait_manager.desired_contact_states

    reward = -(desired_contact * (1.0 - torch.exp(-(foot_vel_norm**2) / gait_vel_sigma)))
    return reward.mean(dim=-1)


def wtw_feet_clearance_cmd_linear(
    env: MujocoLocomotionEnv,
    foot_radius: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WTW: penalize foot height error during swing, scaled by commanded footswing height.

    Site heights are reordered to ``gait_manager.foot_names`` so the
    elementwise math against ``foot_phases`` / ``desired_contact_states``
    lines up.
    """
    robot = env.scene_manager.get_entity(asset_cfg.name)
    site_ids = _site_ids_matching_gait(env, asset_cfg)
    foot_height = robot.data.site_pos_w[:, site_ids, 2]

    foot_phases = env.gait_manager.foot_phases
    phases = 1.0 - torch.abs(1.0 - torch.clip((foot_phases * 2.0) - 1.0, 0.0, 1.0) * 2.0)

    footswing_height = env.command_manager.footswing_height
    target_height = footswing_height.unsqueeze(1) * phases + foot_radius

    desired_contact = env.gait_manager.desired_contact_states
    clearance_error = torch.square(target_height - foot_height) * (1.0 - desired_contact)
    return -torch.sum(clearance_error, dim=-1)


def wtw_raibert_heuristic(
    env: MujocoLocomotionEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """WTW: penalize footstep placement error vs Raibert heuristic."""
    feet_names = env.gait_manager.foot_names

    # Get foot positions in gait_manager.foot_names order
    robot = env.scene_manager.get_entity(asset_cfg.name)
    all_site_positions = robot.data.site_pos_w[:, asset_cfg.site_ids, :]
    # site_names from config may differ from gait_manager order — reindex
    site_name_list = list(asset_cfg.site_names)
    foot_name_to_site_idx = {sname: i for i, sname in enumerate(site_name_list)}
    reindex = [foot_name_to_site_idx[fn.replace("_foot", "")] for fn in feet_names]
    foot_positions = all_site_positions[:, reindex, :]

    base_pos = env.get_robot_data().root_link_pos_w
    base_quat = env.get_robot_data().root_link_quat_w

    num_feet = foot_positions.shape[1]
    cur_footsteps_translated = foot_positions - base_pos.unsqueeze(1)

    footsteps_in_body = torch.zeros_like(cur_footsteps_translated)
    for i in range(num_feet):
        footsteps_in_body[:, i, :] = quat_apply_yaw_wxyz(
            quat_conjugate_wxyz(base_quat), cur_footsteps_translated[:, i, :]
        )

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
