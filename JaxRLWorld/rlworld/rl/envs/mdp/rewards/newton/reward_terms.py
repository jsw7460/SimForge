"""Newton-specific reward terms.

Only contains functions that depend on Newton-specific APIs (Warp body
state, Newton contact format, etc.) and cannot be expressed through
the common ``RobotData`` / ``contact_manager`` interface.

General-purpose rewards live in ``common/reward_terms.py``; mjlab-style
delegates live in ``newton/mjlab_rewards.py``.
"""

import torch
import warp as wp

from rlworld.rl.envs.mdp.observations.newton.body_utils import (
    get_bodies_height_with_contact,
    get_bodies_pos,
)
from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
    get_leg_xy_signs,
    penalize_contact_force_count,
)
from rlworld.rl.envs.newton.locomotion_env import NewtonLocomotionEnv
from rlworld.rl.envs.utils.newton.body_cache import get_cache
from rlworld.rl.utils.quat_utils import quat_apply_yaw_wxyz, quat_conjugate_wxyz

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

    reward = -(1.0 - desired_contact) * (1.0 - torch.exp(-(foot_forces**2) / gait_force_sigma))
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

    reward = -(desired_contact * (1.0 - torch.exp(-(foot_vel_norm**2) / gait_vel_sigma)))
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
    phases = 1.0 - torch.abs(1.0 - torch.clip((foot_phases * 2.0) - 1.0, 0.0, 1.0) * 2.0)

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
    """Thin redirect to ``common.penalize_contact_force_count``."""
    return penalize_contact_force_count(env, contact_group=contact_group, force_threshold=force_threshold)
