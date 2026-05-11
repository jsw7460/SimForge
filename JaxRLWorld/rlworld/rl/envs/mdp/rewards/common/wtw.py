"""Walk-These-Ways (WTW) / gait-conditioned reward terms (sim-agnostic).

These terms depend on WTW-specific command channels (``body_pitch``,
``body_roll``, ``body_height``) and/or the processed/raw action history
maintained by :class:`act_manager`. They are split out of
``reward_terms.py`` because they only make sense for gait-conditioned
locomotion presets — other tasks (flat locomotion, getup, …) never
register these command terms, so importing them here keeps the generic
reward module lean.

Exposed symbols:
  - :func:`penalize_action_smoothness_1`
  - :func:`penalize_action_smoothness_2`
  - :func:`penalize_orientation_control`
  - :func:`reward_body_height_cmd`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.utils.quat_utils import (
    quat_from_angle_axis_wxyz,
    quat_mul_wxyz,
    quat_rotate_inverse_wxyz,
)

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


def penalize_action_smoothness_1(env: World) -> torch.Tensor:
    """Penalize 1st-order action changes (processed). WTW: _reward_action_smoothness_1.

    Uses processed_action_history (joint position targets) and masks
    the first step where raw actions are still zero.
    """
    hist = env.act_manager.processed_action_history
    diff = torch.square(hist[0] - hist[1])
    mask = env.act_manager.raw_action_history[1] != 0
    return -torch.sum(diff * mask, dim=1)


def penalize_action_smoothness_2(env: World) -> torch.Tensor:
    """Penalize 2nd-order action changes (processed). WTW: _reward_action_smoothness_2.

    Second-order finite difference of joint position targets, masked
    for the first two steps.
    """
    hist = env.act_manager.processed_action_history
    diff = torch.square(hist[0] - 2.0 * hist[1] + hist[2])
    mask1 = env.act_manager.raw_action_history[1] != 0
    mask2 = env.act_manager.raw_action_history[2] != 0
    return -torch.sum(diff * mask1 * mask2, dim=1)


def penalize_orientation_control(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """Penalize deviation from commanded body orientation. WTW: _reward_orientation_control.

    Constructs desired body quaternion from body_pitch and body_roll commands,
    computes desired projected gravity, and penalizes xy-deviation from actual.
    """
    body_pitch = env.command_manager.body_pitch
    body_roll = env.command_manager.body_roll
    device = body_pitch.device

    # WTW: quat_roll = quat_from_angle_axis(-body_roll, [1,0,0])
    #      quat_pitch = quat_from_angle_axis(-body_pitch, [0,1,0])
    #      desired = quat_mul(quat_roll, quat_pitch)
    axis_x = torch.tensor([1.0, 0.0, 0.0], device=device)
    axis_y = torch.tensor([0.0, 1.0, 0.0], device=device)
    quat_roll = quat_from_angle_axis_wxyz(-body_roll, axis_x)
    quat_pitch = quat_from_angle_axis_wxyz(-body_pitch, axis_y)
    desired_quat = quat_mul_wxyz(quat_roll, quat_pitch)

    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=device).expand(len(body_pitch), -1)
    desired_gravity = quat_rotate_inverse_wxyz(desired_quat, gravity_vec)

    actual_gravity = env.get_robot_data(asset_cfg.name).projected_gravity_b
    return -torch.sum(torch.square(actual_gravity[:, :2] - desired_gravity[:, :2]), dim=1)


def reward_body_height_cmd(
    env: World,
    base_height_target: float = 0.30,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> torch.Tensor:
    """Reward for tracking commanded body height. WTW: _reward_jump.

    Target height = body_height command + base_height_target.
    """
    body_height = env.get_robot_data(asset_cfg.name).root_link_pos_w[:, 2]
    target = env.command_manager.body_height + base_height_target
    return -torch.square(body_height - target)
