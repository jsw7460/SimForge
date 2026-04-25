"""Motion-tracking observation terms.

Sim-agnostic port of
``Mjlab/src/mjlab/tasks/tracking/mdp/observations.py``. Each function
reads the :class:`MotionCommand` from
``env.command_manager.get_term(command_name)`` and returns a flattened
per-env tensor. Orientation is encoded as the first two columns of the
rotation matrix (continuous 6D representation).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from rlworld.rl.envs.mdp.commands.motion import MotionCommand
from rlworld.rl.utils.quat_utils import (
    matrix_from_quat_wxyz,
    subtract_frame_transforms_wxyz,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def motion_anchor_pos_b(env: "World", command_name: str) -> torch.Tensor:
    """Motion anchor position expressed in the robot's anchor frame.

    Returns shape ``(num_envs, 3)``.
    """
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    pos, _ = subtract_frame_transforms_wxyz(
        cmd.robot_anchor_pos_w, cmd.robot_anchor_quat_w,
        cmd.anchor_pos_w, cmd.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: "World", command_name: str) -> torch.Tensor:
    """Motion anchor orientation (relative to robot anchor), 6D rep.

    Returns shape ``(num_envs, 6)``.
    """
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    _, ori = subtract_frame_transforms_wxyz(
        cmd.robot_anchor_pos_w, cmd.robot_anchor_quat_w,
        cmd.anchor_pos_w, cmd.anchor_quat_w,
    )
    mat = matrix_from_quat_wxyz(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: "World", command_name: str) -> torch.Tensor:
    """Live robot body positions expressed in the robot anchor frame.

    Returns shape ``(num_envs, num_tracked_bodies * 3)``.
    """
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    num_bodies = len(cmd.cfg.body_names)
    pos_b, _ = subtract_frame_transforms_wxyz(
        cmd.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, 3),
        cmd.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, 4),
        cmd.robot_body_pos_w,
        cmd.robot_body_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: "World", command_name: str) -> torch.Tensor:
    """Live robot body orientations relative to the robot anchor (6D rep).

    Returns shape ``(num_envs, num_tracked_bodies * 6)``.
    """
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    num_bodies = len(cmd.cfg.body_names)
    _, ori_b = subtract_frame_transforms_wxyz(
        cmd.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, 3),
        cmd.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, 4),
        cmd.robot_body_pos_w,
        cmd.robot_body_quat_w,
    )
    mat = matrix_from_quat_wxyz(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_future_reference_window(
    env: "World", command_name: str,
) -> torch.Tensor:
    """Future motion reference window, flattened for the obs pipeline.

    Returns shape ``(num_envs, T * B * 9)`` where ``T`` is the length of
    ``MotionCommandCfg.future_offsets``, ``B`` is the tracked-body count,
    and 9 = rel_pos(3) + rel_quat_6d(6) per (t, b) token. The
    ``SpaceTimeTransformer`` actor reshapes this back into the
    ``(T, B, 9)`` grid it expects. Returns an empty ``(num_envs, 0)``
    tensor when ``future_offsets`` is empty.
    """
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    features = cmd.future_body_features_in_anchor_frame()
    return features.reshape(env.num_envs, -1)
