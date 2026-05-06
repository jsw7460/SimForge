"""Motion-tracking termination conditions.

Sim-agnostic port of
``Mjlab/src/mjlab/tasks/tracking/mdp/terminations.py``. Each function
returns a :class:`TerminationResult` and reads the
:class:`MotionCommand` from
``env.command_manager.get_term(command_name)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from rlworld.rl.configs.terminations import TerminationResult
from rlworld.rl.envs.mdp.commands.motion import MotionCommand
from rlworld.rl.envs.mdp.rewards.common.motion_tracking import _get_body_indexes
from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs import World


# Gravity vector in world frame. MuJoCo / Newton / Genesis all use
# -Z up-vector by convention; rotated into anchor frames below to
# measure vertical-axis misalignment between robot and motion.
_GRAVITY_W = (0.0, 0.0, -1.0)


def bad_anchor_pos_z_only(
    env: World,
    command_name: str,
    threshold: float,
) -> TerminationResult:
    """Terminate if the robot's anchor Z drifts more than ``threshold`` meters.

    Only the vertical component is checked — horizontal drift is
    tolerated since rewards use yaw-aligned relative frames.
    """
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    err = torch.abs(cmd.anchor_pos_w[:, -1] - cmd.robot_anchor_pos_w[:, -1])
    return TerminationResult(reset=err > threshold, is_timeout=False)


def bad_anchor_ori(
    env: World,
    command_name: str,
    threshold: float,
) -> TerminationResult:
    """Terminate if the anchor's vertical-axis orientation deviates too far.

    Projects world gravity into the motion-anchor frame and into the
    robot-anchor frame, then compares the Z component. A divergence
    above ``threshold`` (in radian-ish units) indicates the robot has
    tilted / capsized relative to the reference.
    """
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    gravity = torch.tensor(
        _GRAVITY_W,
        device=env.device,
        dtype=torch.float32,
    ).expand(env.num_envs, 3)
    motion_gz = quat_rotate_inverse_wxyz(cmd.anchor_quat_w, gravity)[:, 2]
    robot_gz = quat_rotate_inverse_wxyz(cmd.robot_anchor_quat_w, gravity)[:, 2]
    return TerminationResult(
        reset=(motion_gz - robot_gz).abs() > threshold,
        is_timeout=False,
    )


def bad_motion_body_pos_z_only(
    env: World,
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...] | None = None,
) -> TerminationResult:
    """Terminate if any tracked body's Z error exceeds ``threshold``.

    Compares ``body_pos_relative_w`` (yaw-aligned reference) to the
    robot's live per-body world position on the Z axis only. ``body_names``
    filters to a subset (typically end-effectors / feet).
    """
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    idx = _get_body_indexes(cmd, body_names)
    err = torch.abs(cmd.body_pos_relative_w[:, idx, -1] - cmd.robot_body_pos_w[:, idx, -1])
    return TerminationResult(
        reset=torch.any(err > threshold, dim=-1),
        is_timeout=False,
    )
