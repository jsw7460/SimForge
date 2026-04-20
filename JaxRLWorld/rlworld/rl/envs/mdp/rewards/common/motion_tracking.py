"""Motion-tracking reward terms.

Sim-agnostic port of ``Mjlab/src/mjlab/tasks/tracking/mdp/rewards.py``.
Each function reads the :class:`MotionCommand` from
``env.command_manager.get_term(command_name)`` and computes an
exponential error reward ``exp(-err² / std²)`` over anchor or body
pose / velocity errors.

All functions have the signature
``func(env, command_name: str, std: float, body_names=None)``
matching the JaxRLWorld reward-manager dispatch convention
(``func(env, **params)``) with per-term params supplied by
``RewardTermConfig.params``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from rlworld.rl.envs.mdp.commands.motion import MotionCommand
from rlworld.rl.utils.quat_utils import quat_error_magnitude_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def _get_body_indexes(
    command: MotionCommand, body_names: "tuple[str, ...] | None",
) -> list[int]:
    """Indices into ``command.cfg.body_names`` for the subset ``body_names``.

    Returns all indices when ``body_names`` is ``None``.
    """
    return [
        i
        for i, name in enumerate(command.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: "World", command_name: str, std: float,
) -> torch.Tensor:
    """Anchor body world-frame position tracking: ``exp(-||Δpos||² / σ²)``."""
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    err = torch.sum(
        torch.square(cmd.anchor_pos_w - cmd.robot_anchor_pos_w), dim=-1,
    )
    return torch.exp(-err / std ** 2)


def motion_global_anchor_orientation_error_exp(
    env: "World", command_name: str, std: float,
) -> torch.Tensor:
    """Anchor body orientation tracking: ``exp(-angle_err² / σ²)``."""
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    err = quat_error_magnitude_wxyz(cmd.anchor_quat_w, cmd.robot_anchor_quat_w) ** 2
    return torch.exp(-err / std ** 2)


def motion_relative_body_position_error_exp(
    env: "World",
    command_name: str,
    std: float,
    body_names: "tuple[str, ...] | None" = None,
) -> torch.Tensor:
    """Per-body anchor-relative position tracking (average over bodies)."""
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    idx = _get_body_indexes(cmd, body_names)
    err = torch.sum(
        torch.square(
            cmd.body_pos_relative_w[:, idx] - cmd.robot_body_pos_w[:, idx]
        ),
        dim=-1,
    )
    return torch.exp(-err.mean(-1) / std ** 2)


def motion_relative_body_orientation_error_exp(
    env: "World",
    command_name: str,
    std: float,
    body_names: "tuple[str, ...] | None" = None,
) -> torch.Tensor:
    """Per-body anchor-relative orientation tracking (average over bodies)."""
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    idx = _get_body_indexes(cmd, body_names)
    err = (
        quat_error_magnitude_wxyz(
            cmd.body_quat_relative_w[:, idx], cmd.robot_body_quat_w[:, idx],
        )
        ** 2
    )
    return torch.exp(-err.mean(-1) / std ** 2)


def motion_global_body_linear_velocity_error_exp(
    env: "World",
    command_name: str,
    std: float,
    body_names: "tuple[str, ...] | None" = None,
) -> torch.Tensor:
    """Per-body world-frame linear velocity tracking (average over bodies)."""
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    idx = _get_body_indexes(cmd, body_names)
    err = torch.sum(
        torch.square(
            cmd.body_lin_vel_w[:, idx] - cmd.robot_body_lin_vel_w[:, idx]
        ),
        dim=-1,
    )
    return torch.exp(-err.mean(-1) / std ** 2)


def motion_global_body_angular_velocity_error_exp(
    env: "World",
    command_name: str,
    std: float,
    body_names: "tuple[str, ...] | None" = None,
) -> torch.Tensor:
    """Per-body world-frame angular velocity tracking (average over bodies)."""
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    idx = _get_body_indexes(cmd, body_names)
    err = torch.sum(
        torch.square(
            cmd.body_ang_vel_w[:, idx] - cmd.robot_body_ang_vel_w[:, idx]
        ),
        dim=-1,
    )
    return torch.exp(-err.mean(-1) / std ** 2)
