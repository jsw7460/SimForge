"""Unified termination conditions using the RobotData interface.

All functions accept any ``World`` subclass and read state exclusively
through ``env.get_robot_data(entity_name)``, making them simulator-agnostic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.mdp.configs.terminations import TerminationResult
from rlworld.rl.utils.quat_utils import quat_to_euler_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


def roll_pitch_violation(
    env: World,
    roll_threshold_degree: float = 15.0,
    pitch_threshold_degree: float = 15.0,
    entity_name: str = "robot",
) -> TerminationResult:
    """Terminate if robot's roll or pitch exceeds safe thresholds.

    Args:
        env: Any environment with ``get_robot_data``.
        roll_threshold_degree: Maximum allowed roll angle in degrees.
        pitch_threshold_degree: Maximum allowed pitch angle in degrees.
        entity_name: Name of the entity to check.

    Returns:
        TerminationResult indicating which envs should reset.
    """
    quat_wxyz = env.get_robot_data(entity_name).root_link_quat_w
    euler = quat_to_euler_wxyz(quat_wxyz)  # (num_envs, 3) radians

    roll_deg = torch.abs(euler[:, 0]) * (180.0 / torch.pi)
    pitch_deg = torch.abs(euler[:, 1]) * (180.0 / torch.pi)

    violated = (roll_deg > roll_threshold_degree) | (pitch_deg > pitch_threshold_degree)
    return TerminationResult(violated)
