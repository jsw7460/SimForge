from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.terminations import TerminationResult
from rlworld.rl.envs.mdp.observations.newton.state import base_euler

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


def roll_pitch_violation(
    env: "NewtonEnv", roll_threshold_degree: float = 15.0, pitch_threshold_degree: float = 15.0
) -> TerminationResult:
    """Check if robot's roll or pitch exceeds safe thresholds."""
    euler = base_euler(env, rpy=True, degrees=True)
    roll = euler[:, 0]
    pitch = euler[:, 1]

    roll_violated = torch.abs(roll) > roll_threshold_degree
    pitch_violated = torch.abs(pitch) > pitch_threshold_degree
    return TerminationResult(roll_violated | pitch_violated)
