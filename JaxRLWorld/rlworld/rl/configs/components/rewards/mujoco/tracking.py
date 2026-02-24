"""MuJoCo tracking reward components."""
from dataclasses import dataclass
from typing import List
import math

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf


@dataclass
class TrackingRewards:
    """
    MuJoCo reward terms for velocity tracking tasks.

    Includes:
    - Linear velocity tracking (exponential kernel)
    - Angular velocity tracking (exponential kernel)
    """

    tracking_lin_vel_weight: float = 2.0
    tracking_lin_vel_std: float = 0.5  # sqrt(0.25)
    tracking_ang_vel_weight: float = 2.0
    tracking_ang_vel_std: float = 0.707  # sqrt(0.5)

    def to_terms(self) -> List[RewardTermConfig]:
        """Convert to list of RewardTermConfig for config dict."""
        return [
            RewardTermConfig(
                rf.track_linear_velocity,
                weight=self.tracking_lin_vel_weight,
                params={"std": self.tracking_lin_vel_std},
            ),
            RewardTermConfig(
                rf.track_angular_velocity,
                weight=self.tracking_ang_vel_weight,
                params={"std": self.tracking_ang_vel_std},
            ),
        ]
