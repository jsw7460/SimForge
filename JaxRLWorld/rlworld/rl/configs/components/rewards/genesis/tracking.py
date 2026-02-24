from dataclasses import dataclass
from typing import List

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf


@dataclass
class TrackingRewards:
    """
    Reward terms for velocity tracking tasks.

    Includes:
    - Linear velocity tracking
    - Angular velocity tracking
    - Z-axis velocity penalty
    """

    base_name: str = "base"
    tracking_lin_vel_weight: float = 1.0
    tracking_ang_vel_weight: float = 0.2

    def to_terms(self) -> List[RewardTermConfig]:
        """Convert to list of RewardTermConfig for config dict."""
        return [
            RewardTermConfig(rf.tracking_lin_vel, weight=self.tracking_lin_vel_weight),
            RewardTermConfig(rf.tracking_ang_vel, weight=self.tracking_ang_vel_weight, params={"base_name": self.base_name}),
        ]
