from dataclasses import dataclass

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.newton import reward_terms as rf


@dataclass
class TrackingRewards:
    """
    Newton reward terms for velocity tracking tasks.

    Includes:
    - Linear velocity tracking
    - Angular velocity tracking
    """

    tracking_lin_vel_weight: float = 1.0
    tracking_ang_vel_weight: float = 0.2

    def to_terms(self) -> dict[str, RewardTermConfig]:
        """Convert to dict of RewardTermConfig for config dict."""
        return {
            "tracking_lin_vel": RewardTermConfig(rf.tracking_lin_vel, weight=self.tracking_lin_vel_weight),
            "tracking_ang_vel": RewardTermConfig(rf.tracking_ang_vel, weight=self.tracking_ang_vel_weight),
        }
