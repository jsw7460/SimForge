from dataclasses import dataclass

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf


@dataclass
class PostureRewards:
    """Posture-related penalty terms."""

    ang_vel_xy_weight: float = 0.025
    torques_weight: float = 1e-5
    hip_deviation_weight: float = 0.2
    nonflat_gravity_weight: float = 0.2

    hip_joints: str | list[str] = ".*_hip_joint"

    def to_terms(self) -> list[RewardTermConfig]:
        return [
            RewardTermConfig(rf.penalize_ang_vel_xy, weight=self.ang_vel_xy_weight),
            RewardTermConfig(rf.penalize_torques, weight=self.torques_weight),
            RewardTermConfig(rf.penalize_hip_deviation, weight=self.hip_deviation_weight,
                             params={"hip_joints": self.hip_joints}),
            RewardTermConfig(rf.penalize_nonflat_by_gravity, weight=self.nonflat_gravity_weight),
        ]
