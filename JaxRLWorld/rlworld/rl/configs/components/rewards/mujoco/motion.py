"""MuJoCo motion/posture reward components."""
from dataclasses import dataclass
import math

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf


@dataclass
class PostureRewards:
    """
    MuJoCo reward terms for posture and motion quality.

    Includes:
    - Flat orientation reward
    - Posture (default pose) reward
    - Body angular velocity penalty
    - Vertical velocity penalty
    """

    # Flat orientation
    flat_orientation_weight: float | None = 1.0
    flat_orientation_std: float = 0.447  # sqrt(0.2)

    # Posture
    posture_weight: float | None = 1.0
    posture_std: float = 0.25

    # Body angular velocity penalty
    body_ang_vel_weight: float | None = 0.0

    # Vertical velocity penalty
    lin_vel_z_weight: float | None = None

    def to_terms(self) -> dict[str, RewardTermConfig]:
        """Convert to dict of RewardTermConfig for config dict."""
        terms = {}

        if self.flat_orientation_weight is not None:
            terms["flat_orientation"] = RewardTermConfig(
                rf.flat_orientation,
                weight=self.flat_orientation_weight,
                params={"std": self.flat_orientation_std},
            )

        if self.posture_weight is not None:
            terms["posture"] = RewardTermConfig(
                rf.posture,
                weight=self.posture_weight,
                params={"std": self.posture_std},
            )

        if self.body_ang_vel_weight is not None:
            terms["body_angular_velocity_penalty"] = RewardTermConfig(
                rf.body_angular_velocity_penalty,
                weight=self.body_ang_vel_weight,
            )

        if self.lin_vel_z_weight is not None:
            terms["lin_vel_z_penalty"] = RewardTermConfig(
                rf.lin_vel_z_penalty,
                weight=self.lin_vel_z_weight,
            )

        return terms
