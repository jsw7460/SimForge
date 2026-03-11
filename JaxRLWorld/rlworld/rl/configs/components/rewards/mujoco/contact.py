"""MuJoCo contact reward components."""
from dataclasses import dataclass

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf


@dataclass
class ContactRewards:
    """
    MuJoCo reward terms for contact/feet-related behaviors.

    Includes:
    - Feet air time reward
    - Feet clearance penalty
    - Feet slip penalty
    - Soft landing penalty
    """

    # Feet air time
    feet_air_time_weight: float | None = 0.0
    air_time_threshold_min: float = 0.05
    air_time_threshold_max: float = 0.5
    air_time_command_threshold: float = 0.5

    # Feet clearance
    feet_clearance_weight: float | None = -2.0
    clearance_target_height: float = 0.1
    clearance_command_threshold: float = 0.05

    # Feet slip
    feet_slip_weight: float | None = -0.1
    slip_command_threshold: float = 0.05

    # Soft landing
    soft_landing_weight: float | None = -1e-5
    landing_command_threshold: float = 0.05

    def to_terms(self) -> dict[str, RewardTermConfig]:
        """Convert to dict of RewardTermConfig for config dict."""
        terms = {}

        if self.feet_air_time_weight is not None:
            terms["feet_air_time"] = RewardTermConfig(
                rf.feet_air_time,
                weight=self.feet_air_time_weight,
                params={
                    "threshold_min": self.air_time_threshold_min,
                    "threshold_max": self.air_time_threshold_max,
                    "command_threshold": self.air_time_command_threshold,
                },
            )

        if self.feet_clearance_weight is not None:
            terms["feet_clearance"] = RewardTermConfig(
                rf.feet_clearance,
                weight=self.feet_clearance_weight,
                params={
                    "target_height": self.clearance_target_height,
                    "command_threshold": self.clearance_command_threshold,
                },
            )

        if self.feet_slip_weight is not None:
            terms["feet_slip"] = RewardTermConfig(
                rf.feet_slip,
                weight=self.feet_slip_weight,
                params={
                    "command_threshold": self.slip_command_threshold,
                },
            )

        if self.soft_landing_weight is not None:
            terms["soft_landing"] = RewardTermConfig(
                rf.soft_landing,
                weight=self.soft_landing_weight,
                params={
                    "command_threshold": self.landing_command_threshold,
                },
            )

        return terms
