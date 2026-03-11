from dataclasses import dataclass

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf


@dataclass
class RegularizationRewards:
    """
    Regularization reward terms for smooth and stable behavior.

    Includes:
    - Action rate penalty
    - Similar to default pose
    - Base height tracking
    """

    action_rate_weight: float = 0.005
    similar_to_default_weight: float | None = 0.1
    base_height_weight: float | None = 50.0
    lin_vel_z_weight: float = 1.0

    def to_terms(self) -> dict[str, RewardTermConfig]:
        """Convert to dict of RewardTermConfig keyed by function name."""
        terms = {
            "action_rate": RewardTermConfig(rf.action_rate, weight=self.action_rate_weight),
            "lin_vel_z": RewardTermConfig(rf.lin_vel_z, weight=self.lin_vel_z_weight),
        }

        if self.base_height_weight is not None:
            terms["base_height"] = RewardTermConfig(rf.base_height, weight=self.base_height_weight)

        if self.similar_to_default_weight is not None:
            terms["similar_to_default"] = RewardTermConfig(rf.similar_to_default, weight=self.similar_to_default_weight)

        return terms
