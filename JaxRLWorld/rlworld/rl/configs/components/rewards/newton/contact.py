from dataclasses import dataclass

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.newton import reward_terms as rf


@dataclass
class ContactRewards:
    """Contact-related reward terms."""

    feet_height_weight: float | None = 0.05
    feet_height_target: float = 0.2
    feet_air_time_weight: float | None = 1.5
    feet_air_time_threshold: float = 0.5
    feet_slip_weight: float | None = None

    invalid_contact_weight: float = 1.0
    impact_force_weight: float = 0.0001

    feet_links: str | list[str] = ".*_foot"
    contact_allowed_links: str | list[str] = ".*_foot"

    def to_terms(self) -> dict[str, RewardTermConfig]:
        terms = {
            "penalize_invalid_contact": RewardTermConfig(
                rf.penalize_invalid_contact,
                weight=self.invalid_contact_weight,
                params={"allowed_bodies": self.contact_allowed_links},
            ),
            "penalize_impact_force": RewardTermConfig(
                rf.penalize_impact_force,
                weight=self.impact_force_weight,
                params={"feet_bodies": self.feet_links},
            ),
        }

        if self.feet_height_weight is not None:
            terms["reward_feet_height_exp"] = RewardTermConfig(
                rf.reward_feet_height_exp,
                weight=self.feet_height_weight,
                params={"feet_bodies": self.feet_links, "target_height": self.feet_height_target},
            )

        if self.feet_air_time_weight is not None:
            terms["reward_feet_air_time"] = RewardTermConfig(
                rf.reward_feet_air_time,
                weight=self.feet_air_time_weight,
                params={"feet_bodies": self.feet_links, "threshold": self.feet_air_time_threshold},
            )

        if self.feet_slip_weight is not None:
            terms["penalize_feet_slip"] = RewardTermConfig(
                rf.penalize_feet_slip,
                weight=self.feet_slip_weight,
                params={"feet_bodies": self.feet_links}
            )

        return terms
