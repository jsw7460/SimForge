from dataclasses import dataclass

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf


@dataclass
class ContactRewards:
    """Contact-related reward terms."""

    feet_height_weight: float | None = 0.05
    feet_height_target: float = 0.2
    feet_air_time_weight: float | None = 1.5
    feet_air_time_threshold: float = 0.5
    invalid_contact_weight: float = 1.0
    impact_force_weight: float = 0.0001

    feet_links: str | list[str] = ".*_foot"
    contact_allowed_links: str | list[str] = ".*_foot"

    def to_terms(self) -> list[RewardTermConfig]:
        terms = [
            RewardTermConfig(
                rf.penalize_invalid_contact,
                weight=self.invalid_contact_weight,
                params={"contact_allowed_links": self.contact_allowed_links},
            ),
            RewardTermConfig(
                rf.penalize_impact_force,
                weight=self.impact_force_weight,
                params={"links": self.feet_links},
            ),
        ]

        if self.feet_height_weight is not None:
            terms.append(
                RewardTermConfig(
                    rf.reward_feet_height_exp,
                    weight=self.feet_height_weight,
                    params={"feet_links": self.feet_links, "target_height": self.feet_height_target},
                )
            )

        if self.feet_air_time_weight is not None:
            terms.append(
                RewardTermConfig(
                    rf.reward_feet_air_time,
                    weight=self.feet_air_time_weight,
                    params={"links": self.feet_links, "threshold": self.feet_air_time_threshold},
                )
            )

        return terms
