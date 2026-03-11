"""MuJoCo regularization reward components."""
from dataclasses import dataclass

from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf


@dataclass
class RegularizationRewards:
    """
    MuJoCo reward terms for action/motion regularization.

    Includes:
    - Action rate penalty
    - Joint position limits penalty
    - Joint torques penalty
    - Joint velocity penalty
    """

    action_rate_weight: float | None = -0.1
    joint_pos_limits_weight: float | None = -1.0
    joint_torques_weight: float | None = None
    joint_vel_weight: float | None = None

    def to_terms(self) -> dict[str, RewardTermConfig]:
        """Convert to dict of RewardTermConfig for config dict."""
        terms = {}

        if self.action_rate_weight is not None:
            terms["action_rate_l2"] = RewardTermConfig(
                rf.action_rate_l2,
                weight=self.action_rate_weight,
            )

        if self.joint_pos_limits_weight is not None:
            terms["joint_pos_limits"] = RewardTermConfig(
                rf.joint_pos_limits,
                weight=self.joint_pos_limits_weight,
            )

        if self.joint_torques_weight is not None:
            terms["joint_torques_l2"] = RewardTermConfig(
                rf.joint_torques_l2,
                weight=self.joint_torques_weight,
            )

        if self.joint_vel_weight is not None:
            terms["joint_vel_l2"] = RewardTermConfig(
                rf.joint_vel_l2,
                weight=self.joint_vel_weight,
            )

        return terms
