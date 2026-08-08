"""K1 joystick with the robot_lab Booster-T1 FLAT training recipe (all three
backends).

Same robot, scene, OBSERVATIONS (75-D actor, no gait phase — identical to the
G1 recipe so the deploy code is unchanged), commands, events and DR as
:class:`K1JoystickConfig`. What changes is the training package ported from
``robot_lab`` (fan-ziqi) ``booster_t1`` flat env:

- rewards: the T1-flat set (yaw-frame velocity tracking with uprightness
  gating, upward, feet_air_time_positive_biped, feet_slide, joint/action
  penalties, is_terminated), weights verbatim from ``rough_env_cfg`` + the flat
  override (``lin_vel_z_l2 = -0.2``). Adapted to K1: the T1 ``Waist``
  joint_deviation term is dropped (K1 has no waist). Every reward function is
  sim-agnostic (``k1_locomotion`` / ``common``), so unlike the G1 recipe this
  needs no per-backend dispatch.
- action package: plain Gaussian policy, UNIFORM action scale 0.25 (robot_lab
  ``actions.joint_pos.scale``), wide clip.
- PPO: robot_lab's rsl_rl hyperparameters (entropy 0.008, lr 1e-3 adaptive via
  desired_kl 0.01, 24 steps/env, 5 epochs, 4 minibatches).

Sign convention: rewards whose FUNCTION already returns a negative penalty
(``penalize_lin_vel_z`` / ``raw_action_rate_l2`` / ``penalize_joint_pos_limits_l1``)
take a POSITIVE weight; ``is_terminated`` returns +1 so it takes the negative
weight. All ``k1_locomotion`` ports return the natural (positive-penalty /
positive-reward) sign, so their robot_lab weights transfer directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import RewardConfig
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.scene import SceneEntitySelector
from rlworld.rl.envs.mdp.rewards import k1_locomotion as k1_rf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common

from .base import K1JoystickConfig


@dataclass
class K1T1RecipeConfig(K1JoystickConfig):
    """K1 trained the way robot_lab booster_t1 flat is trained."""

    sim_type: str = "newton"
    action_distribution: str = "gaussian"
    action_scale: Any = 0.25  # robot_lab uniform actions.joint_pos.scale
    action_clip: tuple = (-100.0, 100.0)
    run_name: str | None = None
    _RUN_NAMES = {
        "newton": "K1_Newton_T1Recipe",
        "mujoco": "K1_Mujoco_T1Recipe",
        "genesis": "K1_Genesis_T1Recipe",
    }

    def _build_runner_config(self):
        runner = super()._build_runner_config()
        if self.run_name is None:
            runner.run_name = self._RUN_NAMES[self.sim_type]
        return runner

    def _uses_gait_phase(self) -> bool:
        """T1 has no gait-phase clock (rhythm comes from
        feet_air_time_positive_biped), so drop the 4-D phase block — 75-D
        actor, identical obs to the G1 recipe (deploy unchanged)."""
        return False

    def _build_algorithm_config(self) -> PPOConfig:
        ppo = super()._build_algorithm_config()
        # robot_lab rsl_rl_ppo_cfg (flat): the RL_lab humanoid recipe.
        ppo.entropy_coef = 0.01
        ppo.gamma = 0.99
        ppo.lam = 0.95
        ppo.desired_kl = 0.01
        ppo.schedule = "adaptive"
        ppo.num_learning_epochs = 5
        ppo.num_mini_batches = 4
        ppo.num_steps_per_env = 24
        ppo.actor_lr = 1.0e-3
        ppo.critic_lr = 1.0e-3
        return ppo

    def _build_reward_config(self) -> RewardConfig:
        r = self.robot
        feet_sel = SceneEntitySelector(name="robot", body_names=tuple(r.foot_names), preserve_order=True)
        legs_sel = SceneEntitySelector(name="robot", joint_names=(r".*_Hip_.*", r".*_Knee_.*", r".*_Ankle_.*"))
        hipknee_sel = SceneEntitySelector(name="robot", joint_names=(r".*_Hip_.*", r".*_Knee_.*"))
        hip_dev_sel = SceneEntitySelector(name="robot", joint_names=(r".*_Hip_Yaw", r".*_Hip_Roll"))
        arms_sel = SceneEntitySelector(name="robot", joint_names=(r".*_Shoulder_.*", r".*_Elbow_.*"))
        all_joints = SceneEntitySelector(name="robot", joint_names=(r".*",))

        @dataclass
        class _RewardsCfg(RewardConfig):
            # Velocity-tracking (yaw-frame / world-z, uprightness-gated).
            track_lin_vel = RewardTermConfig(func=k1_rf.track_lin_vel_xy_yaw_frame_exp, weight=4.5, params={"std": 0.5})
            track_ang_vel = RewardTermConfig(func=k1_rf.track_ang_vel_z_world_exp, weight=2.5, params={"std": 0.5})
            # Root penalties (positive-penalty funcs → negative weight; the
            # ``penalize_*`` funcs already return negative → positive weight).
            flat_orientation = RewardTermConfig(func=k1_rf.orientation_l2, weight=-0.2)
            ang_vel_xy = RewardTermConfig(func=k1_rf.ang_vel_xy_l2, weight=-0.1)
            lin_vel_z = RewardTermConfig(func=rf_common.penalize_lin_vel_z, weight=0.2)
            # Joint penalties.
            joint_torques = RewardTermConfig(
                func=k1_rf.joint_torques_l2, weight=-3.0e-7, params={"asset_cfg": legs_sel}
            )
            joint_acc = RewardTermConfig(func=k1_rf.K1JointAccL2, weight=-1.25e-7, params={"asset_cfg": hipknee_sel})
            joint_deviation_hip = RewardTermConfig(
                func=k1_rf.joint_deviation_l1, weight=-0.01, params={"asset_cfg": hip_dev_sel}
            )
            joint_deviation_arms = RewardTermConfig(
                func=k1_rf.joint_deviation_l1, weight=-0.05, params={"asset_cfg": arms_sel}
            )
            joint_pos_limits = RewardTermConfig(func=rf_common.penalize_joint_pos_limits_l1, weight=1.0)
            joint_pos_penalty = RewardTermConfig(
                func=k1_rf.joint_pos_penalty, weight=-1.0, params={"asset_cfg": all_joints}
            )
            # Action penalty (func returns negative → positive weight).
            action_rate = RewardTermConfig(func=rf_common.raw_action_rate_l2, weight=0.075)
            # Feet.
            feet_air_time = RewardTermConfig(
                func=k1_rf.K1FeetAirTimePositiveBiped,
                weight=2.0,
                params={"contact_group": "feet_ground_contact", "threshold": 0.4},
            )
            feet_slide = RewardTermConfig(
                func=k1_rf.feet_slide,
                weight=-0.4,
                params={"contact_group": "feet_ground_contact", "asset_cfg": feet_sel},
            )
            # Others.
            upward = RewardTermConfig(func=k1_rf.upward, weight=1.0)
            is_terminated = RewardTermConfig(func=rf_common.is_terminated, weight=-200.0)

        return _RewardsCfg()
