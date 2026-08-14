"""K1 joystick with the G1-flat training recipe (all three backends).

Same robot, scene, observations (82-D actor incl. the phase clock —
unused by these rewards but kept so the two recipes differ ONLY in the
training package), commands, events, DR, and terminations as
:class:`K1JoystickConfig`. What changes is the coherent G1-flat package:

- rewards: the g1_29dof flat set (tracking + variable_posture +
  clearance/swing/slip/soft-landing + explicit smoothness penalties
  ``raw_action_rate_l2`` / body-ang-vel / angular-momentum), adapted to
  K1 joint/body names. ``self_collision_cost`` is dropped — the K1
  collision model is feet-only (non-foot geoms have contype 0, so no
  self-contact signal exists); the foot-pair contact penalty stands in.
  No ``total_clip`` (the G1 recipe does not floor the summed reward).
- action package: plain Gaussian policy, per-joint action scale
  ``0.25 * effort / kp`` (the small scale plays the role the tanh bound
  plays in the pal recipe), wide clip.

Switching recipes = running the other entry script; checkpoints carry
the preset class, so eval (incl. cross-sim) reconstructs the right one
automatically. The reward term FUNCTIONS are per-backend modules
(dispatched below, mirroring how the g1_29dof preset builds them per
sim); the weights/params are identical across backends. On mujoco the
angular-momentum penalty reads the "robot/root_angmom" sensor that
K1SpecFn injects into the spec.

Note: K1's foot reference point (``left_foot_link`` origin) sits
~0.038 m above the sole, so ``target_height`` overstates the actual
sole clearance by that offset (e.g. 0.15 -> ~0.11 m sole clearance).
Adjust if swings come out too flat or too high.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from rlworld.rl.configs.common_config_classes import RewardConfig
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.scene import SceneEntitySelector
from rlworld.rl.envs.mdp.rewards import k1_locomotion as k1_rf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common

from .base import K1JoystickConfig


@dataclass
class K1G1RecipeConfig(K1JoystickConfig):
    """K1 trained the way g1_29dof flat is trained."""

    sim_type: str = "newton"
    action_distribution: str = "gaussian"
    # Domain randomization on a 10 s global interval instead of every reset:
    # cuts the per-reset recompute cost (reset_dr -> interval_dr) while still
    # varying DR over time. See _build_dr_terms / k1_interval_dr_prototype_diag.
    dr_interval_period_s: float | None = 10.0
    # Placeholder: the sim builders override action_scale with the robot's
    # physical_action_scale (0.25·effort/kp), so this value is never used.
    action_scale: Any = 1.0
    action_clip: tuple = (-100.0, 100.0)
    run_name: str | None = None

    # The g1-recipe enables the left/right mirror-symmetry loss by default.
    mirror_symmetry_coeff: float = 1.0

    # Standing-only capture-point (XCoM) support-margin penalty. Weight <= 0;
    # set to 0.0 to disable. sigma/margin are the penalty decay length and the
    # foot-size fattening of the two-feet support segment (see
    # k1_locomotion.capture_point_support_penalty). Validated inputs:
    # k1_capture_point_inputs_diag.
    capture_point_weight: float = -1.0
    capture_point_sigma: float = 0.10
    capture_point_margin: float = 0.10
    capture_point_command_threshold: float = 0.1
    _RUN_NAMES = {
        "newton": "K1_Newton_G1Recipe",
        "mujoco": "K1_Mujoco_G1Recipe",
        "genesis": "K1_Genesis_G1Recipe",
    }

    def _build_runner_config(self):
        runner = super()._build_runner_config()
        if self.run_name is None:
            runner.run_name = self._RUN_NAMES[self.sim_type]
        return runner

    # Shared K1 posture stds (G1's values mapped onto K1 joint groups;
    # the head std is chosen — G1 has no head).
    _STD_WALKING = {
        r".*AAHead_yaw": 0.1,
        r".*Head_pitch": 0.1,
        r".*_Shoulder_Pitch": 0.4,  # loose: allow front-back arm swing while walking
        r".*_Shoulder_Roll": 0.15,
        r".*_Elbow_Pitch": 0.15,
        r".*_Elbow_Yaw": 0.15,
        r".*_Hip_Pitch": 0.3,
        r".*_Hip_Roll": 0.15,
        r".*_Hip_Yaw": 0.15,
        r".*_Knee_Pitch": 0.35,
        r".*_Ankle_Pitch": 0.25,
        r".*_Ankle_Roll": 0.1,
    }
    _STD_RUNNING = {
        r".*AAHead_yaw": 0.2,
        r".*Head_pitch": 0.2,
        r".*_Shoulder_Pitch": 0.5,
        r".*_Shoulder_Roll": 0.2,
        r".*_Elbow_Pitch": 0.35,
        r".*_Elbow_Yaw": 0.35,
        r".*_Hip_Pitch": 0.5,
        r".*_Hip_Roll": 0.2,
        r".*_Hip_Yaw": 0.2,
        r".*_Knee_Pitch": 0.6,
        r".*_Ankle_Pitch": 0.35,
        r".*_Ankle_Roll": 0.15,
    }

    def _uses_gait_phase(self) -> bool:
        """No G1-recipe reward reads the gait phase (walking rhythm comes from
        feet_clearance / feet_swing_height), so drop the 4-D phase obs block
        and its command term — 75-D actor, no deploy-side gait clock."""
        return False

    def _build_reward_config(self) -> RewardConfig:
        if self.sim_type == "mujoco":
            return self._build_reward_mujoco()
        return self._build_reward_mjlab_style()

    def _build_reward_mjlab_style(self) -> RewardConfig:
        """Newton and Genesis share the *_mjlab term-function names."""
        rf = importlib.import_module(f"rlworld.rl.envs.mdp.rewards.{self.sim_type}.mjlab_rewards")
        r = self.robot
        feet_selector = SceneEntitySelector(name="robot", body_names=tuple(r.foot_names), preserve_order=True)
        feet_contact_order = list(r.foot_names)
        posture_params = {
            "std_standing": {".*": 0.05},
            "std_walking": dict(self._STD_WALKING),
            "std_running": dict(self._STD_RUNNING),
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
        }

        @dataclass
        class _RewardsCfg(RewardConfig):
            track_lin_vel = RewardTermConfig(
                func=rf_common.track_lin_vel,
                weight=2.0,
                params={"std": 0.5, "penalize_z": True},
            )
            track_ang_vel = RewardTermConfig(
                func=rf_common.track_ang_vel,
                weight=1.0,
                params={"std": 0.5, "penalize_xy": True},
            )
            flat_orientation = RewardTermConfig(func=rf_common.flat_orientation, weight=1.0, params={"std": 0.25})
            capture_point_support = RewardTermConfig(
                func=k1_rf.capture_point_support_penalty,
                weight=self.capture_point_weight,
                params={
                    "asset_cfg": feet_selector,
                    "sigma": self.capture_point_sigma,
                    "margin": self.capture_point_margin,
                    "command_threshold": self.capture_point_command_threshold,
                },
            )
            variable_posture = RewardTermConfig(func=rf.variable_posture, weight=1.0, params=posture_params)
            # G1's self_collision_cost has no signal on the feet-only K1
            # collision model; the foot-pair contact penalty stands in.
            feet_pair_collision = RewardTermConfig(
                func=k1_rf.contact_pair_penalty,
                weight=-1.0,
                params={"contact_group": "feet_pair_contact"},
            )
            body_angular_velocity_penalty = RewardTermConfig(
                func=rf.body_ang_vel_penalty_mjlab,
                weight=0.025,
                params={"asset_cfg": SceneEntitySelector(name="robot", body_names=(r.trunk_body_name,))},
            )
            angular_momentum_penalty = RewardTermConfig(func=rf.angular_momentum_penalty, weight=0.01)
            joint_pos_limits = RewardTermConfig(func=rf.joint_pos_limits_mjlab, weight=1.0)
            raw_action_rate_l2 = RewardTermConfig(func=rf.raw_action_rate_l2_mjlab, weight=0.1)
            feet_clearance = RewardTermConfig(
                func=rf.feet_clearance_mjlab,
                weight=2.0,
                params={
                    "asset_cfg": feet_selector,
                    "target_height": 0.18,
                    "command_threshold": 0.05,
                },
            )
            feet_swing_height = RewardTermConfig(
                func=rf.feet_swing_height_mjlab,
                weight=0.75,
                params={
                    "asset_cfg": feet_selector,
                    "target_height": 0.18,
                    "command_threshold": 0.05,
                    "contact_order": feet_contact_order,
                },
            )
            feet_slip = RewardTermConfig(
                func=rf.feet_slip_mjlab,
                weight=0.1,
                params={
                    "asset_cfg": feet_selector,
                    "command_threshold": 0.05,
                    "contact_order": feet_contact_order,
                },
            )
            # Signature differs per backend (newton: feet_bodies;
            # genesis: contact_group) — same computation underneath.
            soft_landing = RewardTermConfig(
                func=rf.soft_landing_mjlab,
                weight=1e-5,
                params=(
                    {"feet_bodies": list(r.foot_names), "command_threshold": 0.05}
                    if self.sim_type == "newton"
                    else {
                        "contact_group": "feet_ground_contact",
                        "command_threshold": 0.05,
                    }
                ),
            )

        # No total_clip: the G1 recipe does not floor the summed reward.
        return _RewardsCfg()

    def _build_reward_mujoco(self) -> RewardConfig:
        """mjlab backend: same set via rewards.mujoco.reward_terms."""
        rf = importlib.import_module("rlworld.rl.envs.mdp.rewards.mujoco.reward_terms")
        r = self.robot
        feet_selector = SceneEntitySelector(name="robot", body_names=tuple(r.foot_names), preserve_order=True)
        feet_contact_order = list(r.foot_names)
        posture_params = {
            "asset_cfg": SceneEntitySelector(name="robot", joint_names=(".*",)),
            "std_standing": {".*": 0.05},
            "std_walking": dict(self._STD_WALKING),
            "std_running": dict(self._STD_RUNNING),
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
        }

        @dataclass
        class _RewardsCfg(RewardConfig):
            track_lin_vel = RewardTermConfig(
                func=rf_common.track_lin_vel,
                weight=2.0,
                params={"std": 0.5, "penalize_z": True},
            )
            track_ang_vel = RewardTermConfig(
                func=rf_common.track_ang_vel,
                weight=1.0,
                params={"std": 0.5, "penalize_xy": True},
            )
            flat_orientation = RewardTermConfig(
                func=rf.flat_orientation,
                weight=1.0,
                params={
                    "std": 0.25,
                    "asset_cfg": SceneEntitySelector(name="robot", body_names=(r.trunk_body_name,)),
                },
            )
            capture_point_support = RewardTermConfig(
                func=k1_rf.capture_point_support_penalty,
                weight=self.capture_point_weight,
                params={
                    "asset_cfg": feet_selector,
                    "sigma": self.capture_point_sigma,
                    "margin": self.capture_point_margin,
                    "command_threshold": self.capture_point_command_threshold,
                },
            )
            variable_posture = RewardTermConfig(func=rf.variable_posture, weight=1.0, params=posture_params)
            feet_pair_collision = RewardTermConfig(
                func=k1_rf.contact_pair_penalty,
                weight=-1.0,
                params={"contact_group": "feet_pair_contact"},
            )
            body_angular_velocity_penalty = RewardTermConfig(
                func=rf.body_angular_velocity_penalty,
                weight=0.025,
                params={"asset_cfg": SceneEntitySelector(name="robot", body_names=(r.trunk_body_name,))},
            )
            # Reads the "robot/root_angmom" sensor injected by K1SpecFn.
            angular_momentum_penalty = RewardTermConfig(
                func=rf.angular_momentum_penalty,
                weight=0.01,
                params={"sensor_name": "robot/root_angmom"},
            )
            joint_pos_limits = RewardTermConfig(func=rf.joint_pos_limits, weight=1.0)
            raw_action_rate_l2 = RewardTermConfig(func=rf_common.raw_action_rate_l2, weight=0.1)
            feet_clearance = RewardTermConfig(
                func=rf.feet_clearance,
                weight=2.0,
                params={
                    "asset_cfg": feet_selector,
                    "target_height": 0.18,
                    "command_threshold": 0.05,
                },
            )
            feet_swing_height = RewardTermConfig(
                func=rf.feet_swing_height,
                weight=0.75,
                params={
                    "contact_group": "feet_ground_contact",
                    "asset_cfg": feet_selector,
                    "target_height": 0.18,
                    "command_threshold": 0.05,
                    "contact_order": feet_contact_order,
                },
            )
            feet_slip = RewardTermConfig(
                func=rf.feet_slip,
                weight=0.1,
                params={
                    "contact_group": "feet_ground_contact",
                    "asset_cfg": feet_selector,
                    "command_threshold": 0.05,
                    "contact_order": feet_contact_order,
                },
            )
            soft_landing = RewardTermConfig(
                func=rf.soft_landing,
                weight=1e-5,
                params={
                    "contact_group": "feet_ground_contact",
                    "command_threshold": 0.05,
                },
            )

        # No total_clip: the G1 recipe does not floor the summed reward.
        return _RewardsCfg()
