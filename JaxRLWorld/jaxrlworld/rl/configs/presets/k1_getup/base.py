"""Booster K1 fall recovery (getup) on the velocity task's policy contract.

The task is the T1 getup recipe (``presets/t1_getup``, itself a port of
mjlab_playground's getup task): random fallen poses, an upright-orientation
reward, a body-height ramp, a posture term gated on being upright, and the
regularizers with their step-staged curriculum. Nothing about the task is
K1-specific beyond body names, the standing height and the posture widths.

What is K1-specific is the POLICY CONTRACT, and that is inherited unchanged
from :class:`~jaxrlworld.rl.configs.presets.k1_velocity.base.K1VelocityConfig`:

- the 75-D actor observation, in the same order with the same noise, with the
  velocity command block present and pinned to zero;
- absolute joint targets ``a * physical_action_scale + default_pose`` over the
  same 22 joints, with the same clip and the same command delay;
- the same explicit PD actuators (gains, effort limits, torque-speed curve);
- the same 50 Hz control loop and the same network.

A policy trained here therefore exports through the same path as the velocity
policy and presents the same tensor interface, so the deployment stack runs it
without a code change (a zero command is what it sends with the sticks idle).
``scripts/diag/k1/getup_io_contract_diag.py`` checks the contract against the
velocity preset field by field.

Two departures from the T1 getup recipe follow from keeping that contract:

- **Absolute targets instead of settle-relative ones.** The T1 recipe commands
  a delta on the current joint position with a uniform 0.6 scale; a deployed
  K1 policy commands a displacement from the home pose at the velocity
  recipe's per-joint scale. The hold during the first ``settle_steps`` after a
  reset, which lets a dropped robot come to rest before the policy acts, is
  the action manager's own settle mask and is a training-time reset artefact,
  not part of the policy interface.
- **The velocity recipe's randomization set.** Encoder bias, PD gain scale,
  joint damping, masses and foot friction are what the velocity policy is
  trained against, so the getup policy sees the same plant. The one addition
  is friction on the non-foot shells, which only matter once the robot lies on
  them.

The inherited velocity-only knobs (command envelope, push, gait reward
weights, fall terminations, the motion prior) are ignored by this preset's
builders; ``use_amp`` is refused rather than silently ignored.

Usage::

    from jaxrlworld.rl.configs.presets.k1_getup.base import K1GetupConfig
    cfgs = K1GetupConfig(sim_type="mujoco").build()
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict

from jaxrlworld.rl.configs.common_config_classes import CommandConfig, EventConfig, RewardConfig, RunnerConfig
from jaxrlworld.rl.configs.curriculums import CurriculumManagerConfig, CurriculumTermConfig
from jaxrlworld.rl.configs.events import EventTermConfig
from jaxrlworld.rl.configs.presets.k1_velocity.base import (
    _SIM_TIMINGS,
    _STEPS_PER_ROLLOUT,
    SELF_COLLISION,
    K1VelocityConfig,
)
from jaxrlworld.rl.configs.rewards import RewardTermConfig
from jaxrlworld.rl.configs.scene import SceneEntitySelector
from jaxrlworld.rl.envs.managers.common.command_term import VelocityCommandTermCfg
from jaxrlworld.rl.envs.mdp.curriculums.step_stages import reward_curriculum
from jaxrlworld.rl.envs.mdp.events import common as ev
from jaxrlworld.rl.envs.mdp.events.dr import unified as unified_dr
from jaxrlworld.rl.envs.mdp.rewards import k1_velocity as booster_rf
from jaxrlworld.rl.envs.mdp.rewards.common import getup as rf_getup, reward_terms as rf_common

_SIM_DEFAULT_RUN_NAMES = {
    "mujoco": "K1_Getup_Mujoco",
    "newton": "K1_Getup_Newton",
    "genesis": "K1_Getup_Genesis",
}


def _get_sim_builders(sim_type: str):
    module = {
        "mujoco": "_mujoco_builders",
        "newton": "_newton_builders",
        "genesis": "_genesis_builders",
    }[sim_type]
    return importlib.import_module(f"{__package__}.{module}")


@dataclass
class K1GetupConfig(K1VelocityConfig):
    """Getup task knobs; the policy contract is the parent's."""

    # A getup episode: 6 s, as in mjlab_playground.
    episode_length_s: float = 6.0

    # ── Reset ────────────────────────────────────────────────────────
    # ``reset_fallen_or_standing``: with probability ``fallen_prob`` the root
    # is dropped from ``fall_height`` at a uniformly random orientation with
    # every joint uniform over its soft range and velocities uniform in
    # ``fall_velocity_range``; otherwise the home pose, ``standing_z_offset``
    # above the ground. The drop height is the T1 recipe's 0.8 m scaled to
    # the K1's reach: the farthest shell is about 0.45 m from the trunk origin
    # at full extension, so 0.6 m clears the ground at any orientation.
    fallen_prob: float = 0.6
    fall_height: float = 0.6
    fall_velocity_range: tuple[float, float] = (-0.5, 0.5)
    fall_joint_noise_range: tuple[float, float] | str = "soft_limit"
    standing_z_offset: float = 0.02
    # Control steps after a reset during which the target is held at the
    # current joint position so a dropped robot lands before the policy acts.
    settle_steps: int = 30

    # ── Reward ───────────────────────────────────────────────────────
    orientation_std: float = 0.707
    # Trunk origin height to ramp towards. ``None`` is the home keyframe's
    # height (``robot.base_init_height``), which is where the posture term
    # then takes over. The T1 recipe adds a second ramp on the waist body;
    # the K1 has no separate waist link (its waist shell is on the trunk).
    trunk_desired_height: float | None = None
    posture_gate_threshold: float = 0.01
    w_orientation: float = 1.0
    w_trunk_height: float = 1.0
    w_gated_posture: float = 5.0
    # Regularizers. The action-rate and joint-velocity weights are the stage-0
    # values of the curriculum below; dof_pos_limits is inherited at 1.0.
    w_action_rate: float = 0.01
    w_joint_vel_l2: float = 0.0
    w_self_collisions: float = 0.1
    # Per-joint widths of the gated posture kernel, the T1 recipe's values on
    # the K1's joint names. The leading ``.*`` absorbs Newton's entity prefix.
    posture_std_dict: Dict[str, float] = field(
        default_factory=lambda: {
            r".*_Hip_Roll": 0.08,
            r".*_Hip_Yaw": 0.08,
            r".*_Hip_Pitch": 0.12,
            r".*_Knee_Pitch": 0.15,
            r".*_Ankle_Pitch": 0.2,
            r".*_Ankle_Roll": 0.2,
            r".*Head_Yaw": 0.15,
            r".*Head_Pitch": 0.15,
            r".*_Shoulder_Pitch": 0.5,
            r".*_Shoulder_Roll": 0.5,
            r".*_Elbow_Pitch": 0.5,
            r".*_Elbow_Yaw": 0.5,
        }
    )

    # ── Randomization ────────────────────────────────────────────────
    # Sliding friction of the non-foot shells, absolute. The asset declares
    # 0.45; a robot pushing itself up off its forearms and knees feels this
    # directly, which the velocity recipe (feet only) never exercises.
    body_friction_range: tuple[float, float] = (0.3, 0.9)

    # ── Training ─────────────────────────────────────────────────────
    max_iterations: int = 10_000

    # ── Assembly ─────────────────────────────────────────────────────

    def build(self):
        if self.use_amp:
            raise ValueError("K1GetupConfig has no motion prior; use_amp must stay False")
        builders = _get_sim_builders(self.sim_type)
        timing = _SIM_TIMINGS[self.sim_type]

        cfgs = builders.CONFIGS_FOR_RUN_CLS(
            env=builders.build_env(self, timing),
            scene=builders.build_scene(self, timing),
            visualization=builders.build_visualization(self),
            # The parent's actor / critic groups, unchanged: that is the contract.
            observation=self._build_observation_config(),
            action=builders.build_action(self),
            reward=self._build_reward_config(),
            command=self._build_command_config(),
            event=self._build_event_config(),
            algorithm=self._build_algorithm_config(),
            nn=self._build_nn_config(),
            runner=self._build_runner_config(),
        )
        cfgs.curriculum = self._build_curriculum_config()
        cfgs.preset_module = type(self).__module__
        cfgs.preset_class_name = type(self).__name__
        cfgs.preset_kwargs = self._get_preset_kwargs()
        return cfgs

    @property
    def resolved_trunk_desired_height(self) -> float:
        if self.trunk_desired_height is None:
            return self.robot.base_init_height
        return self.trunk_desired_height

    # ── Command ──────────────────────────────────────────────────────

    def _build_command_config(self) -> CommandConfig:
        """The velocity command term, pinned to zero.

        The term stays so the actor's ``velocity_command`` block exists with
        the same function the velocity preset reads; every environment is a
        standing environment and the ranges are degenerate, so it is always
        ``[0, 0, 0]``. The heading mode is off because it would regenerate a
        yaw-rate command from the heading error while the robot rolls over.
        """
        return CommandConfig(
            terms={
                "velocity": VelocityCommandTermCfg(
                    resampling_time_range=self.command_resampling_time_range,
                    lin_vel_x_range=(0.0, 0.0),
                    lin_vel_y_range=(0.0, 0.0),
                    ang_vel_range=(0.0, 0.0),
                    rel_standing_envs=1.0,
                    heading_command=False,
                    rel_heading_envs=0.0,
                )
            }
        )

    # ── Reward ───────────────────────────────────────────────────────

    def _build_reward_config(self) -> RewardConfig:
        r = self.robot
        sim = self.sim_type
        is_mujoco = sim == "mujoco"
        rf = importlib.import_module(
            "jaxrlworld.rl.envs.mdp.rewards.mujoco.reward_terms"
            if is_mujoco
            else f"jaxrlworld.rl.envs.mdp.rewards.{sim}.mjlab_rewards"
        )
        fn_joint_limits = rf.joint_pos_limits if is_mujoco else rf.joint_pos_limits_mjlab
        fn_action_rate = rf_common.raw_action_rate_l2 if is_mujoco else rf.raw_action_rate_l2_mjlab
        # ``height_to_target`` looks the body up through ``RobotData.body_pos_w``,
        # whose name semantics differ per backend: Newton matches a regex
        # against entity-prefixed labels (``K1/Trunk``), so it needs the
        # leading ``.*``; Genesis takes the exact link name and rejects a
        # pattern; MuJoCo accepts either.
        trunk_body = f".*{r.trunk_body_name}" if sim == "newton" else r.trunk_body_name
        cfg = self

        @dataclass
        class _RewardsCfg(RewardConfig):
            orientation_upright = RewardTermConfig(
                func=rf_getup.orientation_upright,
                weight=cfg.w_orientation,
                params={"std": cfg.orientation_std},
            )
            trunk_height = RewardTermConfig(
                func=rf_getup.height_to_target,
                weight=cfg.w_trunk_height,
                params={"desired_height": cfg.resolved_trunk_desired_height, "body_name": trunk_body},
            )
            gated_posture = RewardTermConfig(
                func=rf_getup.GatedPostureTracker,
                weight=cfg.w_gated_posture,
                params={"std_dict": cfg.posture_std_dict, "gate_threshold": cfg.posture_gate_threshold},
            )
            dof_pos_limits = RewardTermConfig(func=fn_joint_limits, weight=cfg.w_dof_pos_limits)
            action_rate = RewardTermConfig(func=fn_action_rate, weight=cfg.w_action_rate)
            joint_vel_l2 = RewardTermConfig(func=rf_common.penalize_dof_vel, weight=cfg.w_joint_vel_l2)
            self_collisions = RewardTermConfig(
                func=booster_rf.self_collision_substep_count,
                weight=cfg.w_self_collisions,
                params={"contact_group": SELF_COLLISION},
            )

        # The self-collision term reads substep contact history, which the
        # compiled reward chain does not snapshot (see the velocity preset).
        return _RewardsCfg(compile_terms=False)

    # ── Events ───────────────────────────────────────────────────────

    def _build_event_config(self) -> EventConfig:
        r = self.robot
        terms: Dict[str, EventTermConfig] = {
            "reset_fallen_or_standing": EventTermConfig(
                func=ev.reset_fallen_or_standing,
                mode="reset",
                params={
                    "fallen_prob": self.fallen_prob,
                    "fall_height": self.fall_height,
                    "fall_velocity_range": self.fall_velocity_range,
                    "fall_joint_noise_range": self.fall_joint_noise_range,
                    "standing_z_offset": self.standing_z_offset,
                    "default_pos": (0.0, 0.0, r.base_init_height),
                    "default_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
                    "default_joint_pos_dict": r.default_joint_angles,
                },
            ),
        }
        # No push: the task's disturbance is the reset itself.
        terms.update(self._build_dr_terms())

        events = EventConfig()
        for name, term in terms.items():
            setattr(events, name, term)
        return events

    def _build_dr_terms(self) -> Dict[str, EventTermConfig]:
        """The velocity recipe's set plus sliding friction on the non-foot shells."""
        terms = super()._build_dr_terms()
        r = self.robot
        lo, hi = self.body_friction_range
        if self.sim_type == "genesis":
            # Genesis' setter is multiplicative on the asset's 0.45 and
            # addresses links, not geoms.
            asset_friction = 0.45
            body_friction = EventTermConfig(
                func=unified_dr.randomize_friction,
                mode="reset_dr",
                params={
                    "asset_cfg": SceneEntitySelector(name="robot", body_names=(r.non_foot_body_pattern,)),
                    "friction_range": (lo / asset_friction, hi / asset_friction),
                    "operation": "scale",
                },
            )
        else:
            body_friction = EventTermConfig(
                func=unified_dr.randomize_friction,
                mode="reset_dr",
                params={
                    "asset_cfg": SceneEntitySelector(name="robot", geom_names=(r.non_foot_geom_pattern,)),
                    "friction_range": (lo, hi),
                    "operation": "abs",
                    "axes": [0],
                },
            )
        if self.dr_interval_period_s is not None:
            body_friction.mode = "interval_dr"
            body_friction.interval_dr_period_s = self.dr_interval_period_s
        terms["dr_body_friction"] = body_friction
        return terms

    # ── Curriculum ───────────────────────────────────────────────────

    def _build_curriculum_config(self) -> CurriculumManagerConfig:
        """mjlab_playground's getup schedule: tighten the regularizers once
        the policy can get up. Stages are (iteration x steps-per-rollout)."""

        @dataclass
        class _CurriculumCfg(CurriculumManagerConfig):
            action_rate_weight: CurriculumTermConfig = field(
                default_factory=lambda: CurriculumTermConfig(
                    func=reward_curriculum,
                    params={
                        "reward_name": "action_rate",
                        "stages": [
                            {"step": 0, "weight": 0.01},
                            {"step": 600 * _STEPS_PER_ROLLOUT, "weight": 0.05},
                            {"step": 900 * _STEPS_PER_ROLLOUT, "weight": 0.08},
                            {"step": 1200 * _STEPS_PER_ROLLOUT, "weight": 0.10},
                        ],
                    },
                )
            )
            joint_vel_weight: CurriculumTermConfig = field(
                default_factory=lambda: CurriculumTermConfig(
                    func=reward_curriculum,
                    params={
                        "reward_name": "joint_vel_l2",
                        "stages": [
                            {"step": 0, "weight": 0.0},
                            {"step": 900 * _STEPS_PER_ROLLOUT, "weight": 0.005},
                            {"step": 1200 * _STEPS_PER_ROLLOUT, "weight": 0.008},
                            {"step": 1500 * _STEPS_PER_ROLLOUT, "weight": 0.010},
                        ],
                    },
                )
            )

        return _CurriculumCfg()

    # ── Training ─────────────────────────────────────────────────────

    def _build_runner_config(self) -> RunnerConfig:
        return RunnerConfig(
            checkpoint=-1,
            log_interval=1,
            max_iterations=self.max_iterations,
            init_at_random_ep_len=False,
            resume=False,
            resume_path=None,
            run_name=self.run_name or _SIM_DEFAULT_RUN_NAMES[self.sim_type],
            logger="wandb",
            wandb_project="K1_Getup",
            save_interval=2000,
            output_dir="auto",
        )
