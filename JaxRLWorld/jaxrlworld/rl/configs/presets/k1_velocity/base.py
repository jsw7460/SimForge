"""Booster K1 velocity tracking, ported from the booster_mjlab recipe.

A port of the ``Mjlab-Velocity-Flat-Booster-K1`` task onto this framework's
three backends. The source runs on mjlab alone; everything here is arranged so
MuJoCo, Newton and Genesis express the same MDP.

WHAT IS REPRODUCED
------------------
The 75-D actor observation and its noise scales, the 90-D privileged critic
observation, all fifteen reward terms with their weights, the three termination
conditions, the domain-randomization set, the velocity command with heading
control, and the two step-staged curricula (soft-landing weight, command
envelope). The robot is the source asset's full-body collision model with its
PD tuning (see :mod:`jaxrlworld.rl.configs.robots.k1`).

WHAT IS NOT, AND WHY
--------------------
- **Forward-only command envs.** The source dedicates 10% of environments to
  commands with no lateral or yaw component. This framework's velocity command
  term has no such split, and adding one would change a shared command term for
  every preset that uses it.
- **Terrain compliance randomization.** The source randomizes the ground's
  ``solref``/``solimp`` by editing the compiled MuJoCo spec, which only the
  MuJoCo backend would see. On flat ground the effect is a contact-stiffness
  jitter with no cross-backend equivalent, so it is dropped rather than applied
  to one backend in three.
- **Foot clearance measured from the sole.** The source ray-casts a 5x3 grid
  over each sole and takes the lowest sample's height above the terrain. Only
  the MuJoCo backend here can ray-cast, so clearance is the foot link origin's
  height instead, with every target raised by :attr:`foot_sole_offset` to keep
  the same sole clearance. Two consequences: on flat ground the values agree
  except for foot tilt, and this preset is flat-only until the other two
  backends can sample terrain height.
- **Rough terrain.** Follows from the point above, and from the sub-terrain
  library here carrying only random-rough (the source mixes in pyramid slopes
  and Perlin noise).
- **Standing pose penalty ten times the source's weight.** 5.0 against 0.5.
  See :attr:`K1VelocityConfig.w_standing_pose_l1` for why: under the
  source's push the source weight lets a standing policy crouch, and more
  training only deepens it.
- **Foot slip read from a finite difference.** The source reads the foot's
  instantaneous velocity out of the simulator. On the MuJoCo backend that read
  is one substep stale at the step boundary, which inflates touchdown slip
  against the other two by roughly a factor of three -- measured here before,
  on a quadruped. Since the point of this port is that the three backends price
  the same behaviour the same way, all three take the foot velocity from a
  finite difference of its position instead. This is the one place where the
  port deliberately prefers agreement between backends over agreement with the
  source.

SIGN CONVENTION
---------------
Penalties return negative and carry positive weights here; the source does the
opposite. Weights below are therefore the absolute values of the source's, and
the products are identical.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from jaxrlworld.rl.configs.algorithms.amp_ppo import AmpConfig, AmpPPOConfig
from jaxrlworld.rl.configs.algorithms.ppo import PPOConfig, SymmetryConfig
from jaxrlworld.rl.configs.common_config_classes import (
    Activation,
    CommandConfig,
    DistributionType,
    EventConfig,
    MLPActorCfg,
    MLPCriticCfg,
    NNConfig,
    ObservationGroupConfig,
    OrthoInit,
    PPOPolicyConfig,
    RewardConfig,
    RunnerConfig,
    StdType,
)
from jaxrlworld.rl.configs.curriculums import CurriculumManagerConfig, CurriculumTermConfig
from jaxrlworld.rl.configs.events import EventTermConfig
from jaxrlworld.rl.configs.observations import ObservationTermConfig
from jaxrlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from jaxrlworld.rl.configs.rewards import RewardTermConfig
from jaxrlworld.rl.configs.robots.k1 import K1Config
from jaxrlworld.rl.configs.scene import SceneEntitySelector
from jaxrlworld.rl.envs.managers.common.command_term import VelocityCommandTermCfg
from jaxrlworld.rl.envs.mdp.curriculums.step_stages import reward_curriculum
from jaxrlworld.rl.envs.mdp.events import common as ev
from jaxrlworld.rl.envs.mdp.events.dr import unified as unified_dr
from jaxrlworld.rl.envs.mdp.events.motion_pose_pool import MotionPosePoolSpec, reset_from_motion_pose_pool
from jaxrlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    base_lin_vel,
    dof_pos_nominal_difference,
    dof_pos_nominal_difference_biased,
    dof_vel,
    foot_air_time,
    foot_contact_forces,
    foot_contact_indicator,
    foot_height,
    joint_pos_rel,
    joint_vel_rel,
    projected_gravity,
    raw_actions,
)
from jaxrlworld.rl.envs.mdp.observations.k1_locomotion import velocity_command
from jaxrlworld.rl.envs.mdp.rewards import k1_velocity as booster_rf
from jaxrlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common

# Source timing: 5 ms physics step, decimation 4, so a 50 Hz control loop.
_SIM_TIMINGS: Dict[str, Dict[str, Any]] = {
    "mujoco": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "newton": {"dt": 0.005, "substeps": 1, "decimation": 4},
    "genesis": {"dt": 0.005, "substeps": 1, "decimation": 4},
}

_SIM_DEFAULT_RUN_NAMES = {
    "mujoco": "K1_Mujoco",
    "newton": "K1_Newton",
    "genesis": "K1_Genesis",
}

# Contact groups every backend registers under the same names.
FEET_GROUND = "feet_ground_contact"
NON_FOOT_GROUND = "non_foot_ground_contact"
SELF_COLLISION = "self_collision"

# One control step, in source units: the curricula are staged in
# (iteration x steps-per-rollout) products, which is what the source writes.
_STEPS_PER_ROLLOUT = 24


def _get_sim_builders(sim_type: str):
    module = {
        "mujoco": "_mujoco_builders",
        "newton": "_newton_builders",
        "genesis": "_genesis_builders",
    }[sim_type]
    return importlib.import_module(f"{__package__}.{module}")


@dataclass
class K1VelocityConfig:
    """Task knobs. Defaults are the source recipe's; retuning breaks parity."""

    sim_type: str = "mujoco"
    robot: K1Config = field(default_factory=K1Config)
    num_envs: int = 16384
    seed: int = 42
    episode_length_s: float = 20.0

    # ── Command ──────────────────────────────────────────────────────
    # The source samples from a narrow initial envelope that the command
    # curriculum immediately replaces at step 0; these are that stage-0
    # envelope, so training starts where the source's does.
    lin_vel_x_range: tuple = (-1.0, 1.2)
    lin_vel_y_range: tuple = (-1.0, 1.0)
    ang_vel_range: tuple = (-1.0, 1.0)
    command_resampling_time_range: tuple = (1.0, 4.0)
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 0.3
    heading_command: bool = True
    heading_control_stiffness: float = 0.5

    # ── Geometry ─────────────────────────────────────────────────────
    # Drop from the foot link origin down to the sole reference point. Every
    # clearance target is the source's sole-referenced value plus this, because
    # clearance is measured at the link origin here. See the module docstring.
    foot_sole_offset: float = 0.03
    target_sole_clearance: float = 0.06

    # ── Reward weights (absolute values of the source's) ─────────────
    w_track_linear_velocity: float = 2.25
    w_track_angular_velocity: float = 2.0
    w_upright: float = 1.0
    w_body_ang_vel: float = 0.01
    w_angular_momentum: float = 0.005
    w_dof_pos_limits: float = 1.0
    w_action_rate: float = 0.1
    w_air_time: float = 0.1
    w_foot_clearance: float = 2.0
    w_foot_swing_height: float = 0.25
    w_foot_slip: float = 0.2
    # The source's config value. Its curriculum drops this to 0.0001 at step 0
    # and raises it to 0.005 later, so this is only what the term is worth for
    # the steps before the curriculum first runs -- but those steps exist, and
    # setting this to the step-0 value instead made them ten times cheaper here
    # than in the source.
    w_soft_landing: float = 0.001
    w_self_collisions: float = 1.0
    w_upper_body_posture: float = 0.1
    # Deliberate departure from the source's 0.5. At that weight a standing
    # policy under the source's 6-DoF push settles into a deep crouch: the
    # crouch costs ~0.03 per step here while cutting the far larger
    # velocity-tracking loss a kick causes (rest std 0.1), and more training
    # only deepens it. Ten times the weight makes the home pose the cheaper
    # place to absorb a push from. Walking terms are untouched (this term is
    # gated to a zero command).
    w_standing_pose_l1: float = 5.0

    # ── Reward shaping constants ─────────────────────────────────────
    # Linear tracking: the tolerance runs from std_at_rest at a zero command up
    # to std once the command exceeds std / relative_std, and half the term is
    # a linear progress score so a standing policy still feels a gradient.
    track_lin_std: float = math.sqrt(0.25)
    track_lin_std_at_rest: float = math.sqrt(0.01)
    track_lin_relative_std: float = 0.75
    track_lin_progress_weight: float = 0.5
    track_ang_std: float = math.sqrt(0.5)

    upright_std_standing: float = math.sqrt(0.20)
    upright_std_walking: float = math.sqrt(0.25)
    upright_std_running: float = math.sqrt(0.35)
    upright_walking_threshold: float = 0.05
    upright_running_threshold: float = 1.5

    posture_walking_threshold: float = 0.05
    posture_running_threshold: float = 1.0

    air_time_threshold_min: float = 0.05
    air_time_threshold_max: float = 0.5
    air_time_command_threshold: float = 0.2
    gait_command_threshold: float = 0.05
    standing_command_threshold: float = 0.05

    # ── Terminations ─────────────────────────────────────────────────
    fall_limit_angle_deg: float = 63.0
    # Per control step while over the tilt limit. A tilted robot survives an
    # expected 50 steps, so recovery stays reachable instead of being cut off
    # the instant the limit is crossed.
    fall_probability: float = 0.02

    # ── Events ───────────────────────────────────────────────────────
    # The source's push: a 6-DoF velocity kick added to the root every
    # interval, each axis drawn independently. Its planar magnitude peaks at
    # 0.4 m/s and averages about 0.2. A stronger planar-only push (as the K1
    # joystick presets use, up to 1.0 m/s) teaches a standing policy to crouch
    # for push rejection, since the pose penalty is far cheaper than the
    # tracking loss a stumble costs.
    push_interval_range_s: tuple = (1.5, 4.0)
    push_velocity_range: Dict[str, tuple] = field(
        default_factory=lambda: {
            "x": (-0.28, 0.28),
            "y": (-0.28, 0.28),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        }
    )
    dr_interval_period_s: float | None = 10.0

    # ── Action ───────────────────────────────────────────────────────
    # The builders override the scale with the robot's per-joint
    # 0.25 * effort / kp, so this placeholder is never read.
    action_distribution: str = "gaussian"
    action_scale: Any = 1.0
    action_clip: tuple = (-100.0, 100.0)
    action_delay_min: int = 2
    action_delay_max: int = 8

    # Left/right symmetry (Mittal et al. 2024). The source's "-DA" tasks double
    # each minibatch with mirrored samples; that is symmetry_data_augmentation
    # here. The auxiliary mirror loss is the paper's weaker alternative and is
    # kept as a separate switch. Both default off, matching the source's plain
    # "Flat-Booster-K1" task.
    mirror_symmetry_coeff: float = 0.0
    symmetry_data_augmentation: bool = False

    # ── Adversarial motion prior ─────────────────────────────────────
    # The source's "-Amp" tasks add an ``amp`` observation group the
    # discriminator scores: the leg joints relative to the home pose, their
    # velocities, the base linear velocity and the projected gravity, one
    # frame each (the history is stacked by the algorithm). Arms and head are
    # left out so the dataset's arm style does not drive the style reward.
    use_amp: bool = False
    amp_joint_patterns: tuple = (r".*_Hip_.*", r".*_Knee_.*", r".*_Ankle_.*")
    amp_style_reward_weight: float = 0.3
    amp_pose_pool_size: int = 16384
    # The source's standing-pose weight. The 10x value above answers a crouch
    # that appears WITHOUT a motion prior; with one, the pool resets start
    # zero-command envs away from the home pose and the style reward asks for
    # the data's deeper knee bend and forward lean, so the 10x term fights the
    # prior every step (measured: the posture gap to the data did not close
    # over 3000 iterations at 5.0).
    amp_w_standing_pose_l1: float = 0.5
    # Reference clips: the LAFAN1 locomotion set retargeted to the K1, converted
    # at the control rate by ``scripts/k1/convert_lafan_k1.py``.
    amp_motion_dir: str = str(Path(__file__).resolve().parents[4] / "assets" / "motions" / "lafan1_k1")

    # ── Training ─────────────────────────────────────────────────────
    algorithm_name: str = "PPO"
    max_iterations: int = 30_000
    actor_hidden_dims: tuple = (512, 256, 128)
    run_name: str | None = None

    # ── Assembly ─────────────────────────────────────────────────────

    def build(self):
        if self.use_amp and self.algorithm_name == "PPO":
            # The motion prior is an algorithm, not a reward term: switching
            # it on selects AMP_PPO unless another PPO variant was named.
            self.algorithm_name = "AMP_PPO"
        builders = _get_sim_builders(self.sim_type)
        timing = _SIM_TIMINGS[self.sim_type]

        cfgs = builders.CONFIGS_FOR_RUN_CLS(
            env=builders.build_env(self, timing),
            scene=builders.build_scene(self, timing),
            visualization=builders.build_visualization(self),
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

    def _get_preset_kwargs(self) -> Dict[str, Any]:
        from dataclasses import MISSING, fields

        kwargs: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "robot":
                continue
            value = getattr(self, f.name)
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                default = f.default_factory()  # type: ignore[misc]
            else:
                kwargs[f.name] = value
                continue
            if value != default:
                kwargs[f.name] = value
        return kwargs

    @property
    def target_foot_height(self) -> float:
        """Clearance target at the foot link origin, matching the source's sole target."""
        return self.target_sole_clearance + self.foot_sole_offset

    # ── Observations ─────────────────────────────────────────────────
    #
    # Actor (75-D), in source order:
    #   [base_ang_vel(3) projected_gravity(3) joint_pos(22) joint_vel(22)
    #    last_action(22) command(3)]
    # with the source's uniform noise scales and no observation scaling.
    #
    # Critic (90-D): the same block with CLEAN joint state (no encoder bias, no
    # noise), then the privileged extras in source order. The source builds the
    # critic by overriding two keys of the actor dict, which keeps them in their
    # original positions -- this layout reproduces that.

    def _build_observation_config(self):
        builders = _get_sim_builders(self.sim_type)
        obs_cfg_cls = builders.OBSERVATION_CFG_CLS
        feet_bodies = tuple(self.robot.foot_names)

        @dataclass
        class _ActorObsCfg(ObservationGroupConfig):
            base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
            projected_gravity = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
            # Biased by the per-episode encoder offset the DR term writes; the
            # critic below reads the same joints unbiased.
            joint_pos = ObservationTermConfig(
                func=dof_pos_nominal_difference_biased, scale=1.0, noise=Unoise(-0.01, 0.01)
            )
            joint_vel = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
            actions = ObservationTermConfig(func=raw_actions, scale=1.0)
            command = ObservationTermConfig(func=velocity_command, scale=1.0)

        @dataclass
        class _CriticObsCfg(ObservationGroupConfig):
            enable_corruption: bool = False

            base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=1.0)
            projected_gravity = ObservationTermConfig(func=projected_gravity, scale=1.0)
            joint_pos = ObservationTermConfig(func=dof_pos_nominal_difference, scale=1.0)
            joint_vel = ObservationTermConfig(func=dof_vel, scale=1.0)
            actions = ObservationTermConfig(func=raw_actions, scale=1.0)
            command = ObservationTermConfig(func=velocity_command, scale=1.0)
            # Privileged extras.
            base_lin_vel = ObservationTermConfig(func=base_lin_vel, scale=1.0)
            foot_height = ObservationTermConfig(func=foot_height, scale=1.0, params={"body_names": feet_bodies})
            foot_air_time = ObservationTermConfig(func=foot_air_time, scale=1.0, params={"contact_group": FEET_GROUND})
            foot_contact = ObservationTermConfig(
                func=foot_contact_indicator, scale=1.0, params={"contact_group": FEET_GROUND}
            )
            foot_contact_forces = ObservationTermConfig(
                func=foot_contact_forces, scale=1.0, params={"contact_group": FEET_GROUND}
            )

        if not self.use_amp:

            @dataclass
            class _ObsCfg(obs_cfg_cls):
                actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
                critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

            return _ObsCfg()

        amp_joints = SceneEntitySelector(name="robot", joint_names=tuple(self.amp_joint_patterns))

        @dataclass
        class _AmpObsCfg(ObservationGroupConfig):
            """Discriminator features, single frame, no noise (the source's ``amp`` group)."""

            enable_corruption: bool = False

            joint_pos = ObservationTermConfig(func=joint_pos_rel, scale=1.0, params={"asset_cfg": amp_joints})
            joint_vel = ObservationTermConfig(func=joint_vel_rel, scale=1.0, params={"asset_cfg": amp_joints})
            base_lin_vel = ObservationTermConfig(func=base_lin_vel, scale=1.0)
            projected_gravity = ObservationTermConfig(func=projected_gravity, scale=1.0)

        @dataclass
        class _ObsCfgAmp(obs_cfg_cls):
            actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
            critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)
            amp: _AmpObsCfg = field(default_factory=_AmpObsCfg)

        return _ObsCfgAmp()

    # ── Rewards ──────────────────────────────────────────────────────

    @property
    def _upper_body_std_standing(self) -> Dict[str, float]:
        return {r"Head_.*": 0.05, r".*_Shoulder_.*": 0.05, r".*_Elbow_.*": 0.05}

    @property
    def _upper_body_std_walking(self) -> Dict[str, float]:
        return {r"Head_.*": 0.05, r".*_Shoulder_.*": 0.15, r".*_Elbow_.*": 0.15}

    @property
    def _upper_body_std_running(self) -> Dict[str, float]:
        return {
            r"Head_.*": 0.05,
            r".*_Shoulder_Pitch": 0.5,
            r".*_Shoulder_Roll": 0.2,
            r".*_Elbow_.*": 0.35,
        }

    def _build_reward_config(self) -> RewardConfig:
        r = self.robot
        sim = self.sim_type
        is_mujoco = sim == "mujoco"
        rf = importlib.import_module(
            "jaxrlworld.rl.envs.mdp.rewards.mujoco.reward_terms"
            if is_mujoco
            else f"jaxrlworld.rl.envs.mdp.rewards.{sim}.mjlab_rewards"
        )
        feet_selector = SceneEntitySelector(name="robot", body_names=tuple(r.foot_names), preserve_order=True)
        upper_body = SceneEntitySelector(name="robot", joint_names=r.upper_body_joint_patterns)
        all_joints = SceneEntitySelector(name="robot", joint_names=(".*",))
        trunk = SceneEntitySelector(name="robot", body_names=(r.trunk_body_name,))
        feet_order = list(r.foot_names)
        target_height = self.target_foot_height

        # Term-name suffixes and parameter shapes differ per backend for the
        # same computation; resolve both here so the term table below reads the
        # same on all three.
        if is_mujoco:
            fn_air_time = rf.feet_air_time
            p_air_time = {"contact_group": FEET_GROUND}
            fn_swing = rf.feet_swing_height
            p_swing = {"contact_group": FEET_GROUND}
            fn_slip = rf.feet_slip_fd
            p_slip = {"contact_group": FEET_GROUND}
            fn_soft = rf.soft_landing
            p_soft = {"contact_group": FEET_GROUND}
            fn_clearance = rf.feet_clearance
            fn_body_ang_vel = rf.body_angular_velocity_penalty
            fn_angmom = rf.angular_momentum_penalty
            p_angmom = {"sensor_name": "robot/root_angmom"}
            fn_joint_limits = rf.joint_pos_limits
            fn_action_rate = rf_common.raw_action_rate_l2
        else:
            fn_air_time = rf.feet_air_time_mjlab
            p_air_time = {"feet_bodies": feet_order} if sim == "newton" else {"contact_group": FEET_GROUND}
            fn_swing = rf.feet_swing_height_mjlab
            p_swing = {"contact_order": feet_order}
            fn_slip = rf.feet_slip_fd_mjlab
            p_slip = {"contact_order": feet_order}
            fn_soft = rf.soft_landing_mjlab
            p_soft = {"feet_bodies": feet_order} if sim == "newton" else {"contact_group": FEET_GROUND}
            fn_clearance = rf.feet_clearance_mjlab
            fn_body_ang_vel = rf.body_ang_vel_penalty_mjlab
            fn_angmom = rf.angular_momentum_penalty
            p_angmom = {}
            fn_joint_limits = rf.joint_pos_limits_mjlab
            fn_action_rate = rf.raw_action_rate_l2_mjlab

        cfg = self

        @dataclass
        class _RewardsCfg(RewardConfig):
            # Tracking.
            track_linear_velocity = RewardTermConfig(
                func=booster_rf.track_lin_vel_relative_std,
                weight=cfg.w_track_linear_velocity,
                params={
                    "std": cfg.track_lin_std,
                    "std_at_rest": cfg.track_lin_std_at_rest,
                    "relative_std": cfg.track_lin_relative_std,
                    "progress_weight": cfg.track_lin_progress_weight,
                },
            )
            track_angular_velocity = RewardTermConfig(
                func=rf_common.track_ang_vel,
                weight=cfg.w_track_angular_velocity,
                params={"std": cfg.track_ang_std},
            )
            # Posture.
            upright = RewardTermConfig(
                func=booster_rf.variable_upright,
                weight=cfg.w_upright,
                params={
                    "std_standing": cfg.upright_std_standing,
                    "std_walking": cfg.upright_std_walking,
                    "std_running": cfg.upright_std_running,
                    "walking_threshold": cfg.upright_walking_threshold,
                    "running_threshold": cfg.upright_running_threshold,
                },
            )
            upper_body_posture = RewardTermConfig(
                func=booster_rf.upper_body_posture_penalty,
                weight=cfg.w_upper_body_posture,
                params={
                    "asset_cfg": upper_body,
                    "std_standing": cfg._upper_body_std_standing,
                    "std_walking": cfg._upper_body_std_walking,
                    "std_running": cfg._upper_body_std_running,
                    "walking_threshold": cfg.posture_walking_threshold,
                    "running_threshold": cfg.posture_running_threshold,
                },
            )
            standing_pose_l1 = RewardTermConfig(
                func=booster_rf.standing_pose_deviation_l1,
                weight=cfg.amp_w_standing_pose_l1 if cfg.use_amp else cfg.w_standing_pose_l1,
                params={
                    "asset_cfg": all_joints,
                    "command_threshold": cfg.standing_command_threshold,
                },
            )
            # Gait shaping.
            air_time = RewardTermConfig(
                func=fn_air_time,
                weight=cfg.w_air_time,
                params={
                    **p_air_time,
                    "threshold_min": cfg.air_time_threshold_min,
                    "threshold_max": cfg.air_time_threshold_max,
                    "command_threshold": cfg.air_time_command_threshold,
                },
            )
            foot_clearance = RewardTermConfig(
                func=fn_clearance,
                weight=cfg.w_foot_clearance,
                params={
                    "asset_cfg": feet_selector,
                    "target_height": target_height,
                    "command_threshold": cfg.gait_command_threshold,
                },
            )
            foot_swing_height = RewardTermConfig(
                func=fn_swing,
                weight=cfg.w_foot_swing_height,
                params={
                    **p_swing,
                    "asset_cfg": feet_selector,
                    "target_height": target_height,
                    "command_threshold": cfg.gait_command_threshold,
                },
            )
            # Finite-difference foot velocity, not the engine's instantaneous
            # read: the MuJoCo backend's is one substep stale at the step
            # boundary, which inflates touchdown slip against the other two.
            foot_slip = RewardTermConfig(
                func=fn_slip,
                weight=cfg.w_foot_slip,
                params={
                    **p_slip,
                    "asset_cfg": feet_selector,
                    "command_threshold": cfg.gait_command_threshold,
                },
            )
            soft_landing = RewardTermConfig(
                func=fn_soft,
                weight=cfg.w_soft_landing,
                params={**p_soft, "command_threshold": cfg.gait_command_threshold},
            )
            # Regularizers.
            body_ang_vel = RewardTermConfig(
                func=fn_body_ang_vel, weight=cfg.w_body_ang_vel, params={"asset_cfg": trunk}
            )
            angular_momentum = RewardTermConfig(func=fn_angmom, weight=cfg.w_angular_momentum, params=dict(p_angmom))
            dof_pos_limits = RewardTermConfig(func=fn_joint_limits, weight=cfg.w_dof_pos_limits)
            action_rate = RewardTermConfig(func=fn_action_rate, weight=cfg.w_action_rate)
            # The shared self-collision penalties count bodies, or return a
            # bare 0/1; the source counts substeps. See the term's docstring.
            self_collisions = RewardTermConfig(
                func=booster_rf.self_collision_substep_count,
                weight=cfg.w_self_collisions,
                params={"contact_group": SELF_COLLISION},
            )

        # The source does not floor the summed reward.
        #
        # The compiled reward chain is off. Three terms in this set read
        # through it and cannot: the self-collision and soft-landing terms ask
        # the contact manager for substep force history, and Newton's air-time
        # term resolves its foot bodies through the scene model. None of those
        # are part of the chain's snapshot, and each raises rather than
        # degrading. Enabling compilation means replacing all three with
        # snapshot-only equivalents, which would change what they compute.
        return _RewardsCfg(compile_terms=False)

    # ── Commands ─────────────────────────────────────────────────────

    def _build_command_config(self) -> CommandConfig:
        return CommandConfig(
            terms={
                "velocity": VelocityCommandTermCfg(
                    resampling_time_range=self.command_resampling_time_range,
                    lin_vel_x_range=self.lin_vel_x_range,
                    lin_vel_y_range=self.lin_vel_y_range,
                    ang_vel_range=self.ang_vel_range,
                    rel_standing_envs=self.rel_standing_envs,
                    heading_command=self.heading_command,
                    heading_control_stiffness=self.heading_control_stiffness,
                    rel_heading_envs=self.rel_heading_envs,
                )
            }
        )

    # ── Events ───────────────────────────────────────────────────────

    def _build_event_config(self) -> EventConfig:
        terms: Dict[str, EventTermConfig] = {
            "reset_root": EventTermConfig(
                func=ev.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.01, 0.05),
                        "yaw": (-3.14, 3.14),
                    },
                    "velocity_range": {
                        "x": (-1.0, 1.0),
                        "y": (-1.0, 1.0),
                        "z": (0.01, 0.3),
                        "roll": (-0.1, 0.1),
                        "pitch": (-0.1, 0.1),
                        "yaw": (-0.5, 0.5),
                    },
                    "default_pos": (0.0, 0.0, self.robot.base_init_height),
                },
            ),
            "reset_joints": EventTermConfig(
                func=ev.reset_joints_by_offset,
                mode="reset",
                params={"position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)},
            ),
            "push": EventTermConfig(
                func=ev.push_by_setting_velocity,
                mode="interval",
                interval_range_s=self.push_interval_range_s,
                params={"velocity_range": dict(self.push_velocity_range)},
            ),
        }
        if self.use_amp:
            # The source's "-Amp" tasks replace the home-pose resets with
            # draws from a pool of validated reference frames (random planar
            # offset and yaw, the frame's own velocities), so episodes start
            # inside the motion distribution the discriminator scores.
            del terms["reset_root"], terms["reset_joints"]
            terms["reset_robot_from_motion"] = EventTermConfig(
                func=reset_from_motion_pose_pool,
                mode="reset",
                params={
                    "spec": MotionPosePoolSpec(
                        mjcf_path=self.robot.mjcf_path,
                        motion_files=self._amp_motion_files(),
                        root_body_name=self.robot.base_link_name,
                        mirror_augmentation=True,
                        speed_augmentations=self._AMP_SPEED_AUGMENTATIONS,
                        pool_size=self.amp_pose_pool_size,
                        seed=self.seed,
                        xy_range=(-0.5, 0.5),
                        z_range=(0.0, 0.05),
                        yaw_range=(-3.14, 3.14),
                        foot_site_names=("left_foot", "right_foot"),
                        foot_body_names=tuple(self.robot.foot_names),
                        min_foot_height=0.0,
                    )
                },
            )
        terms.update(self._build_dr_terms())

        events = EventConfig()
        for name, term in terms.items():
            setattr(events, name, term)
        return events

    def _build_dr_terms(self) -> Dict[str, EventTermConfig]:
        """The source's randomization set, minus the terrain-compliance term.

        Full-tensor inertia randomization has no equivalent here; the source
        perturbs each body's pseudo-inertia, which this framework covers with a
        scalar mass draw plus a trunk COM offset. Those two carry most of the
        same effect and are what every other preset in the repository uses.
        """
        r = self.robot
        all_bodies = SceneEntitySelector(name="robot", body_names=(".*",))
        trunk = SceneEntitySelector(name="robot", body_names=(r.trunk_body_name,))
        all_joints = SceneEntitySelector(name="robot")

        if self.sim_type == "genesis":
            # Genesis' friction and mass setters are multiplicative, so the
            # ranges are expressed relative to the MJCF defaults (foot friction
            # 1.0, trunk mass 6.5 kg) to land on the same absolute values.
            friction_term = EventTermConfig(
                func=unified_dr.randomize_friction,
                mode="reset_dr",
                params={
                    "asset_cfg": SceneEntitySelector(name="robot", body_names=tuple(r.foot_names)),
                    "friction_range": (0.75, 1.25),
                    "operation": "scale",
                },
            )
        else:
            friction_term = EventTermConfig(
                func=unified_dr.randomize_friction,
                mode="reset_dr",
                params={
                    "asset_cfg": SceneEntitySelector(name="robot", geom_names=r.foot_geom_names),
                    "friction_range": (0.75, 1.25),
                    "operation": "abs",
                    "axes": [0],
                },
            )

        terms: Dict[str, EventTermConfig] = {
            "dr_foot_friction": friction_term,
            # The trunk carries the payload, so its mass and COM move further
            # than the limbs'.
            "dr_trunk_mass": EventTermConfig(
                func=unified_dr.randomize_body_mass,
                mode="reset_dr",
                params={"asset_cfg": trunk, "mass_range": (0.95, 1.05), "operation": "scale"},
            ),
            "dr_trunk_com": EventTermConfig(
                func=unified_dr.randomize_body_com_offset,
                mode="reset_dr",
                params={
                    "asset_cfg": trunk,
                    "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
                    "operation": "add",
                },
            ),
            "dr_link_mass": EventTermConfig(
                func=unified_dr.randomize_body_mass,
                mode="reset_dr",
                params={"asset_cfg": all_bodies, "mass_range": (0.95, 1.05), "operation": "scale"},
            ),
            # The asset declares no passive joint damping at all, so the PD law
            # is the only thing resisting motion. A real harmonic drive is not
            # frictionless, and the arms are light enough that the difference
            # shows: they tremble at a standstill without it. Randomized
            # ABSOLUTE, so the policy has to hold still across the whole band
            # rather than learn one particular plant.
            "dr_joint_damping": EventTermConfig(
                func=unified_dr.randomize_joint_damping,
                mode="reset_dr",
                params={"asset_cfg": all_joints, "damping_range": (0.0, 1.0), "operation": "abs"},
            ),
            "dr_encoder_bias": EventTermConfig(
                func=unified_dr.randomize_encoder_bias,
                mode="reset_dr",
                params={"asset_cfg": all_joints, "bias_range": (-0.015, 0.015)},
            ),
            "dr_kp": EventTermConfig(
                func=unified_dr.randomize_pd_gains,
                mode="reset_dr",
                params={"asset_cfg": all_joints, "kp_range": (0.8, 1.2), "operation": "scale"},
            ),
            "dr_kd": EventTermConfig(
                func=unified_dr.randomize_pd_gains,
                mode="reset_dr",
                params={"asset_cfg": all_joints, "kd_range": (0.8, 1.2), "operation": "scale"},
            ),
        }

        # Re-sampling every reset costs a model recompute per episode; moving
        # the whole set onto one global timer keeps the variation and pays that
        # cost once per period instead.
        if self.dr_interval_period_s is not None:
            for term in terms.values():
                if term.mode == "reset_dr":
                    term.mode = "interval_dr"
                    term.interval_dr_period_s = self.dr_interval_period_s
        return terms

    # ── Curriculum ───────────────────────────────────────────────────

    def _build_curriculum_config(self) -> CurriculumManagerConfig:
        """The source's two step-staged schedules.

        The soft-landing penalty ramps up over training so early exploration is
        not punished for landing hard before it can land softly. The command
        envelope widens on a fixed step schedule; this framework also ships a
        performance-gated widener, but a fixed schedule is what the source runs
        and what parity requires.
        """
        soft_landing_stages = [
            {"step": 0, "weight": 0.0001},
            {"step": 1000 * _STEPS_PER_ROLLOUT, "weight": 0.001},
            {"step": 7000 * _STEPS_PER_ROLLOUT, "weight": 0.005},
        ]
        command_stages = [
            {"step": 0, "lin_vel_x_range": (-1.0, 1.2), "lin_vel_y_range": (-1.0, 1.0), "ang_vel_range": (-1.0, 1.0)},
            {
                "step": 5000 * _STEPS_PER_ROLLOUT,
                "lin_vel_x_range": (-1.0, 1.5),
                "lin_vel_y_range": (-1.25, 1.25),
                "ang_vel_range": (-1.25, 1.25),
            },
            {
                "step": 10000 * _STEPS_PER_ROLLOUT,
                "lin_vel_x_range": (-1.25, 1.5),
                "lin_vel_y_range": (-1.5, 1.5),
                "ang_vel_range": (-1.5, 1.5),
            },
            {
                "step": 15000 * _STEPS_PER_ROLLOUT,
                "lin_vel_x_range": (-1.5, 1.75),
                "lin_vel_y_range": (-1.75, 1.75),
                "ang_vel_range": (-1.5, 1.5),
            },
        ]

        @dataclass
        class _CurriculumCfg(CurriculumManagerConfig):
            soft_landing_weight: CurriculumTermConfig = field(
                default_factory=lambda: CurriculumTermConfig(
                    func=reward_curriculum,
                    params={"reward_name": "soft_landing", "stages": soft_landing_stages},
                )
            )
            command_envelope: CurriculumTermConfig = field(
                default_factory=lambda: CurriculumTermConfig(
                    func=command_envelope_curriculum,
                    params={"command_name": "velocity", "stages": command_stages},
                )
            )

        return _CurriculumCfg()

    # ── Training ─────────────────────────────────────────────────────

    # ── Motion prior ─────────────────────────────────────────────────

    _AMP_SPEED_AUGMENTATIONS = (1.1, 0.9, 1.2, 0.8)

    def _amp_motion_files(self) -> tuple[str, ...]:
        motion_files = tuple(sorted(str(p) for p in Path(self.amp_motion_dir).glob("*.npz")))
        if not motion_files:
            raise FileNotFoundError(
                f"no reference clips under {self.amp_motion_dir}; run " "jaxrlworld.scripts.k1.convert_lafan_k1"
            )
        return motion_files

    def _build_algorithm_config(self) -> PPOConfig:
        ppo = dict(
            clip_param=0.2,
            obs_normalization=True,
            entropy_coef=0.01,
            gamma=0.99,
            lam=0.95,
            actor_lr=1.0e-3,
            critic_lr=1.0e-3,
            max_grad_norm=1.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            schedule="adaptive",
            desired_kl=0.01,
            use_clipped_value_loss=True,
            value_loss_coef=1.0,
            symmetry_cfg=SymmetryConfig(
                use_mirror_loss=self.mirror_symmetry_coeff > 0.0,
                mirror_loss_coeff=self.mirror_symmetry_coeff,
                use_data_augmentation=self.symmetry_data_augmentation,
            ),
        )
        if not self.use_amp:
            return PPOConfig(algorithm_name=self.algorithm_name, **ppo)
        if self.algorithm_name != "AMP_PPO":
            raise ValueError(f"use_amp=True trains with AMP_PPO; algorithm_name is {self.algorithm_name!r}")
        motion_files = self._amp_motion_files()
        # The source's "Flat-Amp" recipe: BCE discriminator (256, 128) with
        # minibatch std and empirical normalization, style weight 0.3 (its
        # curriculum holds it there), 10-frame windows, mirror + speed
        # {+-10%, +-20%} expert augmentation, replay 200k / 1000 per rollout.
        return AmpPPOConfig(
            algorithm_name=self.algorithm_name,
            **ppo,
            amp=AmpConfig(
                motion_files=motion_files,
                root_body_name=self.robot.base_link_name,
                mirror_augmentation=True,
                speed_augmentations=self._AMP_SPEED_AUGMENTATIONS,
                amp_group="amp",
                num_amp_obs_steps=10,
                # Departure from the source: with the simulator's instantaneous
                # joint velocities the discriminator learned the velocity
                # texture (7x the clips' frame-to-frame jitter) instead of the
                # gait, a cue the policy cannot remove; see the config field.
                joint_velocity_from_positions=True,
                style_reward_weight=self.amp_style_reward_weight,
                discriminator_hidden_dims=(256, 128),
                loss_type="bce",
                grad_penalty_lambda=10.0,
            ),
        )

    def _build_nn_config(self) -> NNConfig:
        return NNConfig(
            policy=PPOPolicyConfig(
                actor=MLPActorCfg(
                    activation=Activation.ELU,
                    init=OrthoInit(output_gain=1.0),
                    hidden_dims=list(self.actor_hidden_dims),
                ),
                critic=MLPCriticCfg(
                    activation=Activation.ELU,
                    init=OrthoInit(output_gain=1.0),
                    hidden_dims=list(self.actor_hidden_dims),
                ),
                init_noise_std=1.0,
                distribution_type=DistributionType(self.action_distribution),
                std_type=StdType.STATE_INDEPENDENT,
            ),
        )

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
            wandb_project="K1_Booster_Velocity",
            save_interval=2000,
            output_dir="auto",
        )


class command_envelope_curriculum:
    """Widen the velocity command ranges on a fixed step schedule.

    The shared ``command_velocity_range`` term widens on a performance gate
    instead, which is a different schedule and would not reproduce the source.
    This applies the last stage whose ``step`` the global control-step counter
    has passed, writing the ranges straight onto the live command term config.
    """

    __name__ = "command_envelope_curriculum"

    _RANGE_KEYS = ("lin_vel_x_range", "lin_vel_y_range", "ang_vel_range")

    def __init__(self, env, cfg: CurriculumTermConfig) -> None:
        stages = cfg.params["stages"]
        if not stages or stages[0]["step"] != 0:
            raise ValueError("command envelope stages must start at step 0")
        steps = [stage["step"] for stage in stages]
        if steps != sorted(steps):
            raise ValueError(f"command envelope stages must be ordered by step, got {steps}")
        for stage in stages:
            missing = [k for k in self._RANGE_KEYS if k not in stage]
            if missing:
                raise ValueError(f"command envelope stage at step {stage['step']} is missing {missing}")
        self._stages = stages
        self._term = env.command_manager.get_term(cfg.params["command_name"])

    def __call__(self, env, env_ids, command_name: str, stages) -> dict:
        del env_ids, command_name, stages
        step = env.env_step_counter
        active = self._stages[0]
        for stage in self._stages:
            if step >= stage["step"]:
                active = stage
        for key in self._RANGE_KEYS:
            setattr(self._term.cfg, key, active[key])
        return {
            "lin_vel_x_max": active["lin_vel_x_range"][1],
            "lin_vel_y_max": active["lin_vel_y_range"][1],
            "ang_vel_max": active["ang_vel_range"][1],
        }
