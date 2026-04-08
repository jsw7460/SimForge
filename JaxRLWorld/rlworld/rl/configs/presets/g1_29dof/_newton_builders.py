"""Newton-specific builders for G1 29-DOF flat-terrain locomotion.

These functions are dispatched from ``G1FlatConfig.build()`` when
``sim_type == "newton"``. The bodies are extracted directly from the
old ``presets/g1_29dof/newton/base.py`` so the produced configs are
identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import warp as wp

from rlworld.rl.actuators import DelayedPDActuatorCfg
from rlworld.rl.configs import EventConfig, RewardConfig
from rlworld.rl.configs.common_config_classes import (
    ObservationGroupConfig,
    TerminationsConfig,
)
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonConfigsForRun,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    GroundPlaneCfg,
    InitialStateCfg,
    NewtonEntityCfg,
)
from rlworld.rl.configs.sensors import NewtonContactSensorConfig, NewtonIMUSensorConfig
from rlworld.rl.envs.mdp.configs import TerminationTermConfig
from rlworld.rl.envs.mdp.events import newton_event_terms as newton_ef
from rlworld.rl.envs.mdp.events.dr import newton as newton_dr
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    dof_pos,
    dof_vel,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.observations.genesis.exteroception import command as command_obs
from rlworld.rl.envs.mdp.observations.newton import state
from rlworld.rl.envs.mdp.reset import newton_reset_terms as initf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.rewards.newton import reward_terms as rf_newton
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf

if TYPE_CHECKING:
    from .base import G1FlatConfig


# ── Module-level constants exposed to base.G1FlatConfig.build() ──────

CONFIGS_FOR_RUN_CLS = NewtonConfigsForRun
OBSERVATION_CFG_CLS = NewtonObservationConfig


def _initial_quat() -> Any:
    """Initial yaw of the robot at reset (no rotation for G1)."""
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.0)


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: "G1FlatConfig") -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: "G1FlatConfig", timing: Dict[str, Any]) -> NewtonEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        roll_pitch = TerminationTermConfig(
            common_tf.roll_pitch_violation,
            {"roll_threshold_degree": 70.0, "pitch_threshold_degree": 70.0},
        )
        max_episode = TerminationTermConfig(max_episode_exceed)

    return NewtonEnvConfig(
        num_envs=cfg.num_envs,
        env_name="NewtonEnv",
        task_name="G1_12Dof_Velocity_Tracking",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: "G1FlatConfig", timing: Dict[str, Any]) -> NewtonSceneConfig:
    r = cfg.robot
    quat = _initial_quat()

    return NewtonSceneConfig(
        dt=timing["dt"],
        substeps=timing["substeps"],
        gravity=(0.0, 0.0, -9.81),
        solver_type="mujoco",
        robot_cfg=r,
        entities={
            "ground": GroundPlaneCfg(),
            "robot": NewtonEntityCfg(
                urdf_path=r.urdf_path,
                init_state=InitialStateCfg(
                    pos=(0.0, 0.0, r.base_init_height),
                    rot=(quat[0], quat[1], quat[2], quat[3]),
                    joint_pos=r.default_joint_angles,
                ),
                floating=True,
                collapse_fixed_joints=True,
                articulation=ArticulationCfg(
                    actuators=(
                        DelayedPDActuatorCfg(
                            target_names_expr=(".*",),
                            stiffness=r.p_gains,
                            damping=r.d_gains,
                            armature=r.armature,
                            min_delay=0,
                            max_delay=2,
                        ),
                    ),
                ),
                body_label_prefix=r.name,
                sites={"imu_site_base": r.base_link_name},
            ),
        },
        sensors=[
            NewtonIMUSensorConfig(
                entity_name="robot",
                sensor_name="imu_base",
                site_names=["imu_site_base"],
            ),
            NewtonContactSensorConfig(
                entity_name="robot",
                sensor_name="foot_contact",
                sensing_obj_bodies=r.foot_names,
                counterpart_shapes="ground_plane",
                use_regex=True,
                include_total=False,
            ),
            NewtonContactSensorConfig(
                entity_name="robot",
                sensor_name="self_collision",
                sensing_obj_bodies=["*"],
                counterpart_bodies=["*"],
                include_total=False,
            ),
        ],
        add_ground=True,
        env_spacing=(2.0, 2.0, 0.0),
    )


def build_observation(cfg: "G1FlatConfig") -> NewtonObservationConfig:
    feet_bodies = tuple(cfg.robot.prefixed_foot_names)

    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
        projected_gravity_obs = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
        dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)

    @dataclass
    class _CriticObsCfg(ObservationGroupConfig):
        base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
        projected_gravity = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        dof_pos = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        dof_vel = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
        base_height_obs = ObservationTermConfig(func=state.base_height, scale=1.0)
        base_quat_obs = ObservationTermConfig(func=state.base_quat, scale=1.0)
        foot_height_obs = ObservationTermConfig(
            func=state.feet_height, scale=1.0, params={"feet_bodies": feet_bodies}
        )
        foot_air_time_obs = ObservationTermConfig(
            func=state.feet_air_time, scale=1.0, params={"feet_bodies": feet_bodies}
        )
        foot_contact_obs = ObservationTermConfig(
            func=state.feet_contact_indicator, scale=1.0, params={"feet_bodies": feet_bodies}
        )
        foot_contact_forces_obs = ObservationTermConfig(
            func=state.feet_contact_force_3d, scale=0.01, params={"feet_bodies": feet_bodies}
        )

    @dataclass
    class _ObsCfg(NewtonObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: "G1FlatConfig") -> NewtonActionConfig:
    r = cfg.robot
    return NewtonActionConfig(
        actuated_dof_names=r.prefixed_actuated_dof_patterns,
        action_scale=r.prefixed_action_scale,
        clip_actions=(-100.0, 100.0),
        offset=r.get_prefixed_action_offset(),
    )


def build_reward(cfg: "G1FlatConfig") -> RewardConfig:
    r = cfg.robot

    @dataclass
    class _RewardsCfg(RewardConfig):
        # Tracking rewards (common -- uses RobotData interface)
        track_lin_vel = RewardTermConfig(
            func=rf_common.track_lin_vel,
            weight=2.0,
            params={"std": 0.5, "penalize_z": True},
        )
        track_ang_vel = RewardTermConfig(
            func=rf_common.track_ang_vel,
            weight=2.0,
            params={"std": 0.707, "penalize_xy": True},
        )

        # Orientation (common -- uses RobotData interface)
        flat_orientation = RewardTermConfig(
            func=rf_common.flat_orientation,
            weight=1.0,
            params={"std": 0.447},
        )

        # Posture (stateful class)
        variable_posture = RewardTermConfig(
            func=rf_mjlab.variable_posture,
            weight=1.0,
            params={
                "std_standing": {".*": 0.05},
                "std_walking": {
                    r".*hip_pitch.*": 0.3,
                    r".*hip_roll.*": 0.15,
                    r".*hip_yaw.*": 0.15,
                    r".*knee.*": 0.35,
                    r".*ankle_pitch.*": 0.25,
                    r".*ankle_roll.*": 0.1,
                    # Waist.
                    r".*waist_yaw.*": 0.2,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.1,
                    # Arms.
                    r".*shoulder_pitch.*": 0.15,
                    r".*shoulder_roll.*": 0.15,
                    r".*shoulder_yaw.*": 0.1,
                    r".*elbow.*": 0.15,
                    r".*wrist.*": 0.3,
                },
                "std_running": {
                    # Lower body.
                    r".*hip_pitch.*": 0.5,
                    r".*hip_roll.*": 0.2,
                    r".*hip_yaw.*": 0.2,
                    r".*knee.*": 0.6,
                    r".*ankle_pitch.*": 0.35,
                    r".*ankle_roll.*": 0.15,
                    # Waist.
                    r".*waist_yaw.*": 0.3,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.2,
                    # Arms.
                    r".*shoulder_pitch.*": 0.5,
                    r".*shoulder_roll.*": 0.2,
                    r".*shoulder_yaw.*": 0.15,
                    r".*elbow.*": 0.35,
                    r".*wrist.*": 0.3,
                },
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        )

        # Self-collision
        self_collision_cost = RewardTermConfig(
            func=rf_newton.wtw_collision,
            weight=1.0,
            params={"contact_group": "self_collision", "force_threshold": 10.0},
        )

        # Penalties
        body_angular_velocity_penalty = RewardTermConfig(
            func=rf_mjlab.body_ang_vel_penalty_mjlab,
            weight=0.05,
            params={"body_name": r.prefixed("torso_link")},
        )
        angular_momentum_penalty = RewardTermConfig(
            func=rf_mjlab.angular_momentum_penalty,
            weight=0.02,
        )
        joint_pos_limits = RewardTermConfig(
            func=rf_mjlab.joint_pos_limits_mjlab,
            weight=1.0,
        )
        raw_action_rate_l2 = RewardTermConfig(
            func=rf_mjlab.raw_action_rate_l2_mjlab,
            weight=0.1,
        )

        # Feet rewards
        feet_clearance = RewardTermConfig(
            func=rf_mjlab.feet_clearance_mjlab,
            weight=2.0,
            params={
                "feet_bodies": r.prefixed_foot_names,
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )
        feet_swing_height = RewardTermConfig(
            func=rf_mjlab.feet_swing_height_mjlab,
            weight=0.25,
            params={
                "feet_bodies": r.prefixed_foot_names,
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )
        feet_slip = RewardTermConfig(
            func=rf_mjlab.feet_slip_mjlab,
            weight=0.1,
            params={
                "feet_bodies": r.prefixed_foot_names,
                "command_threshold": 0.05,
            },
        )
        soft_landing = RewardTermConfig(
            func=rf_mjlab.soft_landing_mjlab,
            weight=1e-5,
            params={
                "feet_bodies": r.prefixed_foot_names,
                "command_threshold": 0.05,
            },
        )

    return _RewardsCfg()


def build_event(cfg: "G1FlatConfig") -> EventConfig:
    r = cfg.robot

    @dataclass
    class _EventsCfg(EventConfig):
        # Reset events
        reset_root = EventTermConfig(
            func=newton_ef.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.01, 0.05),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        )
        reset_dof_pos = EventTermConfig(
            func=initf.initialize_dof_pos_with_noise,
            mode="reset",
            params={
                "position_noise_range": (0.0, 0.0),
                "velocity_noise_range": (0.0, 0.0),
            },
        )

        # Interval events
        push_robot = EventTermConfig(
            func=newton_ef.push_robot,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        )

        # Domain randomization (disabled during eval)
        randomize_body_com = EventTermConfig(
            func=newton_dr.randomize_body_com_offset,
            mode="reset_dr",
            params={
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
                "body_patterns": (r.prefixed("torso_link"),),
            },
        )
        randomize_joint_friction = EventTermConfig(
            func=newton_dr.randomize_joint_friction,
            mode="reset_dr",
            params={"friction_range": (0.0, 0.05)},
        )

    return _EventsCfg()
