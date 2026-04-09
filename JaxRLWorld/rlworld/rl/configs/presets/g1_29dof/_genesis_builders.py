"""Genesis-specific builders for G1 29-DOF flat-terrain locomotion.

These functions are dispatched from ``G1FlatConfig.build()`` when
``sim_type == "genesis"``. The bodies are extracted directly from the
old ``presets/g1_29dof/genesis/base.py`` so the produced configs are
identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import genesis as gs

from rlworld.rl.actuators import DelayedPDActuatorCfg
from rlworld.rl.configs.common_config_classes import (
    EventConfig,
    ObservationGroupConfig,
    RewardConfig,
    TerminationsConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.genesis_config_classes import (
    ActionConfig,
    CurriculumConfig,
    EnvConfig,
    GenesisConfigsForRun,
    GenesisContactSensorCfg,
    ObservationConfig,
    SceneConfig,
)
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.g1_29dof import G1_ACTION_SCALE
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    GenesisEntityCfg,
    GroundPlaneCfg,
    InitialStateCfg,
)
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.mdp.configs import TerminationTermConfig
from rlworld.rl.envs.mdp.events import event_terms as genesis_ef
from rlworld.rl.envs.mdp.events.dr import genesis as genesis_dr
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    base_height,
    base_quat,
    command as command_obs,
    dof_pos,
    dof_vel,
    foot_air_time,
    foot_contact_forces,
    foot_contact_indicator,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.observations.genesis import state  # state.feet_height (sim-specific)
from rlworld.rl.envs.mdp.reset import reset_terms as initf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.genesis import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf_genesis
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf

if TYPE_CHECKING:
    from .base import G1FlatConfig


# ── Module-level constants exposed to base.G1FlatConfig.build() ──────

CONFIGS_FOR_RUN_CLS = GenesisConfigsForRun
OBSERVATION_CFG_CLS = ObservationConfig


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: "G1FlatConfig") -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False)


def build_env(cfg: "G1FlatConfig", timing: Dict[str, Any]) -> EnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        roll_pitch_violation = TerminationTermConfig(
            common_tf.roll_pitch_violation,
            {"roll_threshold_degree": 70.0, "pitch_threshold_degree": 70.0},
        )
        time_out = TerminationTermConfig(max_episode_exceed)

    return EnvConfig(
        env_name="GenesisEnv",
        task_name="G1_Velocity_Tracking",
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        decimation=timing["decimation"],
        episode_length_s=cfg.episode_length_s,
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: "G1FlatConfig", timing: Dict[str, Any]) -> SceneConfig:
    r = cfg.robot
    sim_dt = timing["dt"]

    return SceneConfig(
        entities={
            "base_entity": GroundPlaneCfg(),
            "robot": GenesisEntityCfg(
                urdf_path=r.urdf_path,
                init_state=InitialStateCfg(
                    pos=(0, 0, r.base_init_height),
                    joint_pos=r.default_joint_angles,
                ),
                floating=True,
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
                convexify=True,
                visualize_contact=False,
            ),
        },
        sensors=[
            SensorConfig(
                entity_name="robot",
                link_name="pelvis",
                sensor_class=gs.sensors.IMU,
            ),
        ],
        contact_sensors=[
            GenesisContactSensorCfg(
                name="feet_ground_contact",
                primary_links=["left_ankle_roll_link", "right_ankle_roll_link"],
                secondary_entity=None,
                exclude_self_contact=True,
            ),
            GenesisContactSensorCfg(
                name="self_collision",
                primary_links=[".*"],
                entity_name="robot",
                secondary_entity="self",
            ),
        ],
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=timing["substeps"]),
        rigid_options=gs.options.RigidOptions(
            dt=sim_dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=30,
            batch_dofs_info=True,
        ),
        robot_cfg=r,
    )


def build_observation(cfg: "G1FlatConfig") -> ObservationConfig:
    feet_links = ("left_ankle_roll_link", "right_ankle_roll_link")

    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
        projected_gravity = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        dof_pos = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
        dof_vel = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)

    @dataclass
    class _CriticObsCfg(ObservationGroupConfig):
        base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
        projected_gravity = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        dof_pos = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        dof_vel = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
        base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)
        base_quat_obs = ObservationTermConfig(func=base_quat, scale=1.0)
        foot_height_obs = ObservationTermConfig(
            func=state.feet_height, scale=1.0, params={"links": feet_links}
        )
        foot_air_time_obs = ObservationTermConfig(func=foot_air_time, scale=1.0)
        foot_contact_obs = ObservationTermConfig(func=foot_contact_indicator, scale=1.0)
        foot_contact_forces_obs = ObservationTermConfig(func=foot_contact_forces, scale=0.01)

    @dataclass
    class _ObsCfg(ObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: "G1FlatConfig") -> ActionConfig:
    r = cfg.robot
    return ActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=G1_ACTION_SCALE,
        clip_actions=(-100.0, 100.0),
        offset=r.default_joint_angles,
    )


def build_reward(cfg: "G1FlatConfig") -> RewardConfig:
    r = cfg.robot
    feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]

    @dataclass
    class _RewardsCfg(RewardConfig):
        # Tracking rewards (common — uses RobotData interface)
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

        # Orientation (common — uses RobotData interface)
        flat_orientation = RewardTermConfig(
            func=rf_common.flat_orientation,
            weight=1.0,
            params={"std": 0.447},
        )

        # Posture
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
                    r".*waist_yaw.*": 0.2,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.1,
                    r".*shoulder_pitch.*": 0.15,
                    r".*shoulder_roll.*": 0.15,
                    r".*shoulder_yaw.*": 0.1,
                    r".*elbow.*": 0.15,
                    r".*wrist.*": 0.3,
                },
                "std_running": {
                    r".*hip_pitch.*": 0.5,
                    r".*hip_roll.*": 0.2,
                    r".*hip_yaw.*": 0.2,
                    r".*knee.*": 0.6,
                    r".*ankle_pitch.*": 0.35,
                    r".*ankle_roll.*": 0.15,
                    r".*waist_yaw.*": 0.3,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.2,
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
            func=rf_genesis.wtw_collision,
            weight=1.0,
            params={"contact_group": "self_collision", "force_threshold": 1.0},
        )

        # Penalties
        body_angular_velocity_penalty = RewardTermConfig(
            func=rf_mjlab.body_ang_vel_penalty_mjlab,
            weight=0.05,
            params={"body_name": r.base_link_name},
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
                "feet_links": feet_links,
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )
        feet_swing_height = RewardTermConfig(
            func=rf_mjlab.feet_swing_height_mjlab,
            weight=0.25,
            params={
                "feet_links": feet_links,
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )
        feet_slip = RewardTermConfig(
            func=rf_mjlab.feet_slip_mjlab,
            weight=0.1,
            params={
                "feet_links": feet_links,
                "command_threshold": 0.05,
            },
        )
        soft_landing = RewardTermConfig(
            func=rf_mjlab.soft_landing_mjlab,
            weight=1e-5,
            params={
                "command_threshold": 0.05,
                "contact_group": "feet_ground_contact",
            },
        )

    return _RewardsCfg()


def build_event(cfg: "G1FlatConfig") -> EventConfig:
    @dataclass
    class _EventsCfg(EventConfig):
        # Reset events
        reset_root = EventTermConfig(
            func=genesis_ef.reset_root_state_uniform,
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
            func=genesis_ef.push_by_setting_velocity,
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
            func=genesis_dr.randomize_body_com_offset,
            mode="reset_dr",
            params={
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
                "link_names": ("torso_link",),
            },
        )
        randomize_joint_friction = EventTermConfig(
            func=genesis_dr.randomize_joint_friction,
            mode="reset_dr",
            params={"friction_range": (0.0, 0.05)},
        )

    return _EventsCfg()


def build_curriculum(cfg: "G1FlatConfig") -> CurriculumConfig:
    """Genesis-only: dead curriculum (enable=False) preserved for compat."""
    return CurriculumConfig(
        enable=False,
        initial_level=0,
        max_level=3,
        success_threshold=0.8,
        min_steps_per_level=50000,
        eval_window_size=2,
        curriculum_components={},
        criterion={"tracking_lin_vel_xy": -100, "mean_return": -100},
    )
