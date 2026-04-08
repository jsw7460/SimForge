"""Genesis-specific builders for Go2 flat-terrain locomotion.

These functions are dispatched from ``Go2FlatConfig.build()`` when
``sim_type == "genesis"``. The bodies are extracted directly from the
old ``presets/go2_flat/genesis/base.py`` so the produced configs are
identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import genesis as gs

from rlworld.rl.actuators import DelayedPDActuatorCfg
from rlworld.rl.configs.common_config_classes import (
    CurriculumConfig,
    EventConfig,
    RewardConfig,
    TerminationsConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.genesis_config_classes import (
    ActionConfig,
    EnvConfig,
    GenesisConfigsForRun,
    GenesisContactSensorCfg,
    ObservationConfig,
    SceneConfig,
)
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.robots.go2 import (
    ARMATURE_HIP,
    ARMATURE_KNEE,
    DAMPING_HIP,
    DAMPING_KNEE,
    EFFORT_HIP,
    EFFORT_KNEE,
    GO2_ACTION_SCALE,
    STIFFNESS_HIP,
    STIFFNESS_KNEE,
)
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    GenesisEntityCfg,
    GroundPlaneCfg,
    InitialStateCfg,
)
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.mdp.configs import TerminationTermConfig
from rlworld.rl.envs.mdp.events import event_terms as ef
from rlworld.rl.envs.mdp.events.dr import genesis as genesis_dr
from rlworld.rl.envs.mdp.reset import reset_terms as initf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.genesis import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf

if TYPE_CHECKING:
    from .base import Go2FlatConfig


# ── Module-level constants exposed to base.Go2FlatConfig.build() ─────

CONFIGS_FOR_RUN_CLS = GenesisConfigsForRun
OBSERVATION_CFG_CLS = ObservationConfig


def get_foot_names(robot) -> tuple[str, ...]:
    """Genesis uses bare foot names (e.g. ``FL_foot``)."""
    return robot.foot_names


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: "Go2FlatConfig") -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False)


def build_env(cfg: "Go2FlatConfig", timing: Dict[str, Any]) -> EnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        roll_pitch_violation = TerminationTermConfig(
            common_tf.roll_pitch_violation,
            {"roll_threshold_degree": 30.0, "pitch_threshold_degree": 30.0},
        )
        time_out = TerminationTermConfig(max_episode_exceed)

    return EnvConfig(
        env_name="GenesisLocomotionEnv",
        task_name="Go2_Locomotion",
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        decimation=timing["decimation"],
        episode_length_s=cfg.episode_length_s,
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: "Go2FlatConfig", timing: Dict[str, Any]) -> SceneConfig:
    r = cfg.robot
    sim_dt = timing["dt"]

    return SceneConfig(
        env_spacing=(20.0, 20.0),
        entities={
            "base_entity": GroundPlaneCfg(),
            "robot": GenesisEntityCfg(
                urdf_path=r.urdf_path,
                init_state=InitialStateCfg(
                    pos=(1.5, 1.5, r.base_init_height),
                    joint_pos=r.default_joint_angles,
                ),
                floating=True,
                links_to_keep=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                articulation=ArticulationCfg(
                    actuators=(
                        DelayedPDActuatorCfg(
                            target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                            stiffness=STIFFNESS_HIP,
                            damping=DAMPING_HIP,
                            effort_limit=EFFORT_HIP,
                            armature=ARMATURE_HIP,
                            min_delay=1,
                            max_delay=3,
                        ),
                        DelayedPDActuatorCfg(
                            target_names_expr=(".*_calf_joint",),
                            stiffness=STIFFNESS_KNEE,
                            damping=DAMPING_KNEE,
                            effort_limit=EFFORT_KNEE,
                            armature=ARMATURE_KNEE,
                            min_delay=1,
                            max_delay=3,
                        ),
                    ),
                ),
                convexify=False,
                visualize_contact=True,
            ),
        },
        sensors=[
            SensorConfig(entity_name="robot", link_name="base", sensor_class=gs.sensors.IMU),
        ],
        contact_sensors=[
            GenesisContactSensorCfg(
                name="feet_ground_contact",
                primary_links=r.foot_names,
                secondary_entity="base_entity",
            ),
            GenesisContactSensorCfg(
                name="body_ground_contact",
                primary_links=[".*"],
                exclude_links=(".*foot.*",),
                entity_name="robot",
                exclude_self_contact=False,
                secondary_entity=None,
            ),
        ],
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=timing["substeps"]),
        rigid_options=gs.options.RigidOptions(
            dt=sim_dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            batch_dofs_info=True,
        ),
        robot_cfg=r,
    )


def build_action(cfg: "Go2FlatConfig") -> ActionConfig:
    r = cfg.robot
    return ActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=GO2_ACTION_SCALE,
        clip_actions=(-100.0, 100.0),
        offset=r.get_action_offset(),
    )


def build_reward(cfg: "Go2FlatConfig") -> RewardConfig:
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

        # Posture reward (stateful class)
        variable_posture = RewardTermConfig(
            func=rf_mjlab.variable_posture,
            weight=1.0,
            params={
                "std_standing": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
                },
                "std_walking": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "std_running": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        )

        feet_swing_height = RewardTermConfig(
            func=rf_mjlab.feet_swing_height_mjlab,
            weight=0.25,
            params={
                "feet_links": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )

        feet_clearance = RewardTermConfig(
            func=rf_mjlab.feet_clearance_mjlab,
            weight=2.0,
            params={
                "feet_links": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )

        feet_slip = RewardTermConfig(
            func=rf_mjlab.feet_slip_mjlab,
            weight=0.1,
            params={
                "feet_links": ["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
                "command_threshold": 0.05,
            },
        )

        soft_landing = RewardTermConfig(
            func=rf_mjlab.soft_landing_mjlab,
            weight=1e-5,
            params={
                "command_threshold": 0.05,
            },
        )

        joint_pos_limits = RewardTermConfig(
            func=rf_mjlab.joint_pos_limits_mjlab,
            weight=1.0,
            params={"soft_limit_factor": 1.0},
        )

        raw_action_rate_l2 = RewardTermConfig(
            func=rf_mjlab.raw_action_rate_l2_mjlab,
            weight=0.1,
        )

    return _RewardsCfg()


def build_event(cfg: "Go2FlatConfig") -> EventConfig:
    @dataclass
    class _EventsCfg(EventConfig):
        reset_root = EventTermConfig(
            func=ef.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        )

        reset_dof_pos = EventTermConfig(
            func=initf.initialize_dof_pos_with_noise,
            mode="reset",
            params={"position_noise_range": (math.pi / 360, math.pi / 120)},
        )

        # Domain randomization (disabled during eval)
        randomize_base_mass = EventTermConfig(
            func=genesis_dr.randomize_body_mass,
            mode="reset_dr",
            params={"mass_ratio_range": (0.8, 1.2)},
        )
        randomize_friction = EventTermConfig(
            func=genesis_dr.randomize_friction,
            mode="reset_dr",
            params={"friction_range": (0.3, 1.2)},
        )
        randomize_joint_friction = EventTermConfig(
            func=genesis_dr.randomize_joint_friction,
            mode="reset_dr",
            params={"friction_range": (0.0, 0.05)},
        )
        # Interval terms
        push_robot = EventTermConfig(
            func=ef.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 20.0),
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

    return _EventsCfg()


def build_curriculum(cfg: "Go2FlatConfig") -> CurriculumConfig:
    """Genesis-only: dead curriculum (enable=False) preserved for compat."""
    return CurriculumConfig(
        enable=False,
        initial_level=0,
        max_level=3,
        success_threshold=0.8,
        min_steps_per_level=50000,
        eval_window_size=2,
        curriculum_components={},
        criterion={
            "tracking_lin_vel_xy": -100,
            "mean_return": -100,
        },
    )
