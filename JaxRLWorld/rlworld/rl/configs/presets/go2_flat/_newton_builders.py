"""Newton-specific builders for Go2 flat-terrain locomotion.

These functions are dispatched from ``Go2FlatConfig.build()`` when
``sim_type == "newton"``. The bodies are extracted directly from the
old ``presets/go2_flat/newton/base.py`` so the produced configs are
identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import warp as wp

from rlworld.rl.actuators import DelayedPDActuatorCfg
from rlworld.rl.configs import RewardConfig, EventConfig
from rlworld.rl.configs.common_config_classes import TerminationsConfig
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonConfigsForRun,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    VisualizationConfig,
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
    GroundPlaneCfg,
    InitialStateCfg,
    NewtonEntityCfg as UnifiedNewtonEntityCfg,
)
from rlworld.rl.configs.sensors import NewtonContactSensorConfig, NewtonIMUSensorConfig
from rlworld.rl.envs.mdp.configs import TerminationTermConfig
from rlworld.rl.envs.mdp.reset import newton_reset_terms as initf
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf

if TYPE_CHECKING:
    from .base import Go2FlatConfig


# ── Module-level constants exposed to base.Go2FlatConfig.build() ─────

CONFIGS_FOR_RUN_CLS = NewtonConfigsForRun
OBSERVATION_CFG_CLS = NewtonObservationConfig


def get_foot_names(robot) -> tuple[str, ...]:
    """Newton uses prefixed foot names (e.g. ``go2_description/FL_foot``)."""
    return robot.prefixed_foot_names


def _initial_quat() -> Any:
    """Initial yaw of the robot at reset (90° about world Z)."""
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: "Go2FlatConfig") -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: "Go2FlatConfig", timing: Dict[str, Any]) -> NewtonEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        roll_pitch = TerminationTermConfig(
            common_tf.roll_pitch_violation,
            {"roll_threshold_degree": 30.0, "pitch_threshold_degree": 30.0},
        )
        max_episode = TerminationTermConfig(max_episode_exceed)

    return NewtonEnvConfig(
        env_name="NewtonLocomotionEnv",
        num_envs=cfg.num_envs,
        task_name="Go2 Velocity Tracking",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: "Go2FlatConfig", timing: Dict[str, Any]) -> NewtonSceneConfig:
    r = cfg.robot
    quat = _initial_quat()

    return NewtonSceneConfig(
        dt=timing["dt"],
        substeps=timing["substeps"],
        gravity=(0.0, 0.0, -9.81),
        solver_type="mujoco",
        entities={
            "ground": GroundPlaneCfg(),
            "robot": UnifiedNewtonEntityCfg(
                urdf_path=r.urdf_path,
                init_state=InitialStateCfg(
                    pos=(0.0, 0.0, r.base_init_height),
                    rot=(quat[0], quat[1], quat[2], quat[3]),
                ),
                floating=True,
                collapse_fixed_joints=True,
                links_to_keep=[
                    "go2_description/FL_foot_joint",
                    "go2_description/FR_foot_joint",
                    "go2_description/RL_foot_joint",
                    "go2_description/RR_foot_joint",
                ],
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
                sensing_obj_bodies=list(r.prefixed_foot_names),
            ),
            NewtonContactSensorConfig(
                entity_name="robot",
                sensor_name="body_ground_contact",
                sensing_obj_bodies=["*"],
                exclude_bodies=("*foot*",),
            ),
        ],
        add_ground=True,
        env_spacing=(2.0, 2.0, 0.0),
        robot_cfg=r,
    )


def build_action(cfg: "Go2FlatConfig") -> NewtonActionConfig:
    r = cfg.robot
    return NewtonActionConfig(
        actuated_dof_names=r.prefixed_actuated_dof_patterns,
        action_scale=GO2_ACTION_SCALE,
        clip_actions=(-100.0, 100.0),
        offset=r.get_prefixed_action_offset(),
    )


def build_reward(cfg: "Go2FlatConfig") -> RewardConfig:
    r = cfg.robot
    feet = list(r.prefixed_foot_names)

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
                "feet_bodies": feet,
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )
        feet_clearance = RewardTermConfig(
            func=rf_mjlab.feet_clearance_mjlab,
            weight=2.0,
            params={
                "feet_bodies": feet,
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )
        feet_slip = RewardTermConfig(
            func=rf_mjlab.feet_slip_mjlab,
            weight=0.1,
            params={
                "feet_bodies": feet,
                "command_threshold": 0.05,
            },
        )
        soft_landing = RewardTermConfig(
            func=rf_mjlab.soft_landing_mjlab,
            weight=1e-5,
            params={
                "feet_bodies": feet,
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
    r = cfg.robot
    from rlworld.rl.envs.mdp.events.dr import newton as newton_dr
    from rlworld.rl.envs.mdp.events.common_event_terms import (
        push_by_setting_velocity as _push_fn,
        reset_root_state_uniform as _reset_root_fn,
    )

    # Newton stores initial quat in xyzw; convert to wxyz for common API.
    _iq = _initial_quat()  # wp.quat (xyzw: x, y, z, w)
    _iq_tuple = tuple(float(v) for v in _iq)  # (x, y, z, w)
    _default_quat_wxyz = (_iq_tuple[3], _iq_tuple[0], _iq_tuple[1], _iq_tuple[2])

    @dataclass
    class _EventsCfg(EventConfig):
        reset_root = EventTermConfig(
            func=_reset_root_fn,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
                "default_pos": (0.0, 0.0, r.base_init_height),
                "default_quat_wxyz": _default_quat_wxyz,
            },
        )
        reset_dof_pos = EventTermConfig(
            func=initf.initialize_dof_pos_with_noise,
            params={"position_noise_range": (math.pi / 360, math.pi / 120)},
            mode="reset",
        )
        # Domain randomization (disabled during eval)
        randomize_body_mass = EventTermConfig(
            func=newton_dr.randomize_body_mass,
            params={
                "mass_range": (0.8, 1.2),
                "operation": "scale",
                "body_patterns": r.prefixed("base"),
            },
            mode="reset_dr",
        )
        randomize_friction = EventTermConfig(
            func=newton_dr.randomize_friction,
            mode="reset_dr",
            params={"friction_range": (0.3, 1.2)},
        )
        randomize_joint_friction = EventTermConfig(
            func=newton_dr.randomize_joint_friction,
            mode="reset_dr",
            params={"friction_range": (0.0, 0.05)},
        )
        push_robot = EventTermConfig(
            func=_push_fn,
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
