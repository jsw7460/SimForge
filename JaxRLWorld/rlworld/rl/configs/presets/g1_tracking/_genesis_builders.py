"""Genesis-specific builders for G1 motion tracking task.

Dispatched from :meth:`G1TrackingConfig.build` when
``sim_type == "genesis"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import genesis as gs

from rlworld.rl.actuators import IdealPDActuatorCfg
from rlworld.rl.configs.common_config_classes import (
    ObservationGroupConfig,
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
from rlworld.rl.configs import TerminationTermConfig
from rlworld.rl.envs.mdp.events.dr import genesis as genesis_dr
from rlworld.rl.envs.mdp.observations.common.motion_tracking import (
    motion_anchor_ori_b,
    motion_anchor_pos_b,
    robot_body_ori_b,
    robot_body_pos_b,
)
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    base_height,
    base_lin_vel,
    base_quat,
    command as command_obs,
    dof_pos,
    dof_vel,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import motion_tracking as rf_motion
from rlworld.rl.envs.mdp.rewards.genesis import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf_genesis
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import motion_tracking as tt_motion

if TYPE_CHECKING:
    from .base import G1TrackingConfig


CONFIGS_FOR_RUN_CLS = GenesisConfigsForRun
OBSERVATION_CFG_CLS = ObservationConfig


def build_visualization(cfg: "G1TrackingConfig") -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False)


def build_env(cfg: "G1TrackingConfig", timing: Dict[str, Any]) -> EnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        time_out = TerminationTermConfig(max_episode_exceed)
        bad_anchor_pos = TerminationTermConfig(
            tt_motion.bad_anchor_pos_z_only,
            {
                "command_name": "motion",
                "threshold": cfg.bad_anchor_pos_z_threshold,
            },
        )
        bad_anchor_ori = TerminationTermConfig(
            tt_motion.bad_anchor_ori,
            {
                "command_name": "motion",
                "threshold": cfg.bad_anchor_ori_threshold,
            },
        )
        bad_ee_pos = TerminationTermConfig(
            tt_motion.bad_motion_body_pos_z_only,
            {
                "command_name": "motion",
                "threshold": cfg.bad_motion_body_pos_z_threshold,
                "body_names": cfg.ee_body_names,
            },
        )

    return EnvConfig(
        env_name="GenesisEnv",
        task_name="G1_Tracking",
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        decimation=timing["decimation"],
        episode_length_s=cfg.episode_length_s,
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: "G1TrackingConfig", timing: Dict[str, Any]) -> SceneConfig:
    r = cfg.robot
    sim_dt = timing["dt"]

    return SceneConfig(
        entities={
            "base_entity": GroundPlaneCfg(),
            "robot": GenesisEntityCfg(
                mjcf_path=r.mjcf_path,
                init_state=InitialStateCfg(
                    pos=(0, 0, r.base_init_height),
                    joint_pos=r.default_joint_angles,
                ),
                floating=True,
                articulation=ArticulationCfg(
                    actuators=(
                        IdealPDActuatorCfg(
                            target_names_expr=(".*",),
                            stiffness=r.p_gains,
                            damping=r.d_gains,
                            armature=r.armature,
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
            constraint_timeconst=0.02,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=30,
            batch_dofs_info=True,
        ),
        robot_cfg=r,
    )


def build_observation(cfg: "G1TrackingConfig") -> ObservationConfig:
    motion_params = {"command_name": "motion"}

    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(
            func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2)
        )
        projected_gravity_obs = ObservationTermConfig(
            func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05)
        )
        base_lin_vel_obs = ObservationTermConfig(
            func=base_lin_vel, scale=1.0, noise=Unoise(-0.5, 0.5)
        )
        dof_pos_obs = ObservationTermConfig(
            func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01)
        )
        dof_vel_obs = ObservationTermConfig(
            func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5)
        )
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        motion_anchor_pos = ObservationTermConfig(
            func=motion_anchor_pos_b, scale=1.0, params=motion_params,
            noise=Unoise(-0.25, 0.25),
        )
        motion_anchor_ori = ObservationTermConfig(
            func=motion_anchor_ori_b, scale=1.0, params=motion_params,
            noise=Unoise(-0.05, 0.05),
        )

    @dataclass
    class _CriticObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(func=base_ang_vel, scale=1.0)
        base_lin_vel_obs = ObservationTermConfig(func=base_lin_vel, scale=1.0)
        projected_gravity_obs = ObservationTermConfig(func=projected_gravity, scale=1.0)
        dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0)
        dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=1.0)
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)
        base_quat_obs = ObservationTermConfig(func=base_quat, scale=1.0)
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        motion_anchor_pos = ObservationTermConfig(
            func=motion_anchor_pos_b, scale=1.0, params=motion_params,
        )
        motion_anchor_ori = ObservationTermConfig(
            func=motion_anchor_ori_b, scale=1.0, params=motion_params,
        )
        robot_body_pos = ObservationTermConfig(
            func=robot_body_pos_b, scale=1.0, params=motion_params,
        )
        robot_body_ori = ObservationTermConfig(
            func=robot_body_ori_b, scale=1.0, params=motion_params,
        )

    @dataclass
    class _ObsCfg(ObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: "G1TrackingConfig") -> ActionConfig:
    """default_pos + action * per-joint-scale (Mjlab-equivalent JointPositionAction)."""
    r = cfg.robot
    return ActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=G1_ACTION_SCALE,
        clip_actions=(-100.0, 100.0),
        offset=r.default_joint_angles,
    )


def build_reward(cfg: "G1TrackingConfig") -> RewardConfig:
    motion_params_std = lambda std: {"command_name": "motion", "std": std}

    @dataclass
    class _RewardsCfg(RewardConfig):
        motion_anchor_pos = RewardTermConfig(
            func=rf_motion.motion_global_anchor_position_error_exp,
            weight=cfg.anchor_pos_weight,
            params=motion_params_std(cfg.anchor_pos_std),
        )
        motion_anchor_ori = RewardTermConfig(
            func=rf_motion.motion_global_anchor_orientation_error_exp,
            weight=cfg.anchor_ori_weight,
            params=motion_params_std(cfg.anchor_ori_std),
        )
        motion_body_pos = RewardTermConfig(
            func=rf_motion.motion_relative_body_position_error_exp,
            weight=cfg.body_pos_weight,
            params=motion_params_std(cfg.body_pos_std),
        )
        motion_body_ori = RewardTermConfig(
            func=rf_motion.motion_relative_body_orientation_error_exp,
            weight=cfg.body_ori_weight,
            params=motion_params_std(cfg.body_ori_std),
        )
        motion_body_lin_vel = RewardTermConfig(
            func=rf_motion.motion_global_body_linear_velocity_error_exp,
            weight=cfg.body_lin_vel_weight,
            params=motion_params_std(cfg.body_lin_vel_std),
        )
        motion_body_ang_vel = RewardTermConfig(
            func=rf_motion.motion_global_body_angular_velocity_error_exp,
            weight=cfg.body_ang_vel_weight,
            params=motion_params_std(cfg.body_ang_vel_std),
        )
        raw_action_rate_l2 = RewardTermConfig(
            func=rf_mjlab.raw_action_rate_l2_mjlab,
            weight=cfg.action_rate_l2_weight,
        )
        joint_pos_limits = RewardTermConfig(
            func=rf_mjlab.joint_pos_limits_mjlab,
            weight=cfg.joint_pos_limits_weight,
        )
        self_collision_cost = RewardTermConfig(
            func=rf_genesis.wtw_collision,
            weight=cfg.self_collision_weight,
            params={"contact_group": "self_collision", "force_threshold": 10.0},
        )

    return _RewardsCfg()


def build_dr_terms(cfg: "G1TrackingConfig") -> Dict[str, EventTermConfig]:
    """Genesis DR — scalar friction randomization only.

    Genesis lacks the 3-axis geom friction that Mjlab uses for G1 tracking
    (slide/spin/roll); fall back to scalar friction range matching the
    slide axis.
    """
    return {
        "randomize_friction_scalar": EventTermConfig(
            func=genesis_dr.randomize_friction,
            mode="reset_dr",
            params={"friction_range": (0.3, 1.2)},
        ),
    }
