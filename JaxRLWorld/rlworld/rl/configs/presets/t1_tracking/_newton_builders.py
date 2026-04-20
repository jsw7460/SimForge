"""Newton-specific builders for T1 motion tracking task.

Dispatched from :meth:`T1TrackingConfig.build` when
``sim_type == "newton"``. Mirrors ``t1_getup/_newton_builders.py``
structure but swaps in the motion-tracking MDP terms.

Newton resolves body names with the entity prefix baked in, so
``BODY_NAME_PREFIX = "T1/"`` is fed into MotionCommandCfg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import warp as wp

from rlworld.rl.actuators import ImplicitActuatorCfg
from rlworld.rl.configs import RewardConfig
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
from rlworld.rl.envs.mdp.events.dr import newton as newton_dr
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
    dof_pos_nominal_difference,
    dof_vel,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import motion_tracking as rf_motion
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.rewards.newton import reward_terms as rf_newton
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import motion_tracking as tt_motion

if TYPE_CHECKING:
    from .base import T1TrackingConfig


# ── Module-level constants exposed to T1TrackingConfig.build() ──────

CONFIGS_FOR_RUN_CLS = NewtonConfigsForRun
OBSERVATION_CFG_CLS = NewtonObservationConfig

# Newton resolves body names against ``<entity_name>/<body>`` labels.
BODY_NAME_PREFIX = "T1/"


def _initial_quat() -> Any:
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.0)


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: "T1TrackingConfig") -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: "T1TrackingConfig", timing: Dict[str, Any]) -> NewtonEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        max_episode = TerminationTermConfig(max_episode_exceed)
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

    return NewtonEnvConfig(
        num_envs=cfg.num_envs,
        env_name="NewtonEnv",
        task_name="T1_Tracking",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: "T1TrackingConfig", timing: Dict[str, Any]) -> NewtonSceneConfig:
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
                mjcf_path=r.mjcf_path,
                init_state=InitialStateCfg(
                    pos=(0.0, 0.0, r.base_init_height),
                    rot=(quat[0], quat[1], quat[2], quat[3]),
                    joint_pos=r.default_joint_angles,
                ),
                floating=True,
                collapse_fixed_joints=True,
                articulation=ArticulationCfg(
                    actuators=(
                        ImplicitActuatorCfg(
                            target_names_expr=(".*",),
                            stiffness=r.p_gains,
                            damping=r.d_gains,
                            armature=r.armature,
                            effort_limit=r.effort_limits,
                        ),
                    ),
                ),
                body_label_prefix=r.name,
                sites={"imu_site_base": f".*{r.base_link_name}"},
                enable_self_collisions=True,
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
                sensor_name="self_collision",
                sensing_obj_bodies=["*"],
                counterpart_bodies=["*"],
                include_total=False,
            ),
        ],
        add_ground=True,
        env_spacing=(2.0, 2.0, 0.0),
    )


def build_observation(cfg: "T1TrackingConfig") -> NewtonObservationConfig:
    motion_params = {"command_name": "motion"}

    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        # Proprioception.
        base_ang_vel_obs = ObservationTermConfig(
            func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2)
        )
        projected_gravity_obs = ObservationTermConfig(
            func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05)
        )
        dof_pos_obs = ObservationTermConfig(
            func=dof_pos, scale=1.0, noise=Unoise(-0.03, 0.03)
        )
        dof_pos_diff_obs = ObservationTermConfig(
            func=dof_pos_nominal_difference, scale=1.0, noise=Unoise(-0.03, 0.03)
        )
        dof_vel_obs = ObservationTermConfig(
            func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5)
        )
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        # Motion reference (anchor-relative + reference joint pos/vel).
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        motion_anchor_pos = ObservationTermConfig(
            func=motion_anchor_pos_b, scale=1.0, params=motion_params,
            noise=Unoise(-0.05, 0.05),
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
        dof_pos_diff_obs = ObservationTermConfig(
            func=dof_pos_nominal_difference, scale=1.0,
        )
        dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=1.0)
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)
        base_quat_obs = ObservationTermConfig(func=base_quat, scale=1.0)
        # Motion reference + live robot body poses (critic only).
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
    class _ObsCfg(NewtonObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: "T1TrackingConfig") -> NewtonActionConfig:
    """Settle-relative joint position action; settle_steps=0 for tracking.

    Motion command writes the initial pose on reset; the policy's first
    step therefore targets ``current_pos + action * scale`` where
    ``current_pos`` is already the motion's reference pose, i.e. a pure
    residual control scheme on top of the reference motion.
    """
    from rlworld.rl.envs.mdp.actions import (
        SettleRelativeJointPositionAction,
        SettleRelativeJointPositionActionCfg,
    )

    r = cfg.robot
    return NewtonActionConfig(
        actuated_dof_names=r.prefixed_actuated_dof_patterns,
        clip_actions=(-100.0, 100.0),
        action_terms={
            "body": SettleRelativeJointPositionActionCfg(
                class_type=SettleRelativeJointPositionAction,
                joint_names=list(r.prefixed_actuated_dof_patterns),
                scale=cfg.action_scale,
                clip=(-100.0, 100.0),
                settle_steps=cfg.settle_steps,
            ),
        },
    )


def build_reward(cfg: "T1TrackingConfig") -> RewardConfig:
    motion_params_std = lambda std: {"command_name": "motion", "std": std}

    @dataclass
    class _RewardsCfg(RewardConfig):
        # Tracking (6 exponential terms).
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
        # Smoothness / safety penalties.
        raw_action_rate_l2 = RewardTermConfig(
            func=rf_mjlab.raw_action_rate_l2_mjlab,
            weight=cfg.action_rate_l2_weight,
        )
        joint_pos_limits = RewardTermConfig(
            func=rf_mjlab.joint_pos_limits_mjlab,
            weight=cfg.joint_pos_limits_weight,
        )
        self_collision_cost = RewardTermConfig(
            func=rf_newton.wtw_collision,
            weight=cfg.self_collision_weight,
            params={"contact_group": "self_collision", "force_threshold": 10.0},
        )

    return _RewardsCfg()


def build_dr_terms(cfg: "T1TrackingConfig") -> Dict[str, EventTermConfig]:
    """Newton DR — same 3-axis friction story as t1_getup.

    No ``reset`` / ``reset_dr`` terms here: motion command owns initial
    state.
    """
    r = cfg.robot
    return {
        "geom_friction_slide": EventTermConfig(
            func=newton_dr.randomize_geom_friction_axis,
            mode="startup",
            params={
                "ranges": (0.8, 1.5),
                "axes": [0],
                "operation": "abs",
                "distribution": "uniform",
                "body_patterns": None,
            },
        ),
        "foot_friction_spin": EventTermConfig(
            func=newton_dr.randomize_geom_friction_axis,
            mode="startup",
            params={
                "ranges": (1e-4, 2e-2),
                "axes": [1],
                "operation": "abs",
                "distribution": "log_uniform",
                "body_patterns": r.foot_body_pattern_newton,
            },
        ),
        "foot_friction_roll": EventTermConfig(
            func=newton_dr.randomize_geom_friction_axis,
            mode="startup",
            params={
                "ranges": (1e-5, 5e-3),
                "axes": [2],
                "operation": "abs",
                "distribution": "log_uniform",
                "body_patterns": r.foot_body_pattern_newton,
            },
        ),
    }
