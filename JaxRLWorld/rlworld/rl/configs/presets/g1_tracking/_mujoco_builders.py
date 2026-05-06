"""MuJoCo (mjlab) builders for G1 motion tracking task.

Dispatched from :meth:`G1TrackingConfig.build` when
``sim_type == "mujoco"``. Uses the same mjlab asset_zoo entries
(``g1_constants.get_spec`` + ``G1_FULL_COLLISION`` +
``G1_ACTION_SCALE``) as the G1 locomotion preset so the physical model
is bit-identical to Mjlab's own reference G1 tracking task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

from mjlab.asset_zoo.robots import G1_ACTION_SCALE as MJLAB_G1_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION as G1_FULL_COLLISION,
    get_spec as g1_get_spec,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from rlworld.rl.actuators import IdealPDActuatorCfg
from rlworld.rl.configs import RewardConfig, TerminationTermConfig
from rlworld.rl.configs.common_config_classes import (
    ObservationGroupConfig,
    TerminationsConfig,
)
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.mujoco_config_classes import (
    MujocoActionConfig,
    MujocoConfigsForRun,
    MujocoEnvConfig,
    MujocoObservationConfig,
    MujocoSceneConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    InitialStateCfg,
    MujocoEntityCfg,
)
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
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf
from rlworld.rl.envs.mdp.terminations.common import motion_tracking as tt_motion
from rlworld.rl.envs.mdp.terminations.mujoco import terminations as tf

if TYPE_CHECKING:
    from .base import G1TrackingConfig


CONFIGS_FOR_RUN_CLS = MujocoConfigsForRun
OBSERVATION_CFG_CLS = MujocoObservationConfig


def build_visualization(cfg: G1TrackingConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: G1TrackingConfig, timing: Dict[str, Any]) -> MujocoEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        time_out = TerminationTermConfig(tf.time_out)
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

    return MujocoEnvConfig(
        num_envs=cfg.num_envs,
        env_name="MujocoEnv",
        task_name="G1_Tracking",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: G1TrackingConfig, timing: Dict[str, Any]) -> MujocoSceneConfig:
    """Scene config using mjlab asset_zoo G1 + self-collision subtree sensor.

    Mirror of the G1 locomotion preset's MuJoCo scene (pelvis self-collision
    subtree, solver settings), sans foot-ground sensor — tracking rewards
    don't need foot-air-time / foot-contact observation.
    """
    r = cfg.robot
    physics_dt = timing["dt"]
    substeps = timing.get("substeps", 1)

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=timing["decimation"],
    )

    robot_entity = MujocoEntityCfg(
        urdf_path=r.urdf_path,  # Ignored (spec_fn is used)
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
        spec_fn=g1_get_spec,
        collisions=(G1_FULL_COLLISION,),
    )

    return MujocoSceneConfig(
        physics_dt=physics_dt,
        substeps=substeps,
        num_envs=cfg.num_envs,
        env_spacing=2.0,
        robot_entity_name="robot",
        entities={"robot": robot_entity},
        sensors=(self_collision_cfg,),
        terrain_type="plane",
        solver_iterations=10,
        solver_ls_iterations=20,
        ccd_iterations=50,
        nconmax=35,
        njmax=250,
        contact_sensor_maxmatch=64,
        preset_class_name=type(cfg).__name__,
        preset_module_path=type(cfg).__module__,
    )


def build_observation(cfg: G1TrackingConfig) -> MujocoObservationConfig:
    motion_params = {"command_name": "motion"}

    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
        projected_gravity_obs = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
        base_lin_vel_obs = ObservationTermConfig(func=base_lin_vel, scale=1.0, noise=Unoise(-0.5, 0.5))
        dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
        dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        motion_anchor_pos = ObservationTermConfig(
            func=motion_anchor_pos_b,
            scale=1.0,
            params=motion_params,
            noise=Unoise(-0.25, 0.25),
        )
        motion_anchor_ori = ObservationTermConfig(
            func=motion_anchor_ori_b,
            scale=1.0,
            params=motion_params,
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
            func=motion_anchor_pos_b,
            scale=1.0,
            params=motion_params,
        )
        motion_anchor_ori = ObservationTermConfig(
            func=motion_anchor_ori_b,
            scale=1.0,
            params=motion_params,
        )
        robot_body_pos = ObservationTermConfig(
            func=robot_body_pos_b,
            scale=1.0,
            params=motion_params,
        )
        robot_body_ori = ObservationTermConfig(
            func=robot_body_ori_b,
            scale=1.0,
            params=motion_params,
        )

    @dataclass
    class _ObsCfg(MujocoObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: G1TrackingConfig) -> MujocoActionConfig:
    """JointPositionAction with mjlab-native G1 action scale + default offset."""
    r = cfg.robot
    return MujocoActionConfig(
        entity_name="robot",
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=MJLAB_G1_ACTION_SCALE,
        clip_actions=(-100.0, 100.0),
        offset=r.get_action_offset(),
    )


def build_reward(cfg: G1TrackingConfig) -> RewardConfig:
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
            func=rf.raw_action_rate_l2,
            weight=cfg.action_rate_l2_weight,
        )
        joint_pos_limits = RewardTermConfig(
            func=rf.joint_pos_limits,
            weight=cfg.joint_pos_limits_weight,
        )
        self_collision_cost = RewardTermConfig(
            func=rf.self_collision_cost,
            weight=cfg.self_collision_weight,
            params={"contact_group": "self_collision", "force_threshold": 10.0},
        )

    return _RewardsCfg()


def build_dr_terms(cfg: G1TrackingConfig) -> Dict[str, EventTermConfig]:
    """MuJoCo DR — 3-axis friction (Mjlab G1 tracking foot_friction)."""
    from rlworld.rl.envs.mdp.events import mujoco as ef
    from rlworld.rl.envs.mdp.events.mujoco import EntityCfg

    return {
        # Foot friction over all of G1's foot collision geoms.
        # Mjlab G1 tracking uses ``^(left|right)_foot[1-7]_collision$``.
        "foot_friction": EventTermConfig(
            func=ef.randomize_friction,
            mode="startup",
            params={
                "ranges": (0.3, 1.2),
                "operation": "abs",
                "axes": [0],
                "distribution": "uniform",
                "entity_cfg": EntityCfg(
                    name="robot",
                    geom_names=tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)),
                ),
                "shared_random": True,
            },
        ),
    }
