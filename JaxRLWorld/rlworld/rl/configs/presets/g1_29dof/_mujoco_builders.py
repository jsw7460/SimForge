"""MuJoCo (mjlab) builders for G1 29-DOF flat-terrain locomotion.

These functions are dispatched from ``G1FlatConfig.build()`` when
``sim_type == "mujoco"``. The bodies are extracted directly from the
old ``presets/g1_29dof/mujoco/base.py`` so the produced configs are
identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

from mjlab.asset_zoo.robots import G1_ACTION_SCALE as MJLAB_G1_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION as G1_FULL_COLLISION,
    get_spec as g1_get_spec,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from rlworld.rl.actuators import DelayedPDActuatorCfg
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
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    base_height,
    base_quat,
    command as command_obs,
    dof_pos,
    dof_vel,
    foot_air_time,
    foot_contact_forces,
    foot_contact_indicator as foot_contact,
    foot_height,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf
from rlworld.rl.envs.mdp.terminations.mujoco import terminations as tf

if TYPE_CHECKING:
    from .base import G1FlatConfig


# ── Module-level constants exposed to base.G1FlatConfig.build() ──────

CONFIGS_FOR_RUN_CLS = MujocoConfigsForRun
OBSERVATION_CFG_CLS = MujocoObservationConfig


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: G1FlatConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: G1FlatConfig, timing: Dict[str, Any]) -> MujocoEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        bad_orientation = TerminationTermConfig(
            tf.bad_orientation,
            {"limit_angle": math.radians(70.0)},
        )
        time_out = TerminationTermConfig(tf.time_out)

    return MujocoEnvConfig(
        num_envs=cfg.num_envs,
        env_name="MujocoEnv",
        task_name="G1 Velocity Tracking",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: G1FlatConfig, timing: Dict[str, Any]) -> MujocoSceneConfig:
    """Build scene config with mjlab SceneCfg and SimulationCfg."""
    r = cfg.robot
    physics_dt = timing["dt"]
    substeps = timing.get("substeps", 1)

    # Contact sensor for feet-ground contact
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Self-collision sensor
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
                    frictionloss=0.3,
                    min_delay=0,
                    max_delay=2,
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
        sensors=(feet_ground_cfg, self_collision_cfg),
        terrain_type="plane",
        solver_iterations=10,
        solver_ls_iterations=20,
        ccd_iterations=50,
        nconmax=35,
        njmax=1500,
        contact_sensor_maxmatch=64,
        preset_class_name=type(cfg).__name__,
        preset_module_path=type(cfg).__module__,
    )


def build_observation(cfg: G1FlatConfig) -> MujocoObservationConfig:
    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
        projected_gravity = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        dof_pos = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
        dof_vel = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
        actions = ObservationTermConfig(func=raw_actions, scale=1.0)

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
            func=foot_height,
            scale=1.0,
            params={"site_names": ("left_foot", "right_foot")},
        )
        foot_air_time_obs = ObservationTermConfig(func=foot_air_time, scale=1.0)
        foot_contact_obs = ObservationTermConfig(func=foot_contact, scale=1.0)
        foot_contact_forces_obs = ObservationTermConfig(func=foot_contact_forces, scale=0.01)

    @dataclass
    class _ObsCfg(MujocoObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: G1FlatConfig) -> MujocoActionConfig:
    r = cfg.robot
    return MujocoActionConfig(
        entity_name="robot",
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=MJLAB_G1_ACTION_SCALE,
        clip_actions=(-100.0, 100.0),
        offset=r.get_action_offset(),
    )


def build_reward(cfg: G1FlatConfig) -> RewardConfig:
    """Build reward configuration matching mjlab G1 velocity task."""
    site_names = ("left_foot", "right_foot")

    @dataclass
    class _RewardsCfg(RewardConfig):
        # Tracking rewards (common — uses RobotData interface)
        track_lin_vel = RewardTermConfig(
            func=rf_common.track_lin_vel,
            weight=2.0,
            params={"std": math.sqrt(0.25), "penalize_z": True},
        )
        track_ang_vel = RewardTermConfig(
            func=rf_common.track_ang_vel,
            weight=2.0,
            params={"std": math.sqrt(0.5), "penalize_xy": True},
        )

        # Orientation reward (mjlab-native, with body-anchored asset_cfg)
        flat_orientation = RewardTermConfig(
            func=rf.flat_orientation,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=("torso_link",),
                ),
            },
        )

        self_collision_cost = RewardTermConfig(
            func=rf.self_collision_cost,
            weight=1.0,
            params={"contact_group": "self_collision", "force_threshold": 10.0},
        )

        # Variable posture reward (G1-specific std values)
        variable_posture = RewardTermConfig(
            func=rf.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=(".*",),
                ),
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

        # Body angular velocity penalty
        body_angular_velocity_penalty = RewardTermConfig(
            func=rf.body_angular_velocity_penalty,
            weight=0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=("torso_link",),
                ),
            },
        )

        # Angular momentum penalty
        angular_momentum_penalty = RewardTermConfig(
            func=rf.angular_momentum_penalty,
            weight=0.02,
            params={"sensor_name": "robot/root_angmom"},
        )

        # Joint position limits
        joint_pos_limits = RewardTermConfig(
            func=rf.joint_pos_limits,
            weight=1.0,
        )

        # Action rate
        raw_action_rate_l2 = RewardTermConfig(
            func=rf.raw_action_rate_l2,
            weight=0.1,
        )

        # Feet clearance
        feet_clearance = RewardTermConfig(
            func=rf.feet_clearance,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", site_names=site_names),
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )

        # Feet swing height
        feet_swing_height = RewardTermConfig(
            func=rf.feet_swing_height,
            weight=0.25,
            params={
                "contact_group": "feet_ground_contact",
                "asset_cfg": SceneEntityCfg(name="robot", site_names=site_names),
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )

        # Feet slip
        feet_slip = RewardTermConfig(
            func=rf.feet_slip,
            weight=0.1,
            params={
                "contact_group": "feet_ground_contact",
                "asset_cfg": SceneEntityCfg(name="robot", site_names=site_names),
                "command_threshold": 0.05,
            },
        )

        # Soft landing
        soft_landing = RewardTermConfig(
            func=rf.soft_landing,
            weight=1e-5,
            params={
                "contact_group": "feet_ground_contact",
                "command_threshold": 0.05,
            },
        )

    return _RewardsCfg()


def build_dr_terms(cfg: G1FlatConfig) -> Dict[str, EventTermConfig]:
    """MuJoCo-specific domain randomization terms."""
    from rlworld.rl.envs.mdp.events import mujoco as ef
    from rlworld.rl.envs.mdp.events.mujoco import EntityCfg

    return {
        "randomize_encoder_bias": EventTermConfig(
            func=ef.randomize_encoder_bias,
            mode="reset_dr",
            params={
                "bias_range": (-0.015, 0.015),
                "entity_cfg": EntityCfg(name="robot"),
            },
        ),
        "randomize_body_com": EventTermConfig(
            func=ef.randomize_body_com_offset,
            mode="reset_dr",
            params={
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
                "operation": "add",
                "entity_cfg": EntityCfg(name="robot", body_names=("torso_link",)),
            },
        ),
        "randomize_joint_friction": EventTermConfig(
            func=ef.randomize_joint_friction,
            mode="reset_dr",
            params={"ranges": (0.0, 0.05), "operation": "abs"},
        ),
    }
