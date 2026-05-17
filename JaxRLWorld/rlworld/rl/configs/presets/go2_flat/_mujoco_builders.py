"""MuJoCo (mjlab) builders for Go2 flat-terrain locomotion.

These functions are dispatched from ``Go2FlatConfig.build()`` when
``sim_type == "mujoco"``. The bodies are extracted directly from the
old ``presets/go2_flat/mujoco/base.py`` so the produced configs are
identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

from mjlab.asset_zoo.robots import GO2_ACTION_SCALE as MJLAB_GO2_ACTION_SCALE

from rlworld.assets.unitree_go2.go2_constants import (
    FULL_COLLISION,
    get_spec as go2_get_spec,
)
from rlworld.rl.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
from rlworld.rl.configs import RewardConfig, TerminationTermConfig
from rlworld.rl.configs.common_config_classes import TerminationsConfig
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.mujoco_config_classes import (
    MujocoActionConfig,
    MujocoConfigsForRun,
    MujocoEnvConfig,
    MujocoObservationConfig,
    MujocoSceneConfig,
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
    STIFFNESS_HIP,
    STIFFNESS_KNEE,
)
from rlworld.rl.configs.scene import SceneEntitySelector
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    InitialStateCfg,
    MujocoEntityCfg,
)
from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from rlworld.rl.envs.mdp.events.dr import mujoco as mujoco_dr, unified as unified_dr
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf
from rlworld.rl.envs.mdp.terminations.mujoco import terminations as tf

if TYPE_CHECKING:
    from .base import Go2FlatConfig

# ── Module-level constants exposed to base.Go2FlatConfig.build() ─────

CONFIGS_FOR_RUN_CLS = MujocoConfigsForRun
OBSERVATION_CFG_CLS = MujocoObservationConfig


def get_foot_names(robot) -> tuple[str, ...]:
    """MuJoCo uses bare foot names."""
    return robot.foot_names


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: Go2FlatConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: Go2FlatConfig, timing: Dict[str, Any]) -> MujocoEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        bad_orientation = TerminationTermConfig(
            tf.bad_orientation,
            {"limit_angle": math.radians(30.0)},
        )
        time_out = TerminationTermConfig(tf.time_out)

    return MujocoEnvConfig(
        num_envs=cfg.num_envs,
        env_name="MujocoLocomotionEnv",
        task_name="Go2 Velocity Tracking",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: Go2FlatConfig, timing: Dict[str, Any]) -> MujocoSceneConfig:
    """Build scene config with mjlab SceneCfg."""
    r = cfg.robot
    physics_dt = timing["dt"]
    substeps = timing.get("substeps", 1)

    foot_names = ("FR", "FL", "RR", "RL")
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=geom_names,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=".*",
            entity="robot",
            exclude=(".*foot.*", ".*calf.*"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=timing["decimation"],
    )

    ActuatorCls, _delay_kwargs = (
        (IdealPDActuatorCfg, {})
        if cfg.use_ideal_pd_actuator
        else (DelayedPDActuatorCfg, {"min_delay": 1, "max_delay": 3})
    )

    # Optional PD overrides — fall back to module-level defaults
    # when the corresponding ``Go2Config.*_override`` is None (i.e.
    # vanilla training); when set the configured value is baked into
    # the actuator config from step 0 of training, mirroring the
    # Newton builder's pattern.
    stiffness_hip = r.kp_hip_override if r.kp_hip_override is not None else STIFFNESS_HIP
    damping_hip = r.kd_hip_override if r.kd_hip_override is not None else DAMPING_HIP
    stiffness_knee = r.kp_knee_override if r.kp_knee_override is not None else STIFFNESS_KNEE
    damping_knee = r.kd_knee_override if r.kd_knee_override is not None else DAMPING_KNEE

    robot_entity = MujocoEntityCfg(
        urdf_path=r.urdf_path,
        init_state=InitialStateCfg(
            pos=(0, 0, r.base_init_height),
            joint_pos=r.default_joint_angles,
        ),
        floating=True,
        articulation=ArticulationCfg(
            actuators=(
                ActuatorCls(
                    target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                    stiffness=stiffness_hip,
                    damping=damping_hip,
                    effort_limit=EFFORT_HIP,
                    armature=ARMATURE_HIP,
                    **_delay_kwargs,
                ),
                ActuatorCls(
                    target_names_expr=(".*_calf_joint",),
                    stiffness=stiffness_knee,
                    damping=damping_knee,
                    effort_limit=EFFORT_KNEE,
                    armature=ARMATURE_KNEE,
                    **_delay_kwargs,
                ),
            ),
        ),
        spec_fn=go2_get_spec,
        collisions=(FULL_COLLISION,),
    )

    return MujocoSceneConfig(
        physics_dt=physics_dt,
        substeps=substeps,
        num_envs=cfg.num_envs,
        env_spacing=2.0,
        robot_entity_name="robot",
        entities={"robot": robot_entity},
        sensors=(feet_ground_cfg, body_ground_cfg),
        terrain_type="plane",
        cone="elliptic",
        solver_iterations=10,
        solver_ls_iterations=20,
        ccd_iterations=50,
        nconmax=35,
        njmax=1500,
        contact_sensor_maxmatch=64,
        preset_class_name=type(cfg).__name__,
        preset_module_path=type(cfg).__module__,
    )


def build_action(cfg: Go2FlatConfig) -> MujocoActionConfig:
    r = cfg.robot
    return MujocoActionConfig(
        entity_name="robot",
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=MJLAB_GO2_ACTION_SCALE,
        clip_actions=(-100.0, 100.0),
        offset=r.get_action_offset(),
    )


def build_reward(cfg: Go2FlatConfig) -> RewardConfig:
    """Build reward configuration matching Genesis/Newton Go2 rewards."""
    site_names = ("FR", "FL", "RR", "RL")

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

        # Orientation reward (mjlab-native, not common)
        flat_orientation = RewardTermConfig(
            func=rf.flat_orientation,
            weight=1.0,
            params={"std": 0.447},
        )

        variable_posture = RewardTermConfig(
            func=rf.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntitySelector(
                    name="robot",
                    joint_names=(".*",),
                ),
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

        joint_pos_limits = RewardTermConfig(
            func=rf.joint_pos_limits,
            weight=1.0,
        )

        raw_action_rate_l2 = RewardTermConfig(
            func=rf_common.raw_action_rate_l2,
            weight=0.1,
        )

        feet_clearance = RewardTermConfig(
            func=rf.feet_clearance,
            weight=2.0,
            params={
                "asset_cfg": SceneEntitySelector(
                    name="robot",
                    site_names=site_names,
                ),
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )

        feet_swing_height = RewardTermConfig(
            func=rf.feet_swing_height,
            weight=0.25,
            params={
                "contact_group": "feet_ground_contact",
                "asset_cfg": SceneEntitySelector(
                    name="robot",
                    site_names=site_names,
                ),
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )

        feet_slip = RewardTermConfig(
            func=rf.feet_slip,
            weight=0.1,
            params={
                "contact_group": "feet_ground_contact",
                "asset_cfg": SceneEntitySelector(
                    name="robot",
                    site_names=site_names,
                ),
                "command_threshold": 0.05,
            },
        )

        soft_landing = RewardTermConfig(
            func=rf.soft_landing,
            weight=1e-5,
            params={
                "contact_group": "feet_ground_contact",
                "command_threshold": 0.05,
            },
        )

    return _RewardsCfg()


def build_dr_terms(cfg: Go2FlatConfig) -> Dict[str, EventTermConfig]:
    """MuJoCo-specific domain randomization terms."""
    r = cfg.robot
    foot_geom_names = (
        "FR_foot_collision",
        "FL_foot_collision",
        "RR_foot_collision",
        "RL_foot_collision",
    )

    terms: Dict[str, EventTermConfig] = {
        "randomize_friction": EventTermConfig(
            func=unified_dr.randomize_friction,
            mode="reset_dr",
            params={
                "asset_cfg": SceneEntitySelector(name="robot", geom_names=foot_geom_names),
                "friction_range": (0.3, 1.2),
                "operation": "abs",
                "shared_random": True,
            },
        ),
        "randomize_base_mass": EventTermConfig(
            func=unified_dr.randomize_body_mass,
            mode="reset_dr",
            params={
                "asset_cfg": SceneEntitySelector(name="robot", body_names=(r.base_link_name,)),
                "mass_range": (0.8, 1.2),
                "operation": "scale",
            },
        ),
        "randomize_joint_friction": EventTermConfig(
            func=unified_dr.randomize_joint_friction,
            mode="reset_dr",
            params={
                "asset_cfg": SceneEntitySelector(name="robot"),
                "friction_range": (0.0, 0.05),
                "operation": "abs",
            },
        ),
    }

    # Fixed-value friction setters — installed only when the
    # corresponding ``Go2Config.*_override`` is non-None. Mirrors the
    # Newton builder exactly so a mujoco-trained "with-override"
    # policy sees the same configured friction every reset that the
    # newton-trained one did.
    if r.foot_friction_override is not None:
        terms["set_foot_friction"] = EventTermConfig(
            func=mujoco_dr.set_foot_friction,
            mode="reset_dr",
            params={
                "value": float(r.foot_friction_override),
                "dr_scale": (0.9, 1.1),
            },
        )
        # Drop the wide friction-range randomisation when an identified
        # value is being pinned — otherwise the two terms race on the
        # same geoms.
        terms.pop("randomize_friction", None)

    if r.joint_frictionloss_override is not None:
        terms["set_joint_friction"] = EventTermConfig(
            func=mujoco_dr.set_joint_friction,
            mode="reset_dr",
            params={
                "value": float(r.joint_frictionloss_override),
                "dr_scale": (0.9, 1.1),
            },
        )
        terms.pop("randomize_joint_friction", None)

    return terms
