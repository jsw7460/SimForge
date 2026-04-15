"""Genesis-specific builders for T1 fall-recovery (getup) task.

Dispatched from :meth:`T1GetupConfig.build` when ``sim_type == "genesis"``.
Mirrors the Newton builder's structure with Genesis-specific scene,
sensor, and DR APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import genesis as gs

from rlworld.rl.actuators import DelayedPDActuatorCfg
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
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    GenesisEntityCfg,
    GroundPlaneCfg,
    InitialStateCfg,
)
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.mdp.configs import TerminationTermConfig
from rlworld.rl.envs.mdp.events.dr import genesis as genesis_dr
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    base_height,
    base_lin_vel,
    base_quat,
    dof_pos,
    dof_vel,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import getup as rf_getup
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.genesis import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.rewards.genesis import reward_terms as rf_genesis
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf

if TYPE_CHECKING:
    from .base import T1GetupConfig


# ── Module-level constants exposed to T1GetupConfig.build() ──────────

CONFIGS_FOR_RUN_CLS = GenesisConfigsForRun
OBSERVATION_CFG_CLS = ObservationConfig


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: "T1GetupConfig") -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False)


def build_env(cfg: "T1GetupConfig", timing: Dict[str, Any]) -> EnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        # NO roll_pitch_violation — the robot starts fallen.
        time_out = TerminationTermConfig(max_episode_exceed)
        energy = TerminationTermConfig(
            common_tf.energy_termination,
            {
                "threshold": cfg.energy_threshold,
                "skip_steps": cfg.settle_steps,
            },
        )

    return EnvConfig(
        env_name="GenesisEnv",
        task_name="T1_Getup",
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        decimation=timing["decimation"],
        episode_length_s=cfg.episode_length_s,
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: "T1GetupConfig", timing: Dict[str, Any]) -> SceneConfig:
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
                link_name=r.base_link_name,
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
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=30,
            batch_dofs_info=True,
        ),
        robot_cfg=r,
    )


def build_observation(cfg: "T1GetupConfig") -> ObservationConfig:
    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(
            func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2)
        )
        projected_gravity_obs = ObservationTermConfig(
            func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05)
        )
        # Unbiased dof_pos (see _newton_builders for rationale).
        dof_pos_obs = ObservationTermConfig(
            func=dof_pos, scale=1.0, noise=Unoise(-0.03, 0.03)
        )
        dof_vel_obs = ObservationTermConfig(
            func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5)
        )
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)

    @dataclass
    class _CriticObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(
            func=base_ang_vel, scale=1.0
        )
        base_lin_vel_obs = ObservationTermConfig(func=base_lin_vel, scale=1.0)
        projected_gravity_obs = ObservationTermConfig(
            func=projected_gravity, scale=1.0
        )
        dof_pos_obs = ObservationTermConfig(
            func=dof_pos, scale=1.0
        )
        dof_vel_obs = ObservationTermConfig(
            func=dof_vel, scale=1.0
        )
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)
        base_quat_obs = ObservationTermConfig(func=base_quat, scale=1.0)

    @dataclass
    class _ObsCfg(ObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: "T1GetupConfig") -> ActionConfig:
    """Settle-relative joint position action (mjlab_playground T1 getup).

    See ``_newton_builders.build_action`` for the rationale; the
    Genesis version is identical in intent — a single
    :class:`SettleRelativeJointPositionAction` spanning every
    actuated joint.
    """
    from rlworld.rl.envs.mdp.actions import (
        SettleRelativeJointPositionAction,
        SettleRelativeJointPositionActionCfg,
    )

    r = cfg.robot
    return ActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        clip_actions=(-100.0, 100.0),
        action_terms={
            "body": SettleRelativeJointPositionActionCfg(
                class_type=SettleRelativeJointPositionAction,
                joint_names=list(r.actuated_dof_patterns),
                scale=cfg.action_scale,
                clip=(-100.0, 100.0),
                settle_steps=cfg.settle_steps,
            ),
        },
    )


def build_reward(cfg: "T1GetupConfig") -> RewardConfig:
    r = cfg.robot

    @dataclass
    class _RewardsCfg(RewardConfig):
        orientation_upright = RewardTermConfig(
            func=rf_getup.orientation_upright,
            weight=cfg.orientation_weight,
            params={"std": cfg.orientation_std},
        )
        trunk_height = RewardTermConfig(
            func=rf_getup.height_to_target,
            weight=cfg.trunk_height_weight,
            params={
                "desired_height": cfg.trunk_desired_height,
                "body_name": r.trunk_body_name,
            },
        )
        waist_height = RewardTermConfig(
            func=rf_getup.height_to_target,
            weight=cfg.waist_height_weight,
            params={
                "desired_height": cfg.waist_desired_height,
                "body_name": r.waist_body_name,
            },
        )
        gated_posture = RewardTermConfig(
            func=rf_getup.GatedPostureTracker,
            weight=cfg.gated_posture_weight,
            params={
                "std_dict": cfg.posture_std_dict,
                "gate_threshold": cfg.posture_gate_threshold,
            },
        )
        joint_pos_limits = RewardTermConfig(
            func=rf_mjlab.joint_pos_limits_mjlab,
            weight=cfg.joint_pos_limits_weight,
        )
        raw_action_rate_l2 = RewardTermConfig(
            func=rf_mjlab.raw_action_rate_l2_mjlab,
            weight=cfg.action_rate_l2_weight,
        )
        joint_vel_l2 = RewardTermConfig(
            func=rf_common.penalize_dof_vel,
            weight=cfg.joint_vel_l2_weight,
        )
        self_collision_cost = RewardTermConfig(
            func=rf_genesis.wtw_collision,
            weight=cfg.self_collision_weight,
            params={"contact_group": "self_collision", "force_threshold": 1.0},
        )
        # Logging-only metric (weight=0).
        getup_success = RewardTermConfig(
            func=rf_getup.GetupSuccessTracker,
            weight=0.0,
            params={
                "desired_height": cfg.trunk_desired_height,
                "body_name": r.trunk_body_name,
            },
        )

    return _RewardsCfg()


def build_dr_terms(cfg: "T1GetupConfig") -> Dict[str, EventTermConfig]:
    """Genesis domain randomization.

    **Known gap vs mjlab_playground**: Genesis's contact solver uses a
    scalar (isotropic) friction cone per-geom, not MuJoCo's 3-vector
    ``(slide, spin, roll)``. The Genesis MJCF parser in
    ``genesis/utils/mjcf.py:607`` takes only ``mj_geom.friction[0]``
    and discards spin/roll, and the underlying ``GeomsInfo.friction``
    field is a single float per geom. As a result, the 3-axis geom
    friction DR that the MuJoCo and Newton backends use cannot be
    reproduced here without patching the Genesis engine itself.

    We fall back to the closest approximation: a scalar friction
    randomization over the same range as mjlab's slide axis
    (``0.3..1.5``) via Genesis's existing ``set_friction_ratio`` path.
    The ``mul`` operation keeps the base friction from the URDF intact
    and multiplies by a sampled ratio — identical semantics to
    ``randomize_friction`` used by the other Genesis presets.
    """
    r = cfg.robot
    return {
        # Encoder bias DR intentionally omitted — see _newton_builders
        # for the rationale (unbiased obs + unbiased action = symmetric).
        "randomize_body_com": EventTermConfig(
            func=genesis_dr.randomize_body_com_offset,
            mode="reset_dr",
            params={
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
                "link_names": (r.trunk_body_name,),
            },
        ),
        # Scalar slide-friction approximation — see docstring for the
        # reason we cannot match mjlab's 3-axis randomization here.
        "randomize_friction_scalar": EventTermConfig(
            func=genesis_dr.randomize_friction,
            mode="reset_dr",
            params={"friction_range": (0.3, 1.5)},
        ),
    }


