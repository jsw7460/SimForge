"""Newton-specific builders for T1 fall-recovery (getup) task.

Dispatched from :meth:`T1GetupConfig.build` when ``sim_type == "newton"``.
Structure mirrors ``g1_29dof/_newton_builders.py`` but:
  - only the ``max_episode`` termination is registered (the robot must
    be allowed to lie on its back)
  - observation has no velocity-command term and no foot observations
  - action config carries ``settle_steps`` from the preset
  - rewards use the cross-sim getup terms from
    ``rewards.common.getup`` plus the common mjlab rewards for
    smoothness / self-collision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import warp as wp

from rlworld.rl.actuators import DelayedPDActuatorCfg
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
from rlworld.rl.envs.mdp.events import common_event_terms as common_ef
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    base_height,
    base_lin_vel,
    base_quat,
    dof_pos,
    dof_pos_biased,
    dof_vel,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import getup as rf_getup
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.rewards.newton import reward_terms as rf_newton
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as common_tf

if TYPE_CHECKING:
    from .base import T1GetupConfig


# ── Module-level constants exposed to T1GetupConfig.build() ──────────

CONFIGS_FOR_RUN_CLS = NewtonConfigsForRun
OBSERVATION_CFG_CLS = NewtonObservationConfig


def _initial_quat() -> Any:
    """Identity orientation — the fallen-reset event overrides this."""
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.0)


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: "T1GetupConfig") -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: "T1GetupConfig", timing: Dict[str, Any]) -> NewtonEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        # NO roll_pitch_violation — the robot starts fallen by design.
        max_episode = TerminationTermConfig(max_episode_exceed)
        energy = TerminationTermConfig(
            common_tf.energy_termination,
            {
                "threshold": cfg.energy_threshold,
                "skip_steps": cfg.settle_steps,
            },
        )

    return NewtonEnvConfig(
        num_envs=cfg.num_envs,
        env_name="NewtonEnv",
        task_name="T1_Getup",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: "T1GetupConfig", timing: Dict[str, Any]) -> NewtonSceneConfig:
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
                sensor_name="self_collision",
                sensing_obj_bodies=["*"],
                counterpart_bodies=["*"],
                include_total=False,
            ),
        ],
        add_ground=True,
        env_spacing=(2.0, 2.0, 0.0),
    )


def build_observation(cfg: "T1GetupConfig") -> NewtonObservationConfig:
    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(
            func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2)
        )
        projected_gravity_obs = ObservationTermConfig(
            func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05)
        )
        # Biased dof_pos: adds per-env encoder bias written by the
        # randomize_encoder_bias DR term so the policy learns robust
        # behaviour under sensor miscalibration (mjlab_playground
        # observation with ``biased=True``).
        dof_pos_obs = ObservationTermConfig(
            func=dof_pos_biased, scale=1.0, noise=Unoise(-0.01, 0.01)
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
    class _ObsCfg(NewtonObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: "T1GetupConfig") -> NewtonActionConfig:
    r = cfg.robot
    return NewtonActionConfig(
        actuated_dof_names=r.prefixed_actuated_dof_patterns,
        action_scale=r.prefixed_action_scale,
        clip_actions=(-100.0, 100.0),
        offset=r.get_prefixed_action_offset(),
        settle_steps=cfg.settle_steps,
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
                "body_name": r.prefixed(r.trunk_body_name),
            },
        )
        waist_height = RewardTermConfig(
            func=rf_getup.height_to_target,
            weight=cfg.waist_height_weight,
            params={
                "desired_height": cfg.waist_desired_height,
                "body_name": r.prefixed(r.waist_body_name),
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
            func=rf_newton.wtw_collision,
            weight=cfg.self_collision_weight,
            params={"contact_group": "self_collision", "force_threshold": 10.0},
        )
        # Logging-only metric (weight=0 so it contributes 0 to the total
        # reward but still gets written to ``rew_buf_per_type`` each step).
        getup_success = RewardTermConfig(
            func=rf_getup.GetupSuccessTracker,
            weight=0.0,
            params={
                "desired_height": cfg.trunk_desired_height,
                "body_name": r.prefixed(r.trunk_body_name),
            },
        )

    return _RewardsCfg()


def build_dr_terms(cfg: "T1GetupConfig") -> Dict[str, EventTermConfig]:
    """Newton domain randomization — minimal set for the MVP skeleton.

    Phase K tuning can widen to match mjlab_playground getup DR
    (friction split, encoder bias, wider body-com ranges).
    """
    r = cfg.robot
    return {
        "randomize_encoder_bias": EventTermConfig(
            func=common_ef.randomize_encoder_bias,
            mode="reset_dr",
            params={"bias_range": (-0.015, 0.015)},
        ),
        "randomize_body_com": EventTermConfig(
            func=newton_dr.randomize_body_com_offset,
            mode="reset_dr",
            params={
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
                "body_patterns": (r.prefixed(r.trunk_body_name),),
            },
        ),
        "randomize_joint_friction": EventTermConfig(
            func=newton_dr.randomize_joint_friction,
            mode="reset_dr",
            params={"friction_range": (0.0, 0.05)},
        ),
    }
