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
                enable_self_collisions=True
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
        # mjlab_playground uses a biased dof_pos observation paired with a
        # ``-encoder_bias`` correction in its relative action. Since our
        # relative action (SettleRelative) does NOT subtract the bias,
        # keeping a biased observation here would create an asymmetry
        # that the policy would have to learn to compensate for. The
        # cleanest choice is to drop encoder_bias entirely on both
        # sides — see build_dr_terms (no randomize_encoder_bias term).
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
    class _ObsCfg(NewtonObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: "T1GetupConfig") -> NewtonActionConfig:
    """Settle-relative joint position action (mjlab_playground T1 getup).

    Uses the new ActionTerm system: a single
    :class:`SettleRelativeJointPositionAction` spanning every
    actuated joint. Target = ``current_joint_pos + raw * scale``,
    with a forced hold at ``current_joint_pos`` for the first
    ``settle_steps`` control steps after each reset. This mirrors
    mjlab_playground's ``SettleRelativeJointPositionActionCfg`` used
    for fall-recovery tasks. Legacy ``scale``/``clip``/``offset``
    fields are intentionally left at their defaults because
    ``action_terms`` takes precedence once non-empty.
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
    """Newton domain randomization — mjlab_playground-faithful.

    Three-axis geom friction randomization matches mjlab's T1 getup:

    * slide (axis 0): uniform ``0.3..1.5`` over every collision shape
    * spin  (axis 1): log_uniform ``1e-4..2e-2`` over foot shapes only
    * roll  (axis 2): log_uniform ``1e-5..5e-3`` over foot shapes only

    Newton stores the three axes as separate ``shape_material_mu*``
    arrays and its MuJoCo solver bridge syncs all three on
    ``notify_model_changed(SHAPE_PROPERTIES)``, so the behaviour is
    bit-compatible with mjlab once the Newton env runs on the
    ``solver_type="mujoco"`` backend (which T1 getup does).
    """
    r = cfg.robot
    return {
        # Encoder bias DR intentionally omitted: our action path does
        # not subtract ``encoder_bias`` from the target (unlike
        # mjlab_playground's SettleRelativeJointPositionAction), so
        # applying a biased observation would desynchronize the
        # policy's action from the physical state. See build_observation
        # for the matching decision to use unbiased dof_pos.
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
        # Slide: randomize across all robot shapes (body_patterns=None).
        # Matches mjlab's ``geom_names=(".*_collision",)`` which in
        # practice covers every collision geom on the robot — Newton's
        # URDF-loaded shapes don't have per-geom names, so we simply
        # skip the filter (touching every robot shape) instead.
        "geom_friction_slide": EventTermConfig(
            func=newton_dr.randomize_geom_friction_axis,
            mode="reset_dr",
            params={
                "ranges": (0.3, 1.5),
                "axes": [0],
                "operation": "abs",
                "distribution": "uniform",
                "body_patterns": None,
            },
        ),
        # Spin/roll: filter by foot body name so only the foot shapes
        # pick up the log-uniform low-range friction. Body-name
        # filtering is needed because Newton's URDF loader drops geom
        # names; ``model.body_shapes[foot_body_idx]`` resolves to the
        # shape indices attached to the foot links.
        "foot_friction_spin": EventTermConfig(
            func=newton_dr.randomize_geom_friction_axis,
            mode="reset_dr",
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
            mode="reset_dr",
            params={
                "ranges": (1e-5, 5e-3),
                "axes": [2],
                "operation": "abs",
                "distribution": "log_uniform",
                "body_patterns": r.foot_body_pattern_newton,
            },
        ),
    }
