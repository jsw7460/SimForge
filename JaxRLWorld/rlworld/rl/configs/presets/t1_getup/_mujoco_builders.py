"""MuJoCo (mjlab) builders for T1 fall-recovery (getup) task.

Dispatched from :meth:`T1GetupConfig.build` when ``sim_type == "mujoco"``.
Unlike the Newton/Genesis builders which load T1 from a URDF, this
builder uses the mjlab asset-zoo entry at
``Mjlab/src/mjlab/asset_zoo/robots/booster_t1/`` — the MJCF + T1_FULL
collision config + HOME_KEYFRAME are taken directly from
``booster_t1.t1_constants``.

MuJoCo-specific reward functions (``rf.joint_pos_limits``,
``rf.raw_action_rate_l2``, ``rf.self_collision_cost``) are used in
place of the Newton/Genesis mjlab_* wrappers. The getup reward terms
(:mod:`rewards.common.getup`) work unchanged because they read state
through the cross-sim ``RobotData`` / ``act_manager`` interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

from mjlab.asset_zoo.robots.booster_t1.t1_constants import (
    FULL_COLLISION as T1_FULL_COLLISION,
    get_spec as t1_get_spec,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from rlworld.rl.actuators import ImplicitActuatorCfg
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
    base_lin_vel,
    base_quat,
    dof_pos,
    dof_pos_nominal_difference,
    dof_vel,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import getup as rf_getup, reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf
from rlworld.rl.envs.mdp.terminations.mujoco import terminations as tf

if TYPE_CHECKING:
    from .base import T1GetupConfig


# ── Module-level constants exposed to T1GetupConfig.build() ──────────

CONFIGS_FOR_RUN_CLS = MujocoConfigsForRun
OBSERVATION_CFG_CLS = MujocoObservationConfig


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: T1GetupConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: T1GetupConfig, timing: Dict[str, Any]) -> MujocoEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        # NO bad_orientation — the robot starts fallen.
        time_out = TerminationTermConfig(tf.time_out)
        # Replaced by ``power_penalty`` reward term — see
        # ``build_reward`` below and the ``power_penalty_weight``
        # curriculum in ``base.py``.
        # energy = TerminationTermConfig(
        #     common_tf.energy_termination,
        #     {
        #         "threshold": cfg.energy_threshold,
        #         "skip_steps": cfg.settle_steps,
        #     },
        # )

    return MujocoEnvConfig(
        num_envs=cfg.num_envs,
        env_name="MujocoEnv",
        task_name="T1_Getup",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: T1GetupConfig, timing: Dict[str, Any]) -> MujocoSceneConfig:
    """Build scene config with mjlab T1 asset + self-collision sensor."""
    r = cfg.robot
    physics_dt = timing["dt"]
    substeps = timing.get("substeps", 1)

    # Self-collision sensor over the entire T1 subtree (rooted at Trunk).
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern=r.trunk_body_name, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=r.trunk_body_name, entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=timing["decimation"],
    )

    robot_entity = MujocoEntityCfg(
        urdf_path=r.urdf_path,  # unused when spec_fn is set, but kept for parity
        init_state=InitialStateCfg(
            pos=(0, 0, r.base_init_height),
            joint_pos=r.default_joint_angles,
        ),
        floating=True,
        articulation=ArticulationCfg(
            actuators=(
                ImplicitActuatorCfg(
                    target_names_expr=(".*",),
                    stiffness=r.p_gains,
                    damping=r.d_gains,
                    armature=r.armature,
                    effort_limit=r.effort_limits,
                    frictionloss=0.1,
                    # min_delay=0,
                    # max_delay=2,
                ),
            ),
        ),
        spec_fn=t1_get_spec,
        collisions=(T1_FULL_COLLISION,),
    )

    # Solver / arena settings are pinned to mjlab_playground's getup
    # task (src/mjlab_playground/getup/getup_env_cfg.py:252-261) so the
    # MuJoCo path is bit-for-bit compatible with mjlab's reference
    # implementation of fall-recovery:
    #   njmax=200, impratio=10, cone=elliptic, iterations=10,
    #   ls_iterations=20, ccd_iterations=default (50), nconmax=auto
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
        nconmax=None,  # mjlab_playground leaves this unset → auto
        njmax=200,  # mjlab_playground getup explicit value
        impratio=10.0,  # mjlab_playground getup explicit value
        cone="elliptic",  # mjlab_playground getup explicit value
        contact_sensor_maxmatch=64,
        preset_class_name=type(cfg).__name__,
        preset_module_path=type(cfg).__module__,
    )


def build_observation(cfg: T1GetupConfig) -> MujocoObservationConfig:
    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
        projected_gravity_obs = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
        # Unbiased dof_pos (see _newton_builders for rationale).
        dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.03, 0.03))
        dof_pos_diff_obs = ObservationTermConfig(func=dof_pos_nominal_difference, scale=1.0, noise=Unoise(-0.03, 0.03))
        dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)

    @dataclass
    class _CriticObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(func=base_ang_vel, scale=1.0)
        base_lin_vel_obs = ObservationTermConfig(func=base_lin_vel, scale=1.0)
        projected_gravity_obs = ObservationTermConfig(func=projected_gravity, scale=1.0)
        dof_pos_obs = ObservationTermConfig(func=dof_pos, scale=1.0)
        dof_pos_diff_obs = ObservationTermConfig(func=dof_pos_nominal_difference, scale=1.0)
        dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=1.0)
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)
        base_quat_obs = ObservationTermConfig(func=base_quat, scale=1.0)

    @dataclass
    class _ObsCfg(MujocoObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: T1GetupConfig) -> MujocoActionConfig:
    """Settle-relative joint position action (mjlab_playground T1 getup).

    See ``_newton_builders.build_action`` for the rationale.
    """
    from rlworld.rl.envs.mdp.actions import (
        SettleRelativeJointPositionAction,
        SettleRelativeJointPositionActionCfg,
    )

    r = cfg.robot
    return MujocoActionConfig(
        entity_name="robot",
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


def build_reward(cfg: T1GetupConfig) -> RewardConfig:
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
            func=rf.joint_pos_limits,
            weight=cfg.joint_pos_limits_weight,
        )
        raw_action_rate_l2 = RewardTermConfig(
            func=rf.raw_action_rate_l2,
            weight=cfg.action_rate_l2_weight,
        )
        joint_vel_l2 = RewardTermConfig(
            func=rf_common.penalize_dof_vel,
            weight=cfg.joint_vel_l2_weight,
        )
        # power_penalty = RewardTermConfig(
        #     func=rf_getup.power_penalty,
        #     weight=cfg.power_penalty_weight,
        #     params={"skip_steps": cfg.settle_steps},
        # )
        self_collision_cost = RewardTermConfig(
            func=rf.self_collision_cost,
            weight=cfg.self_collision_weight,
            params={"contact_group": "self_collision", "force_threshold": 10.0},
        )

    return _RewardsCfg()


def build_dr_terms(cfg: T1GetupConfig) -> Dict[str, EventTermConfig]:
    """MuJoCo domain randomization — mjlab_playground-faithful.

    Three-axis geom friction randomization matches
    ``getup/config/t1/env_cfgs.py::booster_t1_getup_env_cfg``:

    * slide (axis 0): uniform ``0.3..1.5`` over every collision geom
    * spin  (axis 1): log_uniform ``1e-4..2e-2`` over foot geoms only
    * roll  (axis 2): log_uniform ``1e-5..5e-3`` over foot geoms only

    MuJoCo's solver natively uses the 3-vector ``geom_friction`` so
    this is applied via mjlab's ``dr.geom_friction`` under the hood.
    """
    from rlworld.rl.envs.mdp.events import mujoco as ef
    from rlworld.rl.envs.mdp.events.mujoco import EntityCfg

    r = cfg.robot
    # mjlab asset_zoo T1 has explicit collision geom names that survive
    # the MJCF load; T1Config exposes the foot names via a property.
    foot_geom_names = r.foot_geom_names_mjlab
    return {
        # Encoder bias DR intentionally omitted — see _newton_builders
        # for the rationale (unbiased obs + unbiased action = symmetric).
        "randomize_body_com": EventTermConfig(
            func=ef.randomize_body_com_offset,
            mode="startup",
            params={
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
                "operation": "add",
                "entity_cfg": EntityCfg(name="robot", body_names=(r.trunk_body_name,)),
            },
        ),
        # Slide randomization: mjlab's ``geom_names=(".*_collision",)``
        # selector matches every collision geom on the robot. Pass
        # None (unset) for geom_names so mjlab's default asset_cfg
        # covers the full entity — equivalent to the regex and cheaper
        # to resolve than a regex list.
        "geom_friction_slide": EventTermConfig(
            func=ef.randomize_friction,
            mode="startup",
            params={
                "ranges": (0.8, 1.5),
                "operation": "abs",
                "axes": [0],
                "distribution": "uniform",
                "entity_cfg": EntityCfg(name="robot"),
                "shared_random": True,
            },
        ),
        "foot_friction_spin": EventTermConfig(
            func=ef.randomize_friction,
            mode="startup",
            params={
                "ranges": (1e-4, 2e-2),
                "operation": "abs",
                "axes": [1],
                "distribution": "log_uniform",
                "entity_cfg": EntityCfg(name="robot", geom_names=foot_geom_names),
                "shared_random": True,
            },
        ),
        "foot_friction_roll": EventTermConfig(
            func=ef.randomize_friction,
            mode="startup",
            params={
                "ranges": (1e-5, 5e-3),
                "operation": "abs",
                "axes": [2],
                "distribution": "log_uniform",
                "entity_cfg": EntityCfg(name="robot", geom_names=foot_geom_names),
                "shared_random": True,
            },
        ),
    }
