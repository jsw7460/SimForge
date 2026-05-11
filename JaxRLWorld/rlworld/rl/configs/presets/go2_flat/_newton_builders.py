"""Newton-specific builders for Go2 flat-terrain locomotion.

These functions are dispatched from ``Go2FlatConfig.build()`` when
``sim_type == "newton"``. The bodies are extracted directly from the
old ``presets/go2_flat/newton/base.py`` so the produced configs are
identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import warp as wp

from rlworld.rl.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
from rlworld.rl.configs import RewardConfig, TerminationTermConfig
from rlworld.rl.configs.common_config_classes import TerminationsConfig
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonConfigsForRun,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    SolverMuJoCoCfg,
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
from rlworld.rl.configs.scene import SceneEntitySelector
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    GroundPlaneCfg,
    InitialStateCfg,
    NewtonEntityCfg as UnifiedNewtonEntityCfg,
)
from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg, NewtonIMUSensorConfig
from rlworld.rl.envs.mdp.events.dr import unified as unified_dr
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed, terminations as common_tf

if TYPE_CHECKING:
    from .base import Go2FlatConfig


# ── Module-level constants exposed to base.Go2FlatConfig.build() ─────

CONFIGS_FOR_RUN_CLS = NewtonConfigsForRun
OBSERVATION_CFG_CLS = NewtonObservationConfig


def get_foot_names(robot) -> tuple[str, ...]:
    """Newton uses prefixed foot names (e.g. ``go2_description/FL_foot``)."""
    return robot.foot_names


def _initial_quat() -> Any:
    """Identity quaternion at reset (no yaw applied).

    Newton's wp.quat layout is xyzw, so (0, 0, 0, 1) is the identity.
    Removing the previous 90° yaw aligns Newton's reset frame with the
    other simulators (Genesis / mjlab) so per-step quantities like
    foot xy velocity are directly comparable.
    """
    return wp.quat(0.0, 0.0, 0.0, 1.0)


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: Go2FlatConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: Go2FlatConfig, timing: Dict[str, Any]) -> NewtonEnvConfig:
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


def build_scene(cfg: Go2FlatConfig, timing: Dict[str, Any]) -> NewtonSceneConfig:
    r = cfg.robot
    quat = _initial_quat()

    # Resolve PD constants — fall back to module-level defaults from
    # ``rl/configs/robots/go2.py`` when the corresponding SysID-result
    # override on ``Go2Config`` is unset (``None``). When set, the
    # override takes precedence so PPO sees the identified PD gains
    # from step 0 of training. friction overrides are applied via
    # event terms (see ``build_dr_terms`` below) since neither the
    # actuator nor the scene config exposes friction fields.
    stiffness_hip = r.kp_hip_override if r.kp_hip_override is not None else STIFFNESS_HIP
    damping_hip = r.kd_hip_override if r.kd_hip_override is not None else DAMPING_HIP
    stiffness_knee = r.kp_knee_override if r.kp_knee_override is not None else STIFFNESS_KNEE
    damping_knee = r.kd_knee_override if r.kd_knee_override is not None else DAMPING_KNEE

    ActuatorCls, _delay_kwargs = (
        (IdealPDActuatorCfg, {})
        if cfg.use_ideal_pd_actuator
        else (DelayedPDActuatorCfg, {"min_delay": 1, "max_delay": 3})
    )

    return NewtonSceneConfig(
        dt=timing["dt"],
        substeps=timing["substeps"],
        gravity=(0.0, 0.0, -9.81),
        solver_type="mujoco",
        solver_cfg=SolverMuJoCoCfg(impratio=100.0, ccd_iterations=50, cone="elliptic", ls_iterations=20, iterations=10),
        entities={
            "ground": GroundPlaneCfg(),
            "robot": UnifiedNewtonEntityCfg(
                # urdf_path=r.urdf_path,
                mjcf_path=r.mjcf_path,
                init_state=InitialStateCfg(
                    pos=(0.0, 0.0, r.base_init_height),
                    rot=(quat[0], quat[1], quat[2], quat[3]),
                    joint_pos=r.default_joint_angles,
                ),
                floating=True,
                enable_self_collisions=True,
                collapse_fixed_joints=True,
                links_to_keep=[
                    "FL_foot_joint",
                    "FR_foot_joint",
                    "RL_foot_joint",
                    "RR_foot_joint",
                ],
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
                sites={"imu_site_base": r.base_link_name},
            ),
        },
        sensors=[
            NewtonIMUSensorConfig(
                entity_name="robot",
                sensor_name="imu_base",
                site_names=["imu_site_base"],
            ),
        ],
        # Simulator-agnostic contact sensors (same ContactSensorCfg the
        # Genesis / mjlab go2_flat builders use). Newton resolves the
        # ``secondary`` whitelist directly (no inversion). The Newton
        # ground plane is a single global *shape* named "ground_plane"
        # with no parent body, so the ground secondary uses mode="geom".
        # ``exclude`` patterns here are regex (the simulator-agnostic
        # convention) — the Newton backend resolves them against the
        # model labels itself rather than handing them to SensorContact's
        # fnmatch. ``history_length=decimation`` keeps one policy step of
        # substep contact forces for ``penalize_contact_force_count``.
        contact_sensors=[
            ContactSensorCfg(
                name="foot_contact",
                # Foot names ("FL_foot", ...) have no regex metacharacters
                # → matched as exact leaf names.
                primary=ContactMatch(mode="body", pattern=tuple(r.foot_names), entity="robot"),
                secondary=ContactMatch(mode="geom", pattern="ground_plane", entity="ground"),
                history_length=timing["decimation"],
                track_air_time=True,
            ),
            ContactSensorCfg(
                name="body_ground_contact",
                primary=ContactMatch(
                    mode="body",
                    pattern=".*",
                    entity="robot",
                    exclude=(".*foot.*", ".*calf.*"),
                ),
                secondary=ContactMatch(mode="geom", pattern="ground_plane", entity="ground"),
                history_length=timing["decimation"],
            ),
        ],
        add_ground=True,
        env_spacing=(2.0, 2.0, 0.0),
        robot_cfg=r,
    )


def build_action(cfg: Go2FlatConfig) -> NewtonActionConfig:
    r = cfg.robot
    return NewtonActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=GO2_ACTION_SCALE,
        clip_actions=(-100.0, 100.0),
        offset=r.get_action_offset(),
    )


def build_reward(cfg: Go2FlatConfig) -> RewardConfig:
    r = cfg.robot
    feet = list(r.foot_names)

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


def customize_reset_root_params(cfg: Go2FlatConfig, params: Dict[str, Any]) -> None:
    """Newton hook: inject wxyz default quat (Newton native is xyzw)."""
    _iq = _initial_quat()
    _iq_tuple = tuple(float(v) for v in _iq)  # (x, y, z, w)
    params["default_quat_wxyz"] = (
        _iq_tuple[3],
        _iq_tuple[0],
        _iq_tuple[1],
        _iq_tuple[2],
    )


def build_dr_terms(cfg: Go2FlatConfig) -> Dict[str, EventTermConfig]:
    """Newton-specific domain randomization terms.

    Layered scheme:

    1. ``randomize_body_mass`` always installed — mass DR'd ±10 %
       around the URDF's base body value (which equals the identified
       value when used with the SysID-aligned training script's
       ``URDF_PATH``).
    2. ``set_foot_friction`` / ``set_joint_friction`` installed only
       when the matching ``Go2Config.*_override`` is set. Each runs
       every reset with a multiplicative ``dr_scale=(0.9, 1.1)`` band
       so the friction axes also get ±10 % DR centered on their
       identified value.

    The legacy ``randomize_friction`` and ``randomize_joint_friction``
    DR terms (with absolute ranges ``(0.3, 1.2)`` / ``(0.0, 0.05)``) are
    no longer installed — when SysID overrides are active the friction
    center is the identified value, and when they aren't, friction is
    left at URDF defaults rather than randomly scattered. This keeps
    the DR scope tight enough that downstream sim2real comparisons
    actually reflect the SysID center rather than a wide random band.
    """
    from rlworld.rl.envs.mdp.events.dr import newton as newton_dr

    r = cfg.robot
    terms: Dict[str, EventTermConfig] = {
        "randomize_body_mass": EventTermConfig(
            func=unified_dr.randomize_body_mass,
            mode="reset_dr",
            params={
                "asset_cfg": SceneEntitySelector(name="robot", body_names=(cfg.robot.base_link_name,)),
                "mass_range": (0.8, 1.2),
                "operation": "scale",
            },
        ),
        # mjlab parity foot-friction DR: same range / abs / shared_random
        # pattern as mjlab's randomize_friction in _mujoco_builders. The
        # ``.*/<name>`` leaf-regex form matches Newton's MJCF XPath
        # labels (``go2/worldbody/.../FR_foot/FR_foot_collision`` etc.).
        "randomize_friction": EventTermConfig(
            func=unified_dr.randomize_friction,
            mode="reset_dr",
            params={
                "asset_cfg": SceneEntitySelector(
                    name="robot",
                    geom_names=(
                        ".*/FR_foot_collision",
                        ".*/FL_foot_collision",
                        ".*/RR_foot_collision",
                        ".*/RL_foot_collision",
                    ),
                ),
                "friction_range": (0.3, 1.2),
                "operation": "abs",
                "shared_random": True,
            },
        ),
        # mjlab parity joint-friction DR: same abs (0.0, 0.05) range and
        # whole-DOF scope as mjlab's randomize_joint_friction term.
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

    # SysID-aligned friction terms — each fixes its axis at the
    # identified value with ±10 % multiplicative DR. Installed only
    # when the corresponding override field is set, so vanilla
    # training runs (no SysID injection) get only the body_mass DR
    # above and otherwise inherit URDF defaults.
    if r.foot_friction_override is not None:
        terms["set_foot_friction"] = EventTermConfig(
            func=newton_dr.set_foot_friction,
            mode="reset_dr",
            params={
                "value": float(r.foot_friction_override),
                "foot_pattern": ".*foot$",
                "dr_scale": (0.9, 1.1),
            },
        )
    if r.joint_frictionloss_override is not None:
        terms["set_joint_friction"] = EventTermConfig(
            func=newton_dr.set_joint_friction,
            mode="reset_dr",
            params={
                "value": float(r.joint_frictionloss_override),
                "dr_scale": (0.9, 1.1),
            },
        )
    return terms
