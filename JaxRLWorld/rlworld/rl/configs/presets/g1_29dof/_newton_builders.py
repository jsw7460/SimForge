"""Newton-specific builders for G1 29-DOF flat-terrain locomotion.

These functions are dispatched from ``G1FlatConfig.build()`` when
``sim_type == "newton"``. The bodies are extracted directly from the
old ``presets/g1_29dof/newton/base.py`` so the produced configs are
identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import warp as wp

from rlworld.rl.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from rlworld.rl.configs import RewardConfig, SolverMuJoCoCfg, TerminationTermConfig
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
from rlworld.rl.configs.scene import SceneEntitySelector
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    InitialStateCfg,
    NewtonEntityCfg,
)
from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg, NewtonIMUSensorConfig
from rlworld.rl.envs.mdp.events.dr import unified as unified_dr
from rlworld.rl.envs.mdp.observations.common.proprioception import (
    base_ang_vel,
    base_height,
    base_quat,
    command as command_obs,
    dof_pos,
    dof_pos_biased,
    dof_vel,
    foot_air_time,
    foot_contact_forces,
    foot_contact_indicator,
    foot_height,
    projected_gravity,
    raw_actions,
)
from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_mjlab
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed, terminations as common_tf

if TYPE_CHECKING:
    from .base import G1FlatConfig


# ── Module-level constants exposed to base.G1FlatConfig.build() ──────

CONFIGS_FOR_RUN_CLS = NewtonConfigsForRun
OBSERVATION_CFG_CLS = NewtonObservationConfig


def _initial_quat() -> Any:
    """Initial yaw of the robot at reset (no rotation for G1)."""
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.0)


# ── Builders ─────────────────────────────────────────────────────────


def build_visualization(cfg: G1FlatConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: G1FlatConfig, timing: Dict[str, Any]) -> NewtonEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        roll_pitch = TerminationTermConfig(
            common_tf.roll_pitch_violation,
            {"roll_threshold_degree": 70.0, "pitch_threshold_degree": 70.0},
        )
        max_episode = TerminationTermConfig(max_episode_exceed)

        # On generated (finite) terrain, reset the robot before it walks
        # off the mesh edge into the void (matches IsaacLab / mjlab). Only
        # registered for rough so the flat preset's terminations are
        # unchanged.
        if cfg.use_rough_terrain:
            out_of_terrain_bounds = TerminationTermConfig(
                common_tf.terrain_out_of_bounds,
                {"margin": 0.5},
            )

    return NewtonEnvConfig(
        num_envs=cfg.num_envs,
        env_name="NewtonEnv",
        task_name="G1_Velocity_Tracking",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: G1FlatConfig, timing: Dict[str, Any]) -> NewtonSceneConfig:
    r = cfg.robot
    quat = _initial_quat()

    # explicit-PD collection uses the explicit-PD path (no command
    # delay) so kp/kd map onto a clean torque computation; training keeps
    # the trained DelayedPD actuator. Per-joint PD overrides (when set on
    # the robot config) replace the nominal p_gains / d_gains — the
    # actuator's stiffness / damping accept a {joint_regex: value} map
    # natively, so heterogeneous per-joint PD ships without rewiring.
    # Flat follows the Mjlab-Velocity-Flat-Unitree-G1 reference: implicit
    # (mjwarp builtin) position actuators. Rough keeps the DelayedPD
    # sim2real modeling; explicit-PD collection keeps IdealPD.
    if cfg.use_ideal_pd_actuator:
        ActuatorCls, _delay_kwargs = IdealPDActuatorCfg, {}
    elif cfg.use_rough_terrain:
        ActuatorCls, _delay_kwargs = DelayedPDActuatorCfg, {"min_delay": 0, "max_delay": 2}
    else:
        ActuatorCls, _delay_kwargs = ImplicitActuatorCfg, {}
    stiffness = r.kp_per_dof_override if r.kp_per_dof_override is not None else r.p_gains
    damping = r.kd_per_dof_override if r.kd_per_dof_override is not None else r.d_gains

    return NewtonSceneConfig(
        dt=timing["dt"],
        substeps=timing["substeps"],
        gravity=(0.0, 0.0, -9.81),
        solver_type="mujoco",
        robot_cfg=r,
        solver_cfg=SolverMuJoCoCfg(
            cone="pyramidal",
            # Paired with the cone, not left at the default. SolverMuJoCoCfg
            # defaults to 100 because that is Newton's canonical humanoid
            # recipe — and that recipe uses an ELLIPTIC cone. Taking the
            # pyramidal half of the fast recipe while leaving impratio on the
            # stiff one gives a combination nobody chose: friction constraints
            # a hundred times stiffer than normal ones, solved with fewer
            # iterations than either recipe asks for.
            #
            # It is also the only place the three backends disagree on this
            # knob. MuJoCo's default is 1, mjlab holds it at 1 (it overrides
            # the value go2's own XML declares), Genesis leaves it unset, and
            # K1's Newton preset moved to 1 deliberately. Only g1 ran at 100,
            # and only Newton NaN'd — mid-training, one environment in 8192,
            # after a single substep took a joint from 6 to 64 rad/s and the
            # whole 29-DOF state went non-finite two substeps later while the
            # velocities were already coming back down. That is a constraint
            # solve failing to condition, not a joint diverging.
            impratio=1.0,
            iterations=50 if cfg.use_rough_terrain else 10,
            ls_iterations=50 if cfg.use_rough_terrain else 20,
            # Down from the canonical recipe's 50: under the new mjwarp
            # the EPA scratch is 6 arrays totalling num_envs x nconmax x
            # (280 + 132 x ccd_iterations) bytes -- 24.8 GB at 16384
            # envs x nconmax 220 (OOM), 17.7 GB at MuJoCo's default 35
            # (env built, then the CUDA graph exec OOMed), 12.4 GB at
            # 24. EPA only serves CONVEX (mesh) pairs -- the feet are
            # capsules on primitive paths -- and upstream itself runs
            # epa_iterations=16 for box-box scenes, so 24 stays above
            # upstream's own floor.
            ccd_iterations=24,
            # Rough terrain inflates the per-env contact count by an order
            # of magnitude vs flat ground (observed on G1 29-DOF +
            # self-collision + heightfield). ``nconmax`` / ``njmax`` are
            # mjwarp's *per-env* solver buffers (total = num_envs * field).
            #
            # Flat budget: measured with the contact-demand sweep in
            # ``scripts/diag/newton_g1_dr_nan_diag.py`` (2026-07-16,
            # Newton 1.5.0.dev + mujoco-warp 3.10.0.2): peak per-world
            # ncon = 76 under random actions, ~60 just STANDING. The old
            # value 35 dated from an earlier Newton/mjwarp combination
            # that counted fewer contacts for the same scene; after the
            # upstream collision/margin reworks it overflowed even at
            # rest. mjwarp SKIPS contacts beyond the budget silently
            # (collision_driver: "the remaining contacts will be
            # skipped"), so undersizing shows up not as an error but as
            # feet losing contact -> penetration -> velocity blowup ->
            # NaN within ~1 s of training. Re-measure with the diag
            # after any Newton / mujoco-warp bump.
            # Rough budgets re-measured 2026-07-18 under the mjwarp
            # HFIELD contact path (use_mujoco_contacts=True): peak
            # per-world ncon = 58, peak per-world nefc = 167 (4096 envs,
            # 300 random-action steps, full DR) — ~8.6x below the old
            # Newton-MPR-era demand (495). Keep nconmax lean on this
            # path: it also multiplies mjwarp's GJK/EPA scratch
            # (naccdmax defaults to naconmax; ~naconmax * 5 *
            # ccd_iterations vec3s — nconmax=4000 tried to allocate
            # 50 GB).
            njmax=1500 if cfg.use_rough_terrain else 300,
            # Flat re-measured 2026-08-27 after the env rebuild (Warp
            # 1.16.0 combo). Two different demands, do not confuse them:
            # narrowphase peak per-world ncon = 79 (full_100 cell, 4096
            # envs, random actions), but the buffer ALSO caps BROADPHASE
            # candidate pairs, and a 16384-env training launch demanded
            # 135-153 live for those. Undersizing does not fail -- the
            # excess pairs are silently dropped, feet penetrate, the
            # pair count then CREEPS upward every step (measured 151 ->
            # 153; 150 fed that spiral). 220 = observed demand x ~1.4.
            # The ceiling is the EPA scratch bill (num_envs x nconmax x
            # 5 x ccd_iterations vec3s): ~10.8 GB at 16384 envs. nefc
            # peak 151 < njmax 300, unchanged.
            # Rough re-measured 2026-08-27 on the same combo: peak
            # per-world ncon = 55, nefc = 196 (2048 envs, full_100) --
            # but the LIVE broadphase demand at the training scale of
            # 16384 envs is ~153/world on BOTH terrains (the max over
            # that many worlds swamps the terrain difference), so both
            # paths get the same 220. The old 300 dated from the
            # Newton-MPR contact stream and, under the new mjwarp's
            # fatter EPA scratch row (1024 vec3 per slot, was ~250),
            # OOMed at env build; 220 keeps the scratch at ~11 GB for
            # 16384 envs (naccdmax = naconmax / 4 in this version).
            nconmax=220,
            # mjwarp-native collision on BOTH flat and rough. The old
            # rough-only opt-out (use_mujoco_contacts=False → Newton's
            # MPR collide()) dated from when the terrain was a triangle
            # MESH, which mjwarp's GJK/EPA path genuinely mishandled.
            # The terrain importer now emits a NATIVE heightfield, which
            # SolverMuJoCo converts to a first-class mjwarp HFIELD geom
            # with dedicated primitive collision — the same path mjlab
            # runs. Newton's own MPR heightfield midphase enumerates
            # triangle candidates by XY footprint with NO z-culling
            # (elevated trunk/hip shapes sweep in millions of dead
            # triangle pairs), which made rough 2-3x slower than
            # genesis/mjlab (go2 rough profiling 2026-07-18); mjwarp's
            # 3D AABB broadphase culls those for free. NOTE: the rough
            # nconmax/njmax above were sized for the Newton-MPR contact
            # stream — re-measure with the diag under this path.
            use_mujoco_contacts=True,
        ),
        terrain_cfg=cfg.make_terrain_cfg(),
        entities={
            "robot": NewtonEntityCfg(
                mjcf_path=r.mjcf_path,
                init_state=InitialStateCfg(
                    pos=(0.0, 0.0, r.base_init_height),
                    rot=(quat[0], quat[1], quat[2], quat[3]),
                    joint_pos=r.default_joint_angles,
                ),
                floating=True,
                collapse_fixed_joints=True,
                # Preserve the dummy foot-pad frame bodies (welded children of
                # left/right ankle_roll_link in g1.xml). Without this Newton's
                # collapse_fixed_joints merges them into the parent and the
                # feet rewards can't read them. ``links_to_keep`` matches
                # against Newton joint labels — Newton's MJCF importer auto-
                # names the implicit fixed joint ``<body_name>_joint``.
                links_to_keep=("left_foot_frame_joint", "right_foot_frame_joint"),
                articulation=ArticulationCfg(
                    actuators=(
                        ActuatorCls(
                            target_names_expr=(".*",),
                            stiffness=stiffness,
                            damping=damping,
                            armature=r.armature,
                            frictionloss=0.3,
                            **_delay_kwargs,
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
        ],
        contact_sensors=[
            ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(mode="body", pattern=tuple(r.foot_names), entity="robot"),
                secondary=ContactMatch(mode="geom", pattern="ground_plane", entity="terrain"),
                history_length=timing["decimation"],
            ),
            ContactSensorCfg(
                name="self_collision",
                primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
                secondary=ContactMatch(mode="entity", entity="self"),
                history_length=timing["decimation"],
            ),
        ],
        env_spacing=(2.0, 2.0, 0.0),
        # Rough terrain emits many heightfield triangle pairs in
        # ``model.collide()``; size the broad-phase buffer to the env
        # count so contacts aren't silently dropped. ``None`` on flat
        # ground keeps Newton's default.
        collision_max_triangle_pairs=(
            cfg.num_envs * cfg.terrain_collision_pairs_per_env if cfg.use_rough_terrain else None
        ),
    )


def build_observation(cfg: G1FlatConfig) -> NewtonObservationConfig:
    feet_bodies = tuple(cfg.robot.foot_names)

    @dataclass
    class _ActorObsCfg(ObservationGroupConfig):
        base_ang_vel_obs = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
        projected_gravity_obs = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        # ``dof_pos_biased`` = robot_data.joint_pos + act_manager.encoder_bias
        # (per-episode static offset from randomize_encoder_bias DR). Critic
        # below keeps unbiased ``dof_pos``.
        dof_pos_obs = ObservationTermConfig(func=dof_pos_biased, scale=1.0, noise=Unoise(-0.01, 0.01))
        dof_vel_obs = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)

    @dataclass
    class _CriticObsCfg(ObservationGroupConfig):
        enable_corruption: bool = False
        base_ang_vel = ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2))
        projected_gravity = ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05))
        command = ObservationTermConfig(func=command_obs, scale=1.0)
        dof_pos = ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01))
        prev_actions = ObservationTermConfig(func=raw_actions, scale=1.0)
        dof_vel = ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5))
        base_height_obs = ObservationTermConfig(func=base_height, scale=1.0)
        base_quat_obs = ObservationTermConfig(func=base_quat, scale=1.0)
        foot_height_obs = ObservationTermConfig(func=foot_height, scale=1.0, params={"body_names": feet_bodies})
        foot_air_time_obs = ObservationTermConfig(
            func=foot_air_time,
            scale=1.0,
            params={
                "contact_group": "feet_ground_contact",
                "body_names": feet_bodies,
                "use_last": True,
            },
        )
        foot_contact_obs = ObservationTermConfig(
            func=foot_contact_indicator,
            scale=1.0,
            params={"contact_group": "feet_ground_contact", "body_names": feet_bodies},
        )
        foot_contact_forces_obs = ObservationTermConfig(
            func=foot_contact_forces,
            scale=0.01,
            params={"contact_group": "feet_ground_contact", "body_names": feet_bodies},
        )

    @dataclass
    class _ObsCfg(NewtonObservationConfig):
        actor: _ActorObsCfg = field(default_factory=_ActorObsCfg)
        critic: _CriticObsCfg = field(default_factory=_CriticObsCfg)

    return _ObsCfg()


def build_action(cfg: G1FlatConfig) -> NewtonActionConfig:
    r = cfg.robot
    return NewtonActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=r.action_scale,
        clip_actions=(-100.0, 100.0),
        offset=r.get_action_offset(),
    )


def build_reward(cfg: G1FlatConfig) -> RewardConfig:
    r = cfg.robot

    @dataclass
    class _RewardsCfg(RewardConfig):
        # Tracking rewards (common -- uses RobotData interface)
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

        # Orientation (common -- uses RobotData interface)
        flat_orientation = RewardTermConfig(
            func=rf_common.flat_orientation,
            weight=1.0,
            params={"std": 0.447},
        )

        # Posture (stateful class)
        variable_posture = RewardTermConfig(
            func=rf_mjlab.variable_posture,
            weight=1.0,
            params={
                "std_standing": {".*": 0.05},
                "std_walking": {
                    r".*hip_pitch.*": 0.3,
                    r".*hip_roll.*": 0.15,
                    r".*hip_yaw.*": 0.15,
                    r".*knee.*": 0.35,
                    r".*ankle_pitch.*": 0.25,
                    r".*ankle_roll.*": 0.1,
                    # Waist.
                    r".*waist_yaw.*": 0.2,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.1,
                    # Arms.
                    r".*shoulder_pitch.*": 0.15,
                    r".*shoulder_roll.*": 0.15,
                    r".*shoulder_yaw.*": 0.1,
                    r".*elbow.*": 0.15,
                    r".*wrist.*": 0.3,
                },
                "std_running": {
                    # Lower body.
                    r".*hip_pitch.*": 0.5,
                    r".*hip_roll.*": 0.2,
                    r".*hip_yaw.*": 0.2,
                    r".*knee.*": 0.6,
                    r".*ankle_pitch.*": 0.35,
                    r".*ankle_roll.*": 0.15,
                    # Waist.
                    r".*waist_yaw.*": 0.3,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.2,
                    # Arms.
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

        # Self-collision
        self_collision_cost = RewardTermConfig(
            func=rf_common.penalize_any_contact_force,
            weight=1.0,
            params={"contact_group": "self_collision", "force_threshold": 10.0},
        )

        # Penalties
        body_angular_velocity_penalty = RewardTermConfig(
            func=rf_mjlab.body_ang_vel_penalty_mjlab,
            weight=0.05,
            params={"asset_cfg": SceneEntitySelector(name="robot", body_names=("torso_link",))},
        )
        angular_momentum_penalty = RewardTermConfig(
            func=rf_mjlab.angular_momentum_penalty,
            weight=0.02,
        )
        joint_pos_limits = RewardTermConfig(
            func=rf_mjlab.joint_pos_limits_mjlab,
            weight=1.0,
        )
        raw_action_rate_l2 = RewardTermConfig(
            func=rf_mjlab.raw_action_rate_l2_mjlab,
            weight=0.1,
        )

        # Feet rewards — position/velocity reads use the foot-pad frame body
        # (welded child of ankle_roll_link at +0.04m fore / -0.037m sole;
        # matches mjlab's `left_foot` site so values agree across sims).
        # Contacts still come from ankle_roll_link (the frame body has no
        # collision geom), so we pass contact_order explicitly.
        feet_selector = SceneEntitySelector(
            name="robot",
            body_names=("left_foot_frame", "right_foot_frame"),
            preserve_order=True,
        )
        feet_contact_order = list(r.foot_names)
        feet_clearance = RewardTermConfig(
            func=rf_mjlab.feet_clearance_mjlab,
            weight=2.0,
            params={
                "asset_cfg": feet_selector,
                "target_height": 0.1,
                "command_threshold": 0.05,
            },
        )
        feet_swing_height = RewardTermConfig(
            func=rf_mjlab.feet_swing_height_mjlab,
            weight=0.25,
            params={
                "asset_cfg": feet_selector,
                "target_height": 0.1,
                "command_threshold": 0.05,
                "contact_order": feet_contact_order,
            },
        )
        feet_slip = RewardTermConfig(
            func=rf_mjlab.feet_slip_mjlab,
            weight=0.1,
            params={
                "asset_cfg": feet_selector,
                "command_threshold": 0.05,
                "contact_order": feet_contact_order,
            },
        )
        soft_landing = RewardTermConfig(
            func=rf_mjlab.soft_landing_mjlab,
            weight=1e-5,
            params={
                "feet_bodies": r.foot_names,
                "command_threshold": 0.05,
            },
        )

    return _RewardsCfg()


def build_dr_terms(cfg: G1FlatConfig) -> Dict[str, EventTermConfig]:
    """Newton-specific domain randomization terms."""
    r = cfg.robot
    return {
        "randomize_encoder_bias": EventTermConfig(
            func=unified_dr.randomize_encoder_bias,
            mode="reset_dr",
            params={
                "asset_cfg": SceneEntitySelector(name="robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "randomize_body_com": EventTermConfig(
            func=unified_dr.randomize_body_com_offset,
            mode="reset_dr",
            params={
                "asset_cfg": SceneEntitySelector(name="robot", body_names=("torso_link",)),
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
                "operation": "add",
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
