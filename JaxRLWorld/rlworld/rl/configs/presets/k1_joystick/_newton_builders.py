"""Newton builders for the K1 joystick task.

Loads the vendored MJCF through SolverMuJoCo (mjwarp).

Solver recipe: implicitfast + pyramidal cone + impratio 1 + 1 substep
(``_SIM_TIMINGS``), matching mjlab's velocity task on the three knobs
that dominate the mjwarp step cost (substeps, cone, impratio) for ~2-3x
faster physics. iterations stay at the SolverMuJoCoCfg defaults 100/50.

CONTACT TRADE-OFF: an earlier recipe used elliptic cone + impratio 100
+ 2 substeps (Newton's own G1 locomotion recipe) to quiet stance-foot
micro-bounce/chatter that corrupts feet_air_time/feet_phase (see the
k1_feet_contact_parity diag). The current recipe trades some of that
contact quiet for speed — re-check gait quality if it regresses.

Feet contact tracking uses per-foot BODY matches (the four spheres
share the foot link) against the terrain plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

from rlworld.rl.actuators import DelayedPDActuatorCfg
from rlworld.rl.configs import TerminationTermConfig
from rlworld.rl.configs.common_config_classes import TerminationsConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonConfigsForRun,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    SolverMuJoCoCfg,
    VisualizationConfig,
)
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    InitialStateCfg,
    NewtonEntityCfg,
)
from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common.terminations import nan_detection
from rlworld.rl.envs.mdp.terminations.k1_locomotion import gravity_z_positive

if TYPE_CHECKING:
    from .base import K1JoystickConfig

CONFIGS_FOR_RUN_CLS = NewtonConfigsForRun
OBSERVATION_CFG_CLS = NewtonObservationConfig


def build_visualization(cfg: K1JoystickConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: K1JoystickConfig, timing: Dict[str, Any]) -> NewtonEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        fall = TerminationTermConfig(gravity_z_positive)
        nan = TerminationTermConfig(nan_detection)
        max_episode = TerminationTermConfig(max_episode_exceed)

    return NewtonEnvConfig(
        num_envs=cfg.num_envs,
        env_name="NewtonEnv",
        task_name="K1_Joystick",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: K1JoystickConfig, timing: Dict[str, Any]) -> NewtonSceneConfig:
    r = cfg.robot

    return NewtonSceneConfig(
        dt=timing["dt"],
        substeps=timing["substeps"],
        gravity=(0.0, 0.0, -9.81),
        solver_type="mujoco",
        robot_cfg=r,
        # Framework defaults = canonical mjwarp humanoid recipe
        # (implicitfast, elliptic, 100/50 iterations, impratio 100);
        # see the module docstring for why this departs from upstream.
        solver_cfg=SolverMuJoCoCfg(
            # mjlab-parity on the per-iteration-cost knobs (pyramidal cone +
            # impratio 1); with substeps=1 (_SIM_TIMINGS) these are the
            # recipe's dominant speedups. See module docstring for the contact
            # trade-off. iterations stay at the SolverMuJoCoCfg defaults 100/50.
            cone="pyramidal",
            impratio=1.0,
            ccd_iterations=50,
            # mjwarp constraint kernels run at njmax*nworld; the framework
            # default of 1500 rows/world is ~25x the measured demand here
            # (peak nefc/world 61 under random-action churn at 4096 envs;
            # mjlab auto-sizes 64 and trains fine at peak 48). 128 keeps a
            # 2x margin over the measured peak. nconmax stays at the
            # framework default: the contact-slot peak (170k of 614k) has
            # less headroom and contact kernels scale far less steeply.
            njmax=128,
        ),
        entities={
            "robot": NewtonEntityCfg(
                mjcf_path=r.mjcf_path,
                init_state=InitialStateCfg(
                    pos=(0.0, 0.0, r.base_init_height),
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
                            effort_limit=r.effort_limits,
                            tau_scale=r.tau_scale,
                            velocity_limit=r.velocity_limit,
                            knee_point_velocity=r.knee_point_velocity,
                            tau_lpf_time_constant=r.tau_lpf_time_constant,
                            physics_dt=timing["dt"],
                            dyn_gain=r.dyn_gain,
                            dyn_gain_velocity=r.dyn_gain_velocity,
                            frictionloss=0.1,
                            min_delay=cfg.action_delay_min,
                            max_delay=cfg.action_delay_max,
                        ),
                    ),
                    soft_joint_pos_limit_factor=0.95,
                ),
                body_label_prefix=r.name,
                # Foot boxes must collide with each other (collision
                # cost); the MJCF contype/conaffinity masks restrict
                # everything else.
                enable_self_collisions=True,
            ),
        },
        contact_sensors=[
            ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(mode="body", pattern=tuple(r.foot_names), entity="robot"),
                secondary=ContactMatch(mode="geom", pattern="ground_plane", entity="terrain"),
                history_length=timing["decimation"],
            ),
            ContactSensorCfg(
                name="feet_pair_contact",
                primary=ContactMatch(mode="body", pattern=(r.foot_names[0],), entity="robot"),
                secondary=ContactMatch(mode="body", pattern=(r.foot_names[1],), entity="robot"),
                history_length=timing["decimation"],
            ),
        ],
        env_spacing=(2.0, 2.0, 0.0),
    )


def build_action(cfg: K1JoystickConfig) -> NewtonActionConfig:
    r = cfg.robot
    return NewtonActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        # Physical actuator mode supplies a per-joint action scale (0.25·effort/kp);
        # it overrides the recipe's scale. legacy mode ⇒ None ⇒ recipe scale.
        action_scale=(r.physical_action_scale if r.physical_action_scale is not None else cfg.action_scale),
        clip_actions=cfg.action_clip,
        offset=r.get_action_offset(),
    )
