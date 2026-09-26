"""Newton builders for the Booster K1 velocity port.

The MJCF is loaded through SolverMuJoCo (mjwarp), so the contact parameters
baked into the asset's ``<default class="collision">`` blocks reach this
backend the same way they reach the MuJoCo one.

Self-collision is expressed differently here than on MuJoCo: there is no
subtree match, so the group is every robot body against the ``"self"``
sentinel, which means any other link of the same entity. The reward reads the
group as a binary any-self-contact signal, so the two forms agree in value.

Solver recipe mirrors the MuJoCo cell on the knobs that change contact
behaviour (pyramidal cone, impratio 1, one substep at 5 ms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

from jaxrlworld.rl.actuators import DelayedPDActuatorCfg
from jaxrlworld.rl.configs import TerminationTermConfig
from jaxrlworld.rl.configs.common_config_classes import TerminationsConfig
from jaxrlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonConfigsForRun,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    SolverMuJoCoCfg,
    VisualizationConfig,
)
from jaxrlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    InitialStateCfg,
    NewtonEntityCfg,
)
from jaxrlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from jaxrlworld.rl.envs.mdp.terminations import k1_velocity as booster_tf
from jaxrlworld.rl.envs.mdp.terminations.common import max_episode_exceed, terminations as common_tf
from jaxrlworld.rl.envs.mdp.terminations.common.terminations import nan_detection

if TYPE_CHECKING:
    from .base import K1VelocityConfig

CONFIGS_FOR_RUN_CLS = NewtonConfigsForRun
OBSERVATION_CFG_CLS = NewtonObservationConfig


def build_visualization(cfg: K1VelocityConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: K1VelocityConfig, timing: Dict[str, Any]) -> NewtonEnvConfig:
    limit_angle = math.radians(cfg.fall_limit_angle_deg)
    probability = cfg.fall_probability

    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        fell_over = TerminationTermConfig(
            booster_tf.stochastic_bad_orientation,
            {"limit_angle": limit_angle, "probability": probability},
        )
        illegal_contact = TerminationTermConfig(
            common_tf.illegal_contact,
            {"contact_group": "non_foot_ground_contact"},
        )
        # Not in the source recipe: this backend can produce a non-finite
        # state, and an environment that goes NaN poisons the rollout it is
        # in rather than simply failing.
        nan = TerminationTermConfig(nan_detection)
        max_episode = TerminationTermConfig(max_episode_exceed)

    return NewtonEnvConfig(
        num_envs=cfg.num_envs,
        env_name="NewtonEnv",
        task_name="K1_Booster_Velocity",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: K1VelocityConfig, timing: Dict[str, Any]) -> NewtonSceneConfig:
    r = cfg.robot
    history = timing["decimation"]

    return NewtonSceneConfig(
        dt=timing["dt"],
        substeps=timing["substeps"],
        gravity=(0.0, 0.0, -9.81),
        solver_type="mujoco",
        robot_cfg=r,
        solver_cfg=SolverMuJoCoCfg(
            cone="pyramidal",
            impratio=1.0,
            iterations=10,
            ls_iterations=20,
            ccd_iterations=50,
            # A full-body collision model with self-collision on generates far
            # more contact and constraint rows than the feet-only K1 asset, so
            # neither that preset's lean budgets nor the framework defaults
            # apply. mjwarp drops rows past the budget SILENTLY, which shows up
            # as feet sinking through the floor rather than as an error, so
            # these are set above the MuJoCo cell's and must be re-measured
            # with the contact-demand diagnostic after any Newton or
            # mujoco-warp bump.
            nconmax=220,
            njmax=500,
            use_mujoco_contacts=True,
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
                # Preserve the hand end-effector frames: they are welded
                # children of the hand links and carry collision geometry the
                # self-collision group must see.
                links_to_keep=("left_hand_end_ball_joint", "right_hand_end_ball_joint"),
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
                            frictionloss=r.joint_frictionloss,
                            min_delay=cfg.action_delay_min,
                            max_delay=cfg.action_delay_max,
                        ),
                    ),
                    soft_joint_pos_limit_factor=r.soft_joint_pos_limit_factor,
                ),
                body_label_prefix=r.name,
                enable_self_collisions=True,
            ),
        },
        contact_sensors=[
            ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(mode="body", pattern=tuple(r.foot_names), entity="robot"),
                secondary=ContactMatch(mode="geom", pattern="ground_plane", entity="terrain"),
                fields=("found", "force"),
                history_length=history,
            ),
            ContactSensorCfg(
                name="non_foot_ground_contact",
                primary=ContactMatch(
                    mode="body",
                    pattern=".*",
                    entity="robot",
                    exclude=tuple(r.foot_names),
                ),
                secondary=ContactMatch(mode="geom", pattern="ground_plane", entity="terrain"),
                fields=("found", "force"),
                history_length=history,
            ),
            ContactSensorCfg(
                name="self_collision",
                primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
                secondary=ContactMatch(mode="entity", entity="self"),
                fields=("found", "force"),
                history_length=history,
            ),
        ],
        env_spacing=(2.0, 2.0, 0.0),
    )


def build_action(cfg: K1VelocityConfig) -> NewtonActionConfig:
    r = cfg.robot
    return NewtonActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=r.physical_action_scale,
        clip_actions=cfg.action_clip,
        offset=r.get_action_offset(),
    )
