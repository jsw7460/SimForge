"""Genesis builders for the Booster K1 velocity port.

Genesis reads the same MJCF and runs the same MDP on its own solver.

``convexify=True`` is not optional for this asset. With it off Genesis checks
whether each of a link's collision geoms can be convexified alone and, failing
that, MERGES every geom on the link into one shape. The trunk and shanks each
carry two collision geoms, so the merge would silently change their contact
geometry and cost several times the step time.

Self-collision uses the ``"self"`` sentinel rather than a subtree match, which
this backend has no equivalent for. The reward reads the group as a binary
any-self-contact signal, so it agrees in value with the MuJoCo cell's
subtree-against-itself pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

import genesis as gs

from jaxrlworld.rl.actuators import DelayedPDActuatorCfg
from jaxrlworld.rl.configs import TerminationTermConfig
from jaxrlworld.rl.configs.common_config_classes import TerminationsConfig
from jaxrlworld.rl.configs.genesis_config_classes import (
    ActionConfig,
    EnvConfig,
    GenesisConfigsForRun,
    ObservationConfig,
    SceneConfig,
    VisualizationConfig,
)
from jaxrlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    GenesisEntityCfg,
    InitialStateCfg,
)
from jaxrlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from jaxrlworld.rl.envs.mdp.terminations import k1_velocity as booster_tf
from jaxrlworld.rl.envs.mdp.terminations.common import max_episode_exceed, terminations as common_tf
from jaxrlworld.rl.envs.mdp.terminations.common.terminations import nan_detection

if TYPE_CHECKING:
    from .base import K1VelocityConfig

CONFIGS_FOR_RUN_CLS = GenesisConfigsForRun
OBSERVATION_CFG_CLS = ObservationConfig


def build_visualization(cfg: K1VelocityConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: K1VelocityConfig, timing: Dict[str, Any]) -> EnvConfig:
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
        nan = TerminationTermConfig(nan_detection)
        max_episode = TerminationTermConfig(max_episode_exceed)

    return EnvConfig(
        num_envs=cfg.num_envs,
        env_name="GenesisEnv",
        task_name="K1_Booster_Velocity",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: K1VelocityConfig, timing: Dict[str, Any]) -> SceneConfig:
    r = cfg.robot
    sim_dt = timing["dt"]
    history = timing["decimation"]

    return SceneConfig(
        entities={
            "robot": GenesisEntityCfg(
                mjcf_path=r.mjcf_path,
                init_state=InitialStateCfg(
                    pos=(0.0, 0.0, r.base_init_height),
                    joint_pos=r.default_joint_angles,
                ),
                floating=True,
                enable_self_collisions=True,
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
                            physics_dt=sim_dt,
                            dyn_gain=r.dyn_gain,
                            dyn_gain_velocity=r.dyn_gain_velocity,
                            frictionloss=r.joint_frictionloss,
                            min_delay=cfg.action_delay_min,
                            max_delay=cfg.action_delay_max,
                        ),
                    ),
                    soft_joint_pos_limit_factor=r.soft_joint_pos_limit_factor,
                ),
                # See the module docstring: False would merge the multi-geom
                # links rather than skip a convex pass.
                convexify=True,
                visualize_contact=False,
            ),
        },
        sensors=[],
        contact_sensors=[
            ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(mode="body", pattern=tuple(r.foot_names), entity="robot"),
                secondary=ContactMatch(mode="entity", entity="terrain"),
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
                secondary=ContactMatch(mode="entity", entity="terrain"),
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
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=timing["substeps"]),
        rigid_options=gs.options.RigidOptions(
            dt=sim_dt,
            # Genesis defaults to approximate_implicitfast, which folds joint
            # damping into the mass matrix before the constraint solve. That is
            # extra effective damping the other two backends do not apply, and
            # it shows up as smaller limb amplitudes and skewed contact-event
            # rewards.
            integrator=gs.integrator.implicitfast,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=10,
            ls_iterations=20,
            tolerance=1e-5,
            constraint_timeconst=0.02,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            # Raised over the feet-only K1 preset's budget: this asset collides
            # on 22 geoms with self-collision on.
            max_collision_pairs=200,
            batch_dofs_info=True,
            batch_links_info=True,
            contact_pruning_tolerance=None,
        ),
        robot_cfg=r,
    )


def build_action(cfg: K1VelocityConfig) -> ActionConfig:
    r = cfg.robot
    return ActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=r.physical_action_scale,
        clip_actions=cfg.action_clip,
        offset=r.get_action_offset(),
    )
