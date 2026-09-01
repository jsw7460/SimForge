"""Genesis builders for the K1 joystick task.

Genesis integrates the same MDP definition on its own solver.
``enable_self_collisions=True`` is required so the foot boxes can touch
(collision cost); the MJCF contype/conaffinity masks are expected to
suppress every other intra-robot pair — the parity diag verifies that
no non-foot geom ever reaches the floor or another robot geom.
"""

from __future__ import annotations

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
from jaxrlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from jaxrlworld.rl.envs.mdp.terminations.common.terminations import nan_detection
from jaxrlworld.rl.envs.mdp.terminations.k1_locomotion import gravity_z_positive

if TYPE_CHECKING:
    from .base import K1JoystickConfig

CONFIGS_FOR_RUN_CLS = GenesisConfigsForRun
OBSERVATION_CFG_CLS = ObservationConfig


def build_visualization(cfg: K1JoystickConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: K1JoystickConfig, timing: Dict[str, Any]) -> EnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        fall = TerminationTermConfig(gravity_z_positive)
        nan = TerminationTermConfig(nan_detection)
        max_episode = TerminationTermConfig(max_episode_exceed)

    return EnvConfig(
        num_envs=cfg.num_envs,
        env_name="GenesisEnv",
        task_name="K1_Joystick",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: K1JoystickConfig, timing: Dict[str, Any]) -> SceneConfig:
    r = cfg.robot
    sim_dt = timing["dt"]

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
                convexify=True,
                visualize_contact=False,
            ),
        },
        contact_sensors=[
            ContactSensorCfg(
                name="feet_ground_contact",
                primary=ContactMatch(mode="body", pattern=tuple(r.foot_names), entity="robot"),
                secondary=ContactMatch(mode="entity", entity="terrain"),
                history_length=timing["decimation"],
            ),
            ContactSensorCfg(
                name="feet_pair_contact",
                primary=ContactMatch(mode="body", pattern=(r.foot_names[0],), entity="robot"),
                secondary=ContactMatch(mode="body", pattern=(r.foot_names[1],), entity="robot"),
                # Only the boolean is read — the collision penalty asks
                # whether the feet touched, never how hard. mjlab's K1
                # already declared this; Genesis and Newton were left on
                # the default and so disagreed about what the same group
                # produces. On Genesis the force is a signed accumulation
                # over the contact list plus a rotation into the link
                # frame, per substep, for nobody.
                fields=("found",),
                history_length=timing["decimation"],
            ),
        ],
        sim_options=gs.options.SimOptions(
            dt=sim_dt,
            substeps=timing["substeps"],
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            dt=sim_dt,
            # approximate_implicitfast (the Genesis default) is deliberate.
            # implicitfast is the MuJoCo-consistent integrator and closes
            # most of the per-step kinematic gap to the mjlab/newton cells
            # (limb speeds, ground-contact fraction), but training this
            # preset under it fails: velocity tracking never rises. The
            # approximation folds joint damping into the mass matrix
            # before the constraint solve; the extra effective damping
            # costs a small (~4-10%) kinematic offset versus mjlab/newton,
            # accepted in exchange for stable learning.
            integrator=gs.integrator.approximate_implicitfast,
            constraint_timeconst=0.02,
            enable_self_collision=True,
            box_box_detection=True,
            # Per-env dofs_info (frictionloss/armature DR writes
            # (n_envs, n_dofs) tensors; default False keeps dofs_info
            # shared across envs and rejects batched writes).
            batch_dofs_info=True,
            batch_links_info=True,
            contact_pruning_tolerance=None,
        ),
    )


def build_action(cfg: K1JoystickConfig) -> ActionConfig:
    r = cfg.robot
    return ActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        # Physical actuator mode supplies a per-joint action scale (0.25·effort/kp);
        # it overrides the recipe's scale. legacy mode ⇒ None ⇒ recipe scale.
        action_scale=(r.physical_action_scale if r.physical_action_scale is not None else cfg.action_scale),
        clip_actions=cfg.action_clip,
        offset=r.get_action_offset(),
    )
