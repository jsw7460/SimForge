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

from rlworld.rl.actuators import IdealPDActuatorCfg
from rlworld.rl.configs import TerminationTermConfig
from rlworld.rl.configs.common_config_classes import TerminationsConfig
from rlworld.rl.configs.genesis_config_classes import (
    ActionConfig,
    EnvConfig,
    GenesisConfigsForRun,
    ObservationConfig,
    SceneConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    GenesisEntityCfg,
    InitialStateCfg,
)
from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common.terminations import nan_detection
from rlworld.rl.envs.mdp.terminations.k1_locomotion import gravity_z_positive

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
                        IdealPDActuatorCfg(
                            target_names_expr=(".*",),
                            stiffness=r.p_gains,
                            damping=r.d_gains,
                            armature=r.armature,
                            effort_limit=r.effort_limits,
                            frictionloss=0.1,
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
                secondary=ContactMatch(mode="body", pattern=".*", entity="terrain"),
                history_length=timing["decimation"],
            ),
            ContactSensorCfg(
                name="feet_pair_contact",
                primary=ContactMatch(mode="body", pattern=(r.foot_names[0],), entity="robot"),
                secondary=ContactMatch(mode="body", pattern=(r.foot_names[1],), entity="robot"),
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
            # Genesis's default integrator is approximate_implicitfast, a
            # faster approximation with extra effective damping (joint
            # damping folded into the mass matrix before the constraint
            # solve). Under identical PD targets it moves the limbs ~4-10%
            # less than the mjlab/newton cells and skews the contact-event
            # rewards (feet_pair_collision / swing_height / clearance).
            # implicitfast is the MuJoCo-consistent integrator and matches
            # the mjlab/newton K1 presets; measured to close most of the
            # kinematic gap (ground-contact fraction lands exactly on
            # mjlab's 0.785).
            integrator=gs.integrator.approximate_implicitfast,
            constraint_timeconst=0.02,
            enable_self_collision=True,
            box_box_detection=True,
            # Per-env dofs_info (frictionloss/armature DR writes
            # (n_envs, n_dofs) tensors; default False keeps dofs_info
            # shared across envs and rejects batched writes).
            batch_dofs_info=True,
            contact_pruning_tolerance=None,
        ),
    )


def build_action(cfg: K1JoystickConfig) -> ActionConfig:
    r = cfg.robot
    return ActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=cfg.action_scale,
        clip_actions=cfg.action_clip,
        offset=r.get_action_offset(),
    )
