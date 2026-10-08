"""Genesis builders for the K1 getup task.

See ``_mujoco_builders`` for what this module owns; the scene, actuators and
action path are the velocity preset's. The rigid solver options are restated
in full because they are one object: the T1 getup recipe's budget (30 / 40
iterations, 300 collision pairs), its elliptic cone at ``impratio`` 10, and
``convex`` contact resolution, which that recipe pins because ``signorini``
diverged to NaN on the landing impacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict

import genesis as gs

from jaxrlworld.rl.configs import TerminationTermConfig
from jaxrlworld.rl.configs.common_config_classes import TerminationsConfig
from jaxrlworld.rl.configs.genesis_config_classes import ActionConfig, EnvConfig, SceneConfig
from jaxrlworld.rl.configs.presets.k1_velocity import _genesis_builders as velocity
from jaxrlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from jaxrlworld.rl.envs.mdp.terminations.common.terminations import nan_detection

if TYPE_CHECKING:
    from .base import K1GetupConfig

CONFIGS_FOR_RUN_CLS = velocity.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = velocity.OBSERVATION_CFG_CLS
build_visualization = velocity.build_visualization


def build_env(cfg: K1GetupConfig, timing: Dict[str, Any]) -> EnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        nan = TerminationTermConfig(nan_detection)
        max_episode = TerminationTermConfig(max_episode_exceed)

    return EnvConfig(
        num_envs=cfg.num_envs,
        env_name="GenesisEnv",
        task_name="K1_Getup",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: K1GetupConfig, timing: Dict[str, Any]) -> SceneConfig:
    scene = velocity.build_scene(cfg, timing)
    sim_dt = timing["dt"]
    return replace(
        scene,
        rigid_options=gs.options.RigidOptions(
            dt=sim_dt,
            # As in the velocity preset: no extra damping folded into the
            # mass matrix.
            integrator=gs.integrator.implicitfast,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=30,
            ls_iterations=40,
            tolerance=1e-5,
            constraint_timeconst=0.02,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=300,
            batch_dofs_info=True,
            batch_links_info=True,
            contact_pruning_tolerance=None,
            friction_cone=gs.friction_cone.elliptic,
            contact_resolution=gs.contact_resolution.convex,
            impratio=10.0,
        ),
    )


def build_action(cfg: K1GetupConfig) -> ActionConfig:
    return replace(velocity.build_action(cfg), settle_steps=cfg.settle_steps)
