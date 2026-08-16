"""Genesis builders for the two-arm YAM preset.

See ``_newton_builders`` — the same delta, expressed against the
Genesis single-arm builders.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict

from rlworld.rl.configs.genesis_config_classes import (
    ActionConfig,
    EnvConfig,
    SceneConfig,
)
from rlworld.rl.configs.presets.yam_arm import _genesis_builders as single
from rlworld.rl.configs.scene.unified_entity_config import InitialStateCfg

from .base import RIGHT_ROBOT, build_action_terms

if TYPE_CHECKING:
    from .base import YamDualArmConfig

CONFIGS_FOR_RUN_CLS = single.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = single.OBSERVATION_CFG_CLS

build_visualization = single.build_visualization
build_reward = single.build_reward
build_dr_terms = single.build_dr_terms


def build_env(cfg: YamDualArmConfig, timing: Dict[str, Any]) -> EnvConfig:
    env = single.build_env(cfg, timing)
    env.task_name = "Yam_Dual"
    return env


def build_scene(cfg: YamDualArmConfig, timing: Dict[str, Any]) -> SceneConfig:
    scene = single.build_scene(cfg, timing)
    scene.rigid_options.max_collision_pairs = 200
    scene.entities[RIGHT_ROBOT] = replace(
        scene.entities["robot"],
        init_state=InitialStateCfg(
            pos=cfg.right_base_pos,
            joint_pos=cfg.robot.default_joint_angles,
        ),
    )
    return scene


def build_action(cfg: YamDualArmConfig) -> ActionConfig:
    return ActionConfig(
        actuated_dof_names=cfg.robot.actuated_dof_patterns,
        clip_actions=(-100.0, 100.0),
        action_terms=build_action_terms(cfg),
    )
