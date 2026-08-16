"""Newton builders for the two-arm YAM preset.

Only the scene's second entity and the term-based action config differ
from the single-arm preset, so everything else is re-exported from it
rather than restated — the solver settings in particular, which were
tuned for this gripper against this bench.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict

from rlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonEnvConfig,
    NewtonSceneConfig,
)
from rlworld.rl.configs.presets.yam_arm import _newton_builders as single
from rlworld.rl.configs.scene.unified_entity_config import InitialStateCfg

from .base import RIGHT_ROBOT, build_action_terms

if TYPE_CHECKING:
    from .base import YamDualArmConfig

CONFIGS_FOR_RUN_CLS = single.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = single.OBSERVATION_CFG_CLS

build_visualization = single.build_visualization
build_reward = single.build_reward
build_dr_terms = single.build_dr_terms


def build_env(cfg: YamDualArmConfig, timing: Dict[str, Any]) -> NewtonEnvConfig:
    env = single.build_env(cfg, timing)
    env.task_name = "Yam_Dual"
    return env


def build_scene(cfg: YamDualArmConfig, timing: Dict[str, Any]) -> NewtonSceneConfig:
    scene = single.build_scene(cfg, timing)
    left = scene.entities["robot"]
    # Both prefixes are left unset: with more than one articulation the
    # scene assigns each entity its own from the entity name, and a name
    # shared by two entities loaded from the same MJCF would make the
    # bodies of one indistinguishable from the other's.
    scene.entities["robot"] = replace(left, body_label_prefix=None)
    # Two arms put roughly twice the contacts in front of the solver, and
    # an overflow silently DROPS them rather than erroring.
    scene.solver_cfg.nconmax = 800
    scene.entities[RIGHT_ROBOT] = replace(
        left,
        body_label_prefix=None,
        init_state=InitialStateCfg(
            pos=cfg.right_base_pos,
            joint_pos=cfg.robot.default_joint_angles,
        ),
    )
    return scene


def build_action(cfg: YamDualArmConfig) -> NewtonActionConfig:
    return NewtonActionConfig(
        actuated_dof_names=cfg.robot.actuated_dof_patterns,
        clip_actions=(-100.0, 100.0),
        action_terms=build_action_terms(cfg),
    )
