"""genesis builders for the lift task.

The scene is the single-arm preset's, unchanged — same table, same cube,
same solver settings. Only the task differs, and the task lives in the
config rather than here, so the three builder modules stay identical
apart from which module they extend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from rlworld.rl.configs import RewardConfig
from rlworld.rl.configs.genesis_config_classes import EnvConfig
from rlworld.rl.configs.presets.yam_arm import _genesis_builders as arm

if TYPE_CHECKING:
    from .base import YamLiftConfig

CONFIGS_FOR_RUN_CLS = arm.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = arm.OBSERVATION_CFG_CLS

build_visualization = arm.build_visualization
build_scene = arm.build_scene
build_action = arm.build_action
build_dr_terms = arm.build_dr_terms


def build_env(cfg: YamLiftConfig, timing: Dict[str, Any]) -> EnvConfig:
    env = arm.build_env(cfg, timing)
    env.task_name = "Yam_Lift"
    # The arm preset ends an episode only on time-out. This one also ends
    # it when the cube leaves the table, which the arm cannot undo.
    env.terminations = cfg.build_terminations()
    return env


def build_reward(cfg: YamLiftConfig) -> RewardConfig:
    return cfg.build_rewards()
