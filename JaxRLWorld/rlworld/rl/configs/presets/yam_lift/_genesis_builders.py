"""genesis builders for the lift task.

The scene is the single-arm preset's — same table, same cube, same
solver settings — plus one contact sensor the task's termination needs.
Everything else about the task lives in the config, so the three builder
modules differ only in which module they extend and in where their
backend keeps sensors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from rlworld.rl.configs import RewardConfig
from rlworld.rl.configs.genesis_config_classes import EnvConfig
from rlworld.rl.configs.presets.yam_arm import _genesis_builders as arm
from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg

from .base import ARM_TABLE_CONTACT

if TYPE_CHECKING:
    from .base import YamLiftConfig

CONFIGS_FOR_RUN_CLS = arm.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = arm.OBSERVATION_CFG_CLS

build_visualization = arm.build_visualization
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


def build_scene(cfg: YamLiftConfig, timing: Dict[str, Any]):
    """The arm preset's scene, plus a sensor watching the work surface.

    The termination that uses it needs to know when the arm is driving
    into the table rather than working above it, and only a contact
    sensor can say so. Named per backend because mjlab keeps sensors in a
    different field from the other two.
    """
    scene = arm.build_scene(cfg, timing)
    sensor = ContactSensorCfg(
        name=ARM_TABLE_CONTACT,
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        # The whole prop, however its backend spells its bodies.
        secondary=ContactMatch(mode="entity", entity="table"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=timing["decimation"],
    )
    field_name = "sensors" if False else "contact_sensors"
    existing = tuple(getattr(scene, field_name) or ())
    setattr(scene, field_name, [*existing, sensor])
    return scene
