"""MuJoCo (mjlab) builders for the K1 getup task.

The scene, the actuators and the action path are the velocity preset's, taken
from its builders rather than restated, so the policy contract cannot drift
between the two tasks. What this module owns is what a fallen robot changes:

- **Terminations.** Time-out only. The robot starts fallen, so neither the tilt
  termination nor the non-foot ground contact one can apply.
- **Contact solver.** The getup recipe's elliptic cone at ``impratio`` 10, which
  resists the sliding a body pushing itself up does; the velocity recipe's
  pyramidal cone at 1 is tuned for walking. The contact budget is left to
  mjwarp's automatic sizing: a robot lying on its shells carries several times
  the contact rows of one standing on its feet, and the velocity preset's
  fixed ``nconmax`` is sized for the latter. The constraint-row budget is the
  Newton builder's 800 for the same reason; the velocity preset's 300 is a
  standing robot's, and an overflow silently DROPS rows.
- **Settle hold.** The action manager's settle mask holds the current joint
  position for the first ``settle_steps`` after each reset.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict

from jaxrlworld.rl.configs import TerminationTermConfig
from jaxrlworld.rl.configs.common_config_classes import TerminationsConfig
from jaxrlworld.rl.configs.mujoco_config_classes import (
    MujocoActionConfig,
    MujocoEnvConfig,
    MujocoSceneConfig,
)
from jaxrlworld.rl.configs.presets.k1_velocity import _mujoco_builders as velocity
from jaxrlworld.rl.envs.mdp.terminations.mujoco import terminations as tf

if TYPE_CHECKING:
    from .base import K1GetupConfig

CONFIGS_FOR_RUN_CLS = velocity.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = velocity.OBSERVATION_CFG_CLS
build_visualization = velocity.build_visualization


def build_env(cfg: K1GetupConfig, timing: Dict[str, Any]) -> MujocoEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        time_out = TerminationTermConfig(tf.time_out)

    return MujocoEnvConfig(
        num_envs=cfg.num_envs,
        env_name="MujocoEnv",
        task_name="K1_Getup",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: K1GetupConfig, timing: Dict[str, Any]) -> MujocoSceneConfig:
    scene = velocity.build_scene(cfg, timing)
    return replace(
        scene,
        cone="elliptic",
        impratio=10.0,
        nconmax=None,
        njmax=800,
        preset_class_name=type(cfg).__name__,
        preset_module_path=type(cfg).__module__,
    )


def build_action(cfg: K1GetupConfig) -> MujocoActionConfig:
    return replace(velocity.build_action(cfg), settle_steps=cfg.settle_steps)
