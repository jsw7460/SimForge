"""Newton builders for the K1 getup task.

See ``_mujoco_builders`` for what this module owns; the scene, actuators and
action path are the velocity preset's. On this backend the contact budget has
to be stated: mjwarp drops rows past ``nconmax`` silently, and a robot lying
on its shells carries several times the rows of one standing on its feet, so
the velocity preset's walking budget (220 / 500) is raised here. Re-measure
with the contact-demand diagnostic after any Newton or mujoco-warp bump.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict

from jaxrlworld.rl.configs import TerminationTermConfig
from jaxrlworld.rl.configs.common_config_classes import TerminationsConfig
from jaxrlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonEnvConfig,
    NewtonSceneConfig,
    SolverMuJoCoCfg,
)
from jaxrlworld.rl.configs.presets.k1_velocity import _newton_builders as velocity
from jaxrlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from jaxrlworld.rl.envs.mdp.terminations.common.terminations import nan_detection

if TYPE_CHECKING:
    from .base import K1GetupConfig

CONFIGS_FOR_RUN_CLS = velocity.CONFIGS_FOR_RUN_CLS
OBSERVATION_CFG_CLS = velocity.OBSERVATION_CFG_CLS
build_visualization = velocity.build_visualization


def build_env(cfg: K1GetupConfig, timing: Dict[str, Any]) -> NewtonEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        # As in the velocity preset: an environment that goes non-finite
        # poisons its rollout rather than failing.
        nan = TerminationTermConfig(nan_detection)
        max_episode = TerminationTermConfig(max_episode_exceed)

    return NewtonEnvConfig(
        num_envs=cfg.num_envs,
        env_name="NewtonEnv",
        task_name="K1_Getup",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: K1GetupConfig, timing: Dict[str, Any]) -> NewtonSceneConfig:
    scene = velocity.build_scene(cfg, timing)
    return replace(
        scene,
        solver_cfg=SolverMuJoCoCfg(
            cone="elliptic",
            impratio=10.0,
            iterations=20,
            ls_iterations=50,
            ccd_iterations=50,
            nconmax=400,
            njmax=800,
            use_mujoco_contacts=True,
        ),
    )


def build_action(cfg: K1GetupConfig) -> NewtonActionConfig:
    return replace(velocity.build_action(cfg), settle_steps=cfg.settle_steps)
