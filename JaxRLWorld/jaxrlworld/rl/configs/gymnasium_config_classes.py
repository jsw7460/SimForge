"""Minimal config dataclasses for the Gymnasium-backed env path.

Mirrors :mod:`jaxrlworld.rl.configs.genesis_config_classes` /
``newton_config_classes`` / ``mujoco_config_classes`` for the
``GymnasiumEnv`` family (dm_control suite via shimmy, classic control,
etc.).

The wrapper-chain is a Python callable and lives in
``jaxrlworld.rl.envs.gymnasium.dmc_factory`` (passed to the runner via
``runner.gym_env_factory``); these dataclasses only hold the
sim-agnostic algorithm / runner / NN bookkeeping plus the few fields
``GymnasiumEnv`` itself reads (``env_name``, ``task_name``,
``num_envs``, ``seed``, ``episode_length_s``).

Scene / Observation / Action / Reward / Command / Event / Curriculum
cfgs are empty ``BaseConfig`` placeholders by default — ``GymnasiumEnv``
infers obs / action dim straight from ``single_observation_space`` /
``single_action_space`` and otherwise ignores those fields, so we don't
synthesize sim-specific stubs we'd never use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from jaxrlworld.rl.configs.base_config import BaseConfig
from jaxrlworld.rl.configs.common_config_classes import (
    CommandConfig,
    EventConfig,
    NNConfig,
    RewardConfig,
    RunnerConfig,
    VisualizationConfig,
)


@dataclass
class GymnasiumEnvConfig(BaseConfig):
    """Env-level config for a Gymnasium-backed task.

    Mirrors the shape of :class:`GenesisEnvConfig` so the
    ``BaseRunner._create_env_from_config`` Gymnasium branch can read the
    same field names without sim-specific branching.  ``task_name`` is
    the gymnasium env id (e.g. ``"dm_control/cheetah-run-v0"``); the
    wrapper chain that gets applied around ``gym.make(task_name)`` lives
    in ``runner.gym_env_factory``.
    """

    env_name: str = "GymnasiumEnv"
    task_name: str = "Unknown"
    num_envs: int = 1
    seed: int = 0
    # Honoured by ``BaseRunner._create_eval_configs`` and any other
    # episode-length-aware bookkeeping.  The actual episode budget for
    # the underlying ``gym.make`` is configured via the wrapper chain
    # (``max_episode_steps`` argument of ``make_dmc_env_factory``).
    episode_length_s: float = 1000.0
    # Optional ``gym.make`` kwargs passed through by callers that don't
    # use ``runner.gym_env_factory`` (legacy / wrapper-less path).
    gym_make_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GymnasiumConfigsForRun(BaseConfig):
    """Complete configuration for Gymnasium-backed training runs.

    Field set intentionally mirrors :class:`GenesisConfigsForRun` so
    every common runner / algorithm code path (``BaseRunner`` /
    ``ModelBasedRunner`` / ``BaseAlgorithm``) keeps working without
    sim-specific branching.  The sim-side fields (``scene``,
    ``observation``, ``action``, ``reward``, ``command``, ``event``,
    ``curriculum``) are empty ``BaseConfig`` placeholders — Gymnasium
    envs don't consume them.
    """

    sim_type: str = "gymnasium"
    preset_module: str | None = None
    preset_class_name: str | None = None
    preset_kwargs: Dict[str, Any] | None = None
    env: GymnasiumEnvConfig = field(default_factory=GymnasiumEnvConfig)
    # Placeholder cfgs — ``GymnasiumEnv`` ignores these.  Kept so
    # ``cfgs.<field>`` access in shared runner code (extras, eval-cfg
    # synthesis, ...) doesn't ``AttributeError``.
    scene: BaseConfig = field(default_factory=BaseConfig)
    observation: BaseConfig = field(default_factory=BaseConfig)
    action: BaseConfig = field(default_factory=BaseConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    event: EventConfig = field(default_factory=EventConfig)
    # ``CurriculumManagerConfig`` lives in genesis_config_classes' default
    # factory and pulls in sim-side curriculum machinery.  Gymnasium
    # tasks don't use curriculum, so a bare ``BaseConfig`` is enough.
    curriculum: BaseConfig = field(default_factory=BaseConfig)
    algorithm: BaseConfig = field(default_factory=BaseConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
