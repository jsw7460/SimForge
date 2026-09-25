"""Config exports.

Almost everything here is simulator-agnostic (plain dataclasses) and
loads eagerly.  The only exception is ``genesis_config_classes`` — it
does ``import genesis`` at module load — so the names it exports
(``ActionConfig`` / ``EnvConfig`` / ``GenesisConfigsForRun`` /
``ObservationConfig`` / ``SceneConfig``, all Genesis-flavoured) are
exposed lazily via ``__getattr__``.  That keeps ``from jaxrlworld.rl.configs
import CommandConfig`` (done by basically every preset / runner) from
dragging Genesis into a Newton- or MuJoCo-only process.  The Newton and
MuJoCo config-class modules are pure dataclasses, so they stay eager.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .commands.command_term_config import CommandTermConfig
from .common_config_classes import (
    CommandConfig,
    EventConfig,
    FastTD3PolicyConfig,
    GaitConfig,
    NNConfig,
    PolicyConfig,
    PPOPolicyConfig,
    RewardConfig,
    RunnerConfig,
    SACPolicyConfig,
    TD3PolicyConfig,
    VisualizationConfig,
)
from .curriculums.curriculum_term_config import (
    CurriculumManagerConfig,
    CurriculumTermConfig,
)
from .mujoco_config_classes import (
    MujocoActionConfig,
    MujocoConfigsForRun,
    MujocoEnvConfig,
    MujocoObservationConfig,
    MujocoSceneConfig,
)
from .newton_config_classes import (
    NewtonActionConfig,
    NewtonConfigsForRun,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    SolverMuJoCoCfg,
)

# Term-level configs hoisted from the old jaxrlworld.rl.envs.mdp.configs
# location so callers can do `from jaxrlworld.rl.configs import ...` directly.
from .terminations.termination_term_config import TerminationResult, TerminationTermConfig

# ── Lazy, Genesis-importing exports ──────────────────────────────────
# (genesis_config_classes does ``import genesis as gs`` at module load)
_LAZY: dict[str, tuple[str, str]] = {
    "ActionConfig": (".genesis_config_classes", "ActionConfig"),
    "EnvConfig": (".genesis_config_classes", "EnvConfig"),
    "GenesisConfigsForRun": (".genesis_config_classes", "GenesisConfigsForRun"),
    "ObservationConfig": (".genesis_config_classes", "ObservationConfig"),
    "SceneConfig": (".genesis_config_classes", "SceneConfig"),
    # Gymnasium-backed envs (dm_control via shimmy, classic control, ...).
    # Pure-Python module — no simulator import — so it's safe to expose
    # alongside the Genesis-importing names above.
    "GymnasiumConfigsForRun": (".gymnasium_config_classes", "GymnasiumConfigsForRun"),
    "GymnasiumEnvConfig": (".gymnasium_config_classes", "GymnasiumEnvConfig"),
}

# sim_type → (submodule, ConfigsForRun class name) for lazy resolution.
_CONFIGS_FOR_RUN_LOCATIONS: dict[str, tuple[str, str]] = {
    "genesis": (".genesis_config_classes", "GenesisConfigsForRun"),
    "newton": (".newton_config_classes", "NewtonConfigsForRun"),
    "mujoco": (".mujoco_config_classes", "MujocoConfigsForRun"),
    "gymnasium": (".gymnasium_config_classes", "GymnasiumConfigsForRun"),
}


def __getattr__(name: str):
    if name in _LAZY:
        submod, attr = _LAZY[name]
        return getattr(importlib.import_module(submod, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # let type checkers / IDEs see the lazy names + the union
    from .genesis_config_classes import (
        ActionConfig,
        EnvConfig,
        GenesisConfigsForRun,
        ObservationConfig,
        SceneConfig,
    )

    ConfigsForRun = GenesisConfigsForRun | NewtonConfigsForRun | MujocoConfigsForRun
else:
    # Runtime placeholder: ``ConfigsForRun`` is only ever used as an
    # annotation (no isinstance / construction), so a plain object keeps
    # ``from jaxrlworld.rl.configs import ConfigsForRun`` cheap.
    ConfigsForRun = object


__all__ = [
    "CommandTermConfig",
    "CommandConfig",
    "EventConfig",
    "FastTD3PolicyConfig",
    "GaitConfig",
    "NNConfig",
    "PolicyConfig",
    "PPOPolicyConfig",
    "RewardConfig",
    "RunnerConfig",
    "SACPolicyConfig",
    "TD3PolicyConfig",
    "VisualizationConfig",
    "CurriculumManagerConfig",
    "CurriculumTermConfig",
    "MujocoActionConfig",
    "MujocoConfigsForRun",
    "MujocoEnvConfig",
    "MujocoObservationConfig",
    "MujocoSceneConfig",
    "NewtonActionConfig",
    "NewtonConfigsForRun",
    "NewtonEnvConfig",
    "NewtonObservationConfig",
    "NewtonSceneConfig",
    "SolverMuJoCoCfg",
    "TerminationResult",
    "TerminationTermConfig",
    "ConfigsForRun",
    # lazy (Genesis-flavoured)
    "ActionConfig",
    "EnvConfig",
    "GenesisConfigsForRun",
    "ObservationConfig",
    "SceneConfig",
]
