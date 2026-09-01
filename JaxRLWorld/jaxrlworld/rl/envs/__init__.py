"""Environment exports.

Sim-agnostic pieces (``World``, lifecycle, stats collector, ``MultiSimWorld``)
load eagerly.  The concrete per-sim ``World`` subclasses each ``import`` a
heavyweight simulator package at module load (Genesis → ``genesis``,
Newton → ``warp`` + ``newton``, MuJoCo → ``mjlab``; ``GymnasiumEnv`` and
``ManiSkillEnv`` likewise pull their backends), so they are exposed
lazily via ``__getattr__`` — a process that runs one simulator no longer
pays the import cost of the others.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .lifecycle import LifecycleEvent, LifecycleManager
from .multi_sim_world import MultiSimWorld
from .stats_collector import EpisodeStatsCollector
from .world import World

# name → (submodule, attr) for lazily-loaded, simulator-dependent envs.
_LAZY: dict[str, tuple[str, str]] = {
    "GenesisEnv": (".genesis.genesis_env", "GenesisEnv"),
    "GenesisLocomotionEnv": (".genesis.locomotion_env", "GenesisLocomotionEnv"),
    "NewtonEnv": (".newton.newton_env", "NewtonEnv"),
    "NewtonLocomotionEnv": (".newton.locomotion_env", "NewtonLocomotionEnv"),
    "MujocoEnv": (".mujoco.mjlab_env", "MujocoEnv"),
    "MujocoLocomotionEnv": (".mujoco.locomotion_env", "MujocoLocomotionEnv"),
    "GymnasiumEnv": (".gymnasium_env", "GymnasiumEnv"),
    "ManiSkillEnv": (".maniskill_env", "ManiSkillEnv"),
}


def __getattr__(name: str):
    if name in _LAZY:
        submod, attr = _LAZY[name]
        return getattr(importlib.import_module(submod, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # let type checkers / IDEs see the lazy names
    from .genesis.genesis_env import GenesisEnv
    from .genesis.locomotion_env import GenesisLocomotionEnv
    from .gymnasium_env import GymnasiumEnv
    from .maniskill_env import ManiSkillEnv
    from .mujoco.locomotion_env import MujocoLocomotionEnv
    from .mujoco.mjlab_env import MujocoEnv
    from .newton.locomotion_env import NewtonLocomotionEnv
    from .newton.newton_env import NewtonEnv

__all__ = [
    "World",
    "LifecycleEvent",
    "LifecycleManager",
    "EpisodeStatsCollector",
    "MultiSimWorld",
    "GenesisEnv",
    "GenesisLocomotionEnv",
    "NewtonEnv",
    "NewtonLocomotionEnv",
    "MujocoEnv",
    "MujocoLocomotionEnv",
    "GymnasiumEnv",
    "ManiSkillEnv",
]
