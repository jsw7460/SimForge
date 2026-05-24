"""Scene-config exports.

The sim-agnostic pieces (``SceneEntitySelector`` / ``ResolvedEntity``,
``TerrainCfg``, and the unified ``*EntityCfg`` dataclasses) are imported
eagerly — they pull in no simulator package. The sim-specific scene-init
configs (``EntityConfig`` / ``GenesisSceneInitConfig`` import
``genesis``; ``NewtonEntityConfig`` & friends import ``newton`` +
``warp``) are exposed lazily via ``__getattr__`` so importing a unified
config does not drag a simulator into a process that only needs another
one.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .entity_selector import ResolvedEntity, SceneEntitySelector
from .terrain_config import TerrainCfg
from .unified_entity_config import (
    ArticulationCfg,
    EntityCfg,
    GenesisEntityCfg,
    InitialStateCfg,
    MujocoEntityCfg,
    NewtonEntityCfg,
)

# name → (submodule, attr) for lazily-loaded, simulator-dependent exports.
_LAZY: dict[str, tuple[str, str]] = {
    "EntityConfig": (".entity_config", "EntityConfig"),
    "GenesisSceneInitConfig": (".entity_config", "GenesisSceneInitConfig"),
    "NewtonBoxConfig": (".newton_entity_config", "NewtonBoxConfig"),
    "NewtonEntityConfig": (".newton_entity_config", "NewtonEntityConfig"),
}


def __getattr__(name: str):
    if name in _LAZY:
        submod, attr = _LAZY[name]
        return getattr(importlib.import_module(submod, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # let type checkers / IDEs see the lazy names
    from .entity_config import EntityConfig, GenesisSceneInitConfig
    from .newton_entity_config import NewtonBoxConfig, NewtonEntityConfig


__all__ = [
    "ResolvedEntity",
    "SceneEntitySelector",
    "TerrainCfg",
    "ArticulationCfg",
    "EntityCfg",
    "GenesisEntityCfg",
    "InitialStateCfg",
    "MujocoEntityCfg",
    "NewtonEntityCfg",
    "EntityConfig",
    "GenesisSceneInitConfig",
    "NewtonBoxConfig",
    "NewtonEntityConfig",
]
