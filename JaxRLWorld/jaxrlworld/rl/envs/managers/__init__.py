from .base import BaseManager

# Common managers (simulator-independent)
from .common import (
    CommandManager,
    CommandManagerConfig,
    EventManager,
    EventManagerConfig,
    GaitManager,
    GaitManagerConfig,
    ObservationManager as CommonObservationManager,
    ObsManagerConfig as CommonObsManagerConfig,
    RewardManager,
    RewardManagerConfig,
    TerminationConfig,
    TerminationManager,
    gait_config_to_manager_config,
)
from .common.contact import BaseContactManager
from .registry import ManagerRegistry

# Per-sim managers each ``import`` a heavyweight simulator package at module
# load (Genesis → ``genesis``, Newton → ``warp`` + ``newton``,
# MuJoCo → ``mjlab``), so they are exposed lazily via ``__getattr__`` — a
# process that runs one simulator no longer pays the import cost of the others.
# Genesis is also the historical default, so the un-prefixed aliases
# (``ActionManager`` etc.) resolve to the Genesis managers.
_GENESIS_NAMES: dict[str, str] = {
    "GenesisActionManager": "ActionManager",
    "GenesisActionManagerConfig": "ActionManagerConfig",
    "GenesisContactManager": "ContactManager",
    "GenesisObservationManager": "ObservationManager",
    "GenesisObsManagerConfig": "ObsManagerConfig",
    "GenesisSceneManager": "SceneManager",
    "GenesisSceneManagerConfig": "SceneManagerConfig",
    "GenesisVisualizationManager": "VisualizationManager",
    "GenesisVisualizationManagerConfig": "VisualizationManagerConfig",
    # Backward-compatibility aliases (Genesis as default)
    "ActionManager": "ActionManager",
    "ActionManagerConfig": "ActionManagerConfig",
    "ContactManager": "ContactManager",
    "ObservationManager": "ObservationManager",
    "ObsManagerConfig": "ObsManagerConfig",
    "SceneManager": "SceneManager",
    "SceneManagerConfig": "SceneManagerConfig",
    "VisualizationManager": "VisualizationManager",
    "VisualizationManagerConfig": "VisualizationManagerConfig",
}
_NEWTON_NAMES = {
    "NewtonSceneManager",
    "NewtonSceneManagerConfig",
    "NewtonActionManager",
    "NewtonActionManagerConfig",
    "NewtonObservationManager",
    "NewtonObsManagerConfig",
    "NewtonContactManager",
}
_MUJOCO_NAMES = {
    "MujocoSceneManager",
    "MujocoSceneManagerConfig",
    "MujocoActionManager",
    "MujocoActionManagerConfig",
    "MujocoContactManager",
    "MujocoRewardManager",
}


def __getattr__(name):
    if name in _GENESIS_NAMES:
        from . import genesis as genesis_module

        return getattr(genesis_module, _GENESIS_NAMES[name])
    if name in _NEWTON_NAMES:
        from . import newton as newton_module

        return getattr(newton_module, name)
    if name in _MUJOCO_NAMES:
        from . import mujoco as mujoco_module

        return getattr(mujoco_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseManager",
    "BaseContactManager",
    "ManagerRegistry",
    # Common
    "CommandManager",
    "CommandManagerConfig",
    "RewardManager",
    "RewardManagerConfig",
    "TerminationManager",
    "TerminationConfig",
    "EventManager",
    "EventManagerConfig",
    "GaitManager",
    "GaitManagerConfig",
    "gait_config_to_manager_config",
    "CommonObservationManager",
    "CommonObsManagerConfig",
    # Genesis (with prefix)
    "GenesisActionManager",
    "GenesisActionManagerConfig",
    "GenesisObservationManager",
    "GenesisObsManagerConfig",
    "GenesisSceneManager",
    "GenesisSceneManagerConfig",
    "GenesisContactManager",
    "GenesisVisualizationManager",
    "GenesisVisualizationManagerConfig",
    # Newton (lazy loaded)
    "NewtonSceneManager",
    "NewtonSceneManagerConfig",
    "NewtonActionManager",
    "NewtonActionManagerConfig",
    "NewtonObservationManager",
    "NewtonObsManagerConfig",
    "NewtonContactManager",
    # MuJoCo/mjlab (lazy loaded)
    "MujocoSceneManager",
    "MujocoSceneManagerConfig",
    "MujocoActionManager",
    "MujocoActionManagerConfig",
    "MujocoContactManager",
    "MujocoRewardManager",
    # Backward compatibility
    "ActionManager",
    "ActionManagerConfig",
    "ObservationManager",
    "ObsManagerConfig",
    "SceneManager",
    "SceneManagerConfig",
    "ContactManager",
    "VisualizationManager",
    "VisualizationManagerConfig",
]
