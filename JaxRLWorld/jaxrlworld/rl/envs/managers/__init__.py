from .base import BaseManager

# Common managers (simulator-independent)
from .common import (
    CommandManager,
    CommandManagerConfig,
    EventManager,
    GaitManager,
    GaitManagerConfig,
    ObservationManager,
    RewardManager,
    TerminationManager,
    gait_config_to_manager_config,
)
from .common.contact import BaseContactManager
from .registry import ManagerRegistry

# Per-sim managers each ``import`` a heavyweight simulator package at module
# load (Genesis → ``genesis``, Newton → ``warp`` + ``newton``,
# MuJoCo → ``mjlab``), so they are exposed lazily via ``__getattr__`` — a
# process that runs one simulator does not pay the import cost of the others.
# The Genesis package names its managers without a prefix, so they are
# exported here under a ``Genesis`` prefix.
_GENESIS_NAMES: dict[str, str] = {
    "GenesisActionManager": "ActionManager",
    "GenesisActionManagerConfig": "ActionManagerConfig",
    "GenesisContactManager": "ContactManager",
    "GenesisSceneManager": "SceneManager",
    "GenesisSceneManagerConfig": "SceneManagerConfig",
    "GenesisVisualizationManager": "VisualizationManager",
    "GenesisVisualizationManagerConfig": "VisualizationManagerConfig",
}
_NEWTON_NAMES = {
    "NewtonSceneManager",
    "NewtonSceneManagerConfig",
    "NewtonActionManager",
    "NewtonActionManagerConfig",
    "NewtonContactManager",
}
_MUJOCO_NAMES = {
    "MujocoSceneManager",
    "MujocoSceneManagerConfig",
    "MujocoActionManager",
    "MujocoActionManagerConfig",
    "MujocoContactManager",
}


def __getattr__(name):
    if name in _GENESIS_NAMES:
        from . import genesis as genesis_module  # local-import: backend (genesis), lazy package export

        return getattr(genesis_module, _GENESIS_NAMES[name])  # dynamic-attr: lazy module __getattr__ export
    if name in _NEWTON_NAMES:
        from . import newton as newton_module  # local-import: backend (newton), lazy package export

        return getattr(newton_module, name)  # dynamic-attr: lazy module __getattr__ export
    if name in _MUJOCO_NAMES:
        from . import mujoco as mujoco_module  # local-import: backend (mjlab), lazy package export

        return getattr(mujoco_module, name)  # dynamic-attr: lazy module __getattr__ export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseManager",
    "BaseContactManager",
    "ManagerRegistry",
    # Common
    "CommandManager",
    "CommandManagerConfig",
    "RewardManager",
    "TerminationManager",
    "EventManager",
    "GaitManager",
    "GaitManagerConfig",
    "gait_config_to_manager_config",
    "ObservationManager",
    # Genesis (with prefix)
    "GenesisActionManager",
    "GenesisActionManagerConfig",
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
    "NewtonContactManager",
    # MuJoCo/mjlab (lazy loaded)
    "MujocoSceneManager",
    "MujocoSceneManagerConfig",
    "MujocoActionManager",
    "MujocoActionManagerConfig",
    "MujocoContactManager",
]
