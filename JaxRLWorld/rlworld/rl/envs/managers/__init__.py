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
)
from .common.contact import BaseContactManager

# Genesis managers
from .genesis import (
    ActionManager as GenesisActionManager,
    ActionManagerConfig as GenesisActionManagerConfig,
    ContactManager as GenesisContactManager,
    ObservationManager as GenesisObservationManager,
    ObsManagerConfig as GenesisObsManagerConfig,
    SceneManager as GenesisSceneManager,
    SceneManagerConfig as GenesisSceneManagerConfig,
    VisualizationManager as GenesisVisualizationManager,
    VisualizationManagerConfig as GenesisVisualizationManagerConfig,
)
from .registry import ManagerRegistry


# Newton managers - lazy import to avoid warp initialization
def __getattr__(name):
    newton_names = {
        "NewtonSceneManager",
        "NewtonSceneManagerConfig",
        "NewtonActionManager",
        "NewtonActionManagerConfig",
        "NewtonObservationManager",
        "NewtonObsManagerConfig",
        "NewtonContactManager",
    }
    mujoco_names = {
        "MujocoSceneManager",
        "MujocoSceneManagerConfig",
        "MujocoActionManager",
        "MujocoActionManagerConfig",
        "MujocoContactManager",
        "MujocoRewardManager",
    }
    if name in newton_names:
        from . import newton as newton_module

        return getattr(newton_module, name)
    if name in mujoco_names:
        from . import mujoco as mujoco_module

        return getattr(mujoco_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Backward compatibility aliases (Genesis as default)
ActionManager = GenesisActionManager
ActionManagerConfig = GenesisActionManagerConfig
ObservationManager = GenesisObservationManager
ObsManagerConfig = GenesisObsManagerConfig
SceneManager = GenesisSceneManager
SceneManagerConfig = GenesisSceneManagerConfig
ContactManager = GenesisContactManager
VisualizationManager = GenesisVisualizationManager
VisualizationManagerConfig = GenesisVisualizationManagerConfig

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
