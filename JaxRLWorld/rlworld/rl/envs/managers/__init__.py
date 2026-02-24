from .base import BaseManager

# Common managers (simulator-independent)
from .common import (
    CommandManager, CommandManagerConfig,
    RewardManager, RewardManagerConfig,
    TerminationManager, TerminationConfig,
    EventManager, EventManagerConfig,
    GaitManager, GaitManagerConfig,
    ObservationManager as CommonObservationManager,
    ObsManagerConfig as CommonObsManagerConfig,
)

# Genesis managers
from .genesis import (
    ActionManager as GenesisActionManager,
    ActionManagerConfig as GenesisActionManagerConfig,
    ObservationManager as GenesisObservationManager,
    ObsManagerConfig as GenesisObsManagerConfig,
    SceneManager as GenesisSceneManager,
    SceneManagerConfig as GenesisSceneManagerConfig,
    ContactManager as GenesisContactManager,
    VisualizationManager as GenesisVisualizationManager,
    VisualizationManagerConfig as GenesisVisualizationManagerConfig,
)

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
        "MjlabSceneManager",
        "MjlabSceneManagerConfig",
        "MjlabActionManager",
        "MjlabActionManagerConfig",
        "MjlabContactManager",
        "MjlabRewardManager",
        "MjlabStateInitManager",
        "MjlabStateInitConfig",
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
    # Common
    "CommandManager", "CommandManagerConfig",
    "RewardManager", "RewardManagerConfig",
    "TerminationManager", "TerminationConfig",
    "EventManager", "EventManagerConfig",
    "GaitManager", "GaitManagerConfig",
    "CommonObservationManager", "CommonObsManagerConfig",
    # Genesis (with prefix)
    "GenesisActionManager", "GenesisActionManagerConfig",
    "GenesisObservationManager", "GenesisObsManagerConfig",
    "GenesisSceneManager", "GenesisSceneManagerConfig",
    "GenesisContactManager",
    "GenesisVisualizationManager", "GenesisVisualizationManagerConfig",
    # Newton (lazy loaded)
    "NewtonSceneManager", "NewtonSceneManagerConfig",
    "NewtonActionManager", "NewtonActionManagerConfig",
    "NewtonObservationManager", "NewtonObsManagerConfig",
    "NewtonStateInitManager", "NewtonStateInitConfig",
    "NewtonContactManager",
    # MuJoCo/mjlab (lazy loaded)
    "MjlabSceneManager", "MjlabSceneManagerConfig",
    "MjlabActionManager", "MjlabActionManagerConfig",
    "MjlabContactManager", "MjlabRewardManager",
    "MjlabStateInitManager", "MjlabStateInitConfig",
    # Backward compatibility
    "ActionManager", "ActionManagerConfig",
    "ObservationManager", "ObsManagerConfig",
    "SceneManager", "SceneManagerConfig",
    "ContactManager",
    "VisualizationManager", "VisualizationManagerConfig",
]