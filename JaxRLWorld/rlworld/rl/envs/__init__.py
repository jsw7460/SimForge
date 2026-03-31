from .world import World
from .lifecycle import LifecycleEvent, LifecycleManager
from .stats_collector import EpisodeStatsCollector
from .genesis.genesis_env import GenesisEnv
from .newton.newton_env import NewtonEnv
from .genesis.locomotion_env import GenesisLocomotionEnv
from .newton.locomotion_env import NewtonLocomotionEnv
from .gymnasium_env import GymnasiumEnv
from .mujoco.mjlab_env import MujocoEnv
from .multi_sim_world import MultiSimWorld

# Lazy import
def __getattr__(name):
    if name == "NewtonEnv":
        from .newton import NewtonEnv
        return NewtonEnv
    if name == "MujocoEnv":
        from .mujoco.mjlab_env import MujocoEnv
        return MujocoEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "World",
    "LifecycleEvent",
    "LifecycleManager",
    "EpisodeStatsCollector",
    "GenesisEnv",
    "NewtonEnv",
    "GenesisLocomotionEnv",
    "NewtonLocomotionEnv",
    "GymnasiumEnv",
    "MujocoEnv",
    "MultiSimWorld",
]