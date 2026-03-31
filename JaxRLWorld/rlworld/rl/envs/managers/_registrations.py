"""Central manager registration for all simulator backends.

This module populates the ManagerRegistry so that environment code can
create managers via ``ManagerRegistry.create(sim_type, role, ...)``.

Lazy-imported backends (Newton, MuJoCo) are registered through deferred
callables so that heavy dependencies (warp, mjlab) are not imported until
actually needed.
"""

from __future__ import annotations

from rlworld.rl.envs.managers.registry import ManagerRegistry


def _register_common() -> None:
    """Register simulator-agnostic managers shared by all backends."""
    from rlworld.rl.envs.managers.common import (
        CommandManager, CommandManagerConfig,
        RewardManager, RewardManagerConfig,
        TerminationManager, TerminationConfig,
        EventManager, EventManagerConfig,
    )

    for sim_type in ("genesis", "newton", "mujoco"):
        ManagerRegistry.register(sim_type, "command", CommandManager, CommandManagerConfig)
        ManagerRegistry.register(sim_type, "reward", RewardManager, RewardManagerConfig)
        ManagerRegistry.register(sim_type, "termination", TerminationManager, TerminationConfig)
        ManagerRegistry.register(sim_type, "event", EventManager, EventManagerConfig)


def _register_genesis() -> None:
    """Register Genesis-specific managers."""
    from rlworld.rl.envs.managers.genesis import (
        ActionManager, ActionManagerConfig,
        ObservationManager, ObsManagerConfig,
        SceneManager, SceneManagerConfig,
        ContactManager,
        VisualizationManager, VisualizationManagerConfig,
    )

    ManagerRegistry.register("genesis", "scene", SceneManager, SceneManagerConfig)
    ManagerRegistry.register("genesis", "action", ActionManager, ActionManagerConfig)
    ManagerRegistry.register("genesis", "observation", ObservationManager, ObsManagerConfig)
    ManagerRegistry.register("genesis", "contact", ContactManager)
    ManagerRegistry.register("genesis", "visualization", VisualizationManager, VisualizationManagerConfig)


def _register_newton() -> None:
    """Register Newton-specific managers (lazy — imports warp)."""
    from rlworld.rl.envs.managers.newton import (
        NewtonSceneManager, NewtonSceneManagerConfig,
        NewtonActionManager, NewtonActionManagerConfig,
        NewtonObservationManager, NewtonObsManagerConfig,
        NewtonContactManager,
        NewtonVisualizationManager, NewtonVisualizationManagerConfig,
    )

    ManagerRegistry.register("newton", "scene", NewtonSceneManager, NewtonSceneManagerConfig)
    ManagerRegistry.register("newton", "action", NewtonActionManager, NewtonActionManagerConfig)
    ManagerRegistry.register("newton", "observation", NewtonObservationManager, NewtonObsManagerConfig)
    ManagerRegistry.register("newton", "contact", NewtonContactManager)
    ManagerRegistry.register("newton", "visualization", NewtonVisualizationManager, NewtonVisualizationManagerConfig)


def _register_mujoco() -> None:
    """Register MuJoCo/mjlab-specific managers (lazy — imports mjlab)."""
    from rlworld.rl.envs.managers.mujoco import (
        MujocoSceneManager, MujocoSceneManagerConfig,
        MujocoActionManager, MujocoActionManagerConfig,
        MujocoContactManager,
        MujocoRewardManager,
    )
    from rlworld.rl.envs.managers.common import (
        ObservationManager, ObsManagerConfig,
        RewardManagerConfig,
    )

    ManagerRegistry.register("mujoco", "scene", MujocoSceneManager, MujocoSceneManagerConfig)
    ManagerRegistry.register("mujoco", "action", MujocoActionManager, MujocoActionManagerConfig)
    ManagerRegistry.register("mujoco", "observation", ObservationManager, ObsManagerConfig)
    ManagerRegistry.register("mujoco", "contact", MujocoContactManager)
    # MuJoCo overrides the common reward manager with MujocoRewardManager
    ManagerRegistry.register("mujoco", "reward", MujocoRewardManager, RewardManagerConfig)


def register_all_for(sim_type: str) -> None:
    """Register all managers for a given sim_type.

    Called lazily when a backend is first used, avoiding heavy imports
    until they are actually needed.
    """
    _register_common()

    registrars = {
        "genesis": _register_genesis,
        "newton": _register_newton,
        "mujoco": _register_mujoco,
    }

    registrar = registrars.get(sim_type)
    if registrar is None:
        raise ValueError(
            f"Unknown sim_type={sim_type!r}. Available: {list(registrars.keys())}"
        )
    registrar()
