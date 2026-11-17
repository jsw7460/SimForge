"""Central manager registration for all simulator backends.

This module populates the ManagerRegistry so that environment code can
create managers via ``ManagerRegistry.create(sim_type, role, ...)``.

Lazy-imported backends (Newton, MuJoCo) are registered through deferred
callables so that heavy dependencies (warp, mjlab) are not imported until
actually needed. A backend from another package registers its managers
through :class:`~jaxrlworld.rl.envs.backends.BackendSpec`.
"""

from __future__ import annotations

from jaxrlworld.rl.configs.common_config_classes import EventConfig, RewardConfig
from jaxrlworld.rl.envs.backends import BUILTIN_SIM_TYPES, get_backend
from jaxrlworld.rl.envs.managers.common import (
    CommandManager,
    CommandManagerConfig,
    EventManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from jaxrlworld.rl.envs.managers.registry import ManagerRegistry


def _register_common(*extra_sim_types: str) -> None:
    """Register simulator-agnostic managers shared by all backends."""
    for sim_type in (*BUILTIN_SIM_TYPES, *extra_sim_types):
        ManagerRegistry.register(sim_type, "command", CommandManager, CommandManagerConfig)
        ManagerRegistry.register(sim_type, "reward", RewardManager, RewardConfig)
        ManagerRegistry.register(sim_type, "termination", TerminationManager)
        ManagerRegistry.register(sim_type, "event", EventManager, EventConfig)


def _register_genesis() -> None:
    """Register Genesis-specific managers."""
    from jaxrlworld.rl.envs.managers.genesis import (  # local-import: backend (genesis), only on the genesis path
        ActionManager,
        ActionManagerConfig,
        ContactManager,
        SceneManager,
        SceneManagerConfig,
        VisualizationManager,
        VisualizationManagerConfig,
    )
    from jaxrlworld.rl.envs.managers.genesis.terrain_importer import (
        GenesisTerrainImporter,  # local-import: backend (genesis), only on the genesis path
    )

    ManagerRegistry.register("genesis", "scene", SceneManager, SceneManagerConfig)
    ManagerRegistry.register("genesis", "action", ActionManager, ActionManagerConfig)
    ManagerRegistry.register("genesis", "observation", ObservationManager)
    ManagerRegistry.register("genesis", "contact", ContactManager)
    ManagerRegistry.register("genesis", "visualization", VisualizationManager, VisualizationManagerConfig)
    ManagerRegistry.register("genesis", "terrain", GenesisTerrainImporter)


def _register_newton() -> None:
    """Register Newton-specific managers (lazy — imports warp)."""
    from jaxrlworld.rl.envs.managers.newton import (  # local-import: backend (newton), only on the newton path
        NewtonActionManager,
        NewtonActionManagerConfig,
        NewtonContactManager,
        NewtonSceneManager,
        NewtonSceneManagerConfig,
        NewtonVisualizationManager,
        NewtonVisualizationManagerConfig,
    )
    from jaxrlworld.rl.envs.managers.newton.terrain_importer import (
        NewtonTerrainImporter,  # local-import: backend (newton), only on the newton path
    )

    ManagerRegistry.register("newton", "scene", NewtonSceneManager, NewtonSceneManagerConfig)
    ManagerRegistry.register("newton", "action", NewtonActionManager, NewtonActionManagerConfig)
    ManagerRegistry.register("newton", "observation", ObservationManager)
    ManagerRegistry.register("newton", "contact", NewtonContactManager)
    ManagerRegistry.register("newton", "visualization", NewtonVisualizationManager, NewtonVisualizationManagerConfig)
    ManagerRegistry.register("newton", "terrain", NewtonTerrainImporter)


def _register_mujoco() -> None:
    """Register MuJoCo/mjlab-specific managers (lazy — imports mjlab)."""
    from jaxrlworld.rl.envs.managers.mujoco import (  # local-import: backend (mjlab), only on the mujoco path
        MujocoActionManager,
        MujocoActionManagerConfig,
        MujocoContactManager,
        MujocoSceneManager,
        MujocoSceneManagerConfig,
    )
    from jaxrlworld.rl.envs.managers.mujoco.terrain_importer import (
        MujocoTerrainImporter,  # local-import: backend (mjlab), only on the mujoco path
    )

    ManagerRegistry.register("mujoco", "scene", MujocoSceneManager, MujocoSceneManagerConfig)
    ManagerRegistry.register("mujoco", "action", MujocoActionManager, MujocoActionManagerConfig)
    ManagerRegistry.register("mujoco", "observation", ObservationManager)
    ManagerRegistry.register("mujoco", "contact", MujocoContactManager)
    ManagerRegistry.register("mujoco", "terrain", MujocoTerrainImporter)


def register_all_for(sim_type: str) -> None:
    """Register all managers for a given sim_type.

    Called lazily when a backend is first used, avoiding heavy imports
    until they are actually needed. A sim_type that is not built in is
    looked up among the external backends (:func:`get_backend`, which
    raises ``ValueError`` for an unknown one).
    """
    registrars = {
        "genesis": _register_genesis,
        "newton": _register_newton,
        "mujoco": _register_mujoco,
    }

    registrar = registrars.get(sim_type)
    if registrar is None:
        backend = get_backend(sim_type)
        _register_common(sim_type)
        backend.register_managers()
        return
    _register_common()
    registrar()
