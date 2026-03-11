"""Manager registry for automatic backend-specific manager resolution.

Maps (sim_type, role) pairs to the correct manager class, enabling
simulator-agnostic environment setup without hardcoded imports.

Example:
    # Registration (typically at module level in each manager package)
    ManagerRegistry.register("genesis", "action", ActionManager, ActionManagerConfig)
    ManagerRegistry.register("newton", "action", NewtonActionManager, NewtonActionManagerConfig)

    # Resolution (in environment setup)
    cls = ManagerRegistry.get_class("genesis", "action")
    manager = cls(env=self, config=cfg)
"""

from __future__ import annotations

from typing import Any, Type


class ManagerRegistry:
    """Registry mapping (sim_type, manager_role) to manager and config classes."""

    _registry: dict[str, dict[str, dict[str, Any]]] = {}

    @classmethod
    def register(
        cls,
        sim_type: str,
        role: str,
        manager_class: Type,
        config_class: Type | None = None,
    ) -> None:
        """Register a manager class for a (sim_type, role) pair.

        Args:
            sim_type: Simulator backend ("genesis", "newton", "mujoco").
            role: Manager role ("scene", "action", "observation", "contact",
                  "reward", "termination", "command", "event", "visualization").
            manager_class: The manager class to instantiate.
            config_class: Optional config dataclass for this manager.
        """
        if sim_type not in cls._registry:
            cls._registry[sim_type] = {}
        cls._registry[sim_type][role] = {
            "manager_class": manager_class,
            "config_class": config_class,
        }

    @classmethod
    def get_class(cls, sim_type: str, role: str) -> Type:
        """Get the manager class for the given (sim_type, role)."""
        entry = cls._get_entry(sim_type, role)
        return entry["manager_class"]

    @classmethod
    def get_config_class(cls, sim_type: str, role: str) -> Type | None:
        """Get the config class for the given (sim_type, role)."""
        entry = cls._get_entry(sim_type, role)
        return entry["config_class"]

    @classmethod
    def create(cls, sim_type: str, role: str, **kwargs) -> Any:
        """Create a manager instance for the given (sim_type, role)."""
        manager_cls = cls.get_class(sim_type, role)
        return manager_cls(**kwargs)

    @classmethod
    def has(cls, sim_type: str, role: str) -> bool:
        """Check if a (sim_type, role) pair is registered."""
        return sim_type in cls._registry and role in cls._registry[sim_type]

    @classmethod
    def available_roles(cls, sim_type: str) -> list[str]:
        """List registered roles for a sim_type."""
        return list(cls._registry.get(sim_type, {}).keys())

    @classmethod
    def available_sim_types(cls) -> list[str]:
        """List all registered sim_types."""
        return list(cls._registry.keys())

    # ------------------------------------------------------------------
    @classmethod
    def _get_entry(cls, sim_type: str, role: str) -> dict[str, Any]:
        if sim_type not in cls._registry:
            raise KeyError(f"No managers registered for sim_type={sim_type!r}")
        if role not in cls._registry[sim_type]:
            raise KeyError(
                f"No {role!r} manager registered for sim_type={sim_type!r}. "
                f"Available roles: {list(cls._registry[sim_type].keys())}"
            )
        return cls._registry[sim_type][role]
