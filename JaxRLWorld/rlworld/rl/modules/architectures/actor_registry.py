import importlib
from typing import Type

ACTOR_REGISTRY: dict[str, Type] = {}

# Built-in actor modules. External packages register additional actors
# via ``register_actor`` after importing them.
_BUILTIN_ACTOR_MODULES = [
    "rlworld.rl.modules.architectures.mlp.actor",
    "rlworld.rl.modules.architectures.space_time_transformer.actor",
]


def register_actor(actor_class: Type) -> None:
    """Register an actor class so it is resolvable by ``get_actor_class``.

    External packages plug in custom actors by importing this function
    and calling it at import time.
    """
    ACTOR_REGISTRY[actor_class.__name__] = actor_class


def _build_registry() -> None:
    """Scan built-in modules and register all ``*Actor`` classes."""
    for module_path in _BUILTIN_ACTOR_MODULES:
        module = importlib.import_module(module_path)
        for name in dir(module):
            if name.endswith("Actor"):
                obj = getattr(module, name)
                if isinstance(obj, type):
                    ACTOR_REGISTRY[name] = obj


def get_actor_class(name: str) -> Type:
    """Get actor class by name."""
    if not ACTOR_REGISTRY:
        _build_registry()
    if name not in ACTOR_REGISTRY:
        raise ValueError(f"Unknown actor: {name}. Available: {list(ACTOR_REGISTRY.keys())}")
    return ACTOR_REGISTRY[name]
