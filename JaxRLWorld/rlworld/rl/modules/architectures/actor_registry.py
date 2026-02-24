from typing import Type
import importlib

ACTOR_REGISTRY: dict[str, Type] = {}

# Auto-register from these modules
_ACTOR_MODULES = [
    "rlworld.rl.modules.architectures.mlp.actor",
    "rlworld.rl.modules.architectures.aba.actor",
    # "rlworld.rl.modules.architectures.rodrigues.actor",
    "rlworld.rl.modules.architectures.body_transformer.actor",
    # "rlworld.rl.modules.architectures.crba.actor",
    "rlworld.rl.modules.architectures.gnn.actor"
]


def _build_registry():
    """Scan modules and register all *Actor classes"""
    for module_path in _ACTOR_MODULES:
        try:
            module = importlib.import_module(module_path)
            for name in dir(module):
                if name.endswith("Actor"):
                    obj = getattr(module, name)
                    if isinstance(obj, type):
                        ACTOR_REGISTRY[name] = obj

        except ImportError:
            raise NotImplementedError(f"module {module_path} not found")


def get_actor_class(name: str) -> Type:
    """Get actor class by name"""
    if not ACTOR_REGISTRY:
        _build_registry()
    if name not in ACTOR_REGISTRY:
        raise ValueError(
            f"Unknown actor: {name}. "
            f"Available: {list(ACTOR_REGISTRY.keys())}"
        )
    return ACTOR_REGISTRY[name]