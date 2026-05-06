"""Callable ↔ string reference conversion utilities.

String format: ``"module.path:qualified.name"``

Examples::

    "rlworld.rl.envs.rewards.common:track_lin_vel"
    "rlworld.rl.envs.managers.common.gait:QuadrupedOffsets"
"""

import importlib
from typing import Callable


def callable_to_string(fn: Callable) -> str:
    """Convert a callable to its ``"module:qualname"`` string reference.

    Raises ``ValueError`` for lambdas or objects without proper module info.
    """
    if getattr(fn, "__name__", "") == "<lambda>":
        raise ValueError(f"Cannot serialize lambda functions. Convert to a named function: {fn}")

    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)

    if module is None or qualname is None:
        raise ValueError(
            f"Cannot determine module/qualname for {fn!r}. "
            "Ensure it is a named function or class defined at module level."
        )

    # Reject locals (e.g. functions defined inside other functions)
    if "<locals>" in qualname:
        raise ValueError(f"Cannot serialize locally-defined callable {module}:{qualname}. Move it to module level.")

    return f"{module}:{qualname}"


def resolve_callable(ref: str) -> Callable:
    """Resolve a ``"module:qualname"`` string to the actual callable.

    Raises ``ImportError`` if the module cannot be found,
    ``AttributeError`` if the attribute path is invalid.
    """
    if ":" not in ref:
        raise ValueError(f"Invalid callable reference {ref!r}. Expected 'module.path:attr.name' format.")

    module_path, attr_path = ref.split(":", 1)
    module = importlib.import_module(module_path)

    obj = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)

    if not callable(obj):
        raise TypeError(f"Resolved {ref!r} to {obj!r}, which is not callable.")

    return obj
