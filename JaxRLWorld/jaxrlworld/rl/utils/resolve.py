"""Callable ↔ string reference conversion utilities.

String format: ``"module.path:qualified.name"``

Examples::

    "jaxrlworld.rl.envs.rewards.common:track_lin_vel"
    "jaxrlworld.rl.envs.managers.common.gait:QuadrupedOffsets"
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


#: What the package was called before 2026-09. Checkpoints written then
#: record module paths under this name; see the error raised below.
_FORMER_PACKAGE = "rlworld"


def resolve_callable(ref: str) -> Callable:
    """Resolve a ``"module:qualname"`` string to the actual callable.

    Raises ``ImportError`` if the module cannot be found, and says how to
    migrate when the module is missing only because the reference predates
    the package's rename. ``AttributeError`` if the attribute path is
    invalid.
    """
    if ":" not in ref:
        raise ValueError(f"Invalid callable reference {ref!r}. Expected 'module.path:attr.name' format.")

    module_path, attr_path = ref.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if module_path.split(".", 1)[0] != _FORMER_PACKAGE:
            raise
        raise ModuleNotFoundError(
            f"{ref!r} names this package as {_FORMER_PACKAGE!r}, which it was called "
            "before it was renamed to 'jaxrlworld'. A checkpoint saved then records "
            "its reward, termination and event functions by module path, so its "
            "config.yaml has to be rewritten before it will load:\n\n"
            "    python -m jaxrlworld.scripts.migrate_checkpoint_module_paths <checkpoint-dir>\n\n"
            "The rewrite is not done here on the reader's behalf. A checkpoint is a "
            "record of what was run; quietly reinterpreting one leaves no trace that "
            "it was ever written under a different name."
        ) from exc

    obj = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)

    if not callable(obj):
        raise TypeError(f"Resolved {ref!r} to {obj!r}, which is not callable.")

    return obj
