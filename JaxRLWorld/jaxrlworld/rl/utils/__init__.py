"""Utility exports.

The console helpers are imported eagerly; ``jaxrlworld.rl.utils.utils``
pulls in torch, so its names are exposed lazily through ``__getattr__``
and a config-only import (``jaxrlworld.rl.configs`` reaches
``jaxrlworld.rl.utils.resolve`` through this package) stays torch-free.
"""

from __future__ import annotations

import importlib

from .pretty import (
    create_env_panel,
    create_manager_table,
    format_shape,
    format_weight,
    get_console,
    panel_to_string,
    print_env_summary,
    table_to_string,
)

_LAZY: dict[str, str] = {
    "compare_dicts": ".utils",
    "deprecated": ".utils",
    "gs_rand_float": ".utils",
    "set_seed": ".utils",
    "setup_log_dir": ".utils",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_env_panel",
    "create_manager_table",
    "format_shape",
    "format_weight",
    "get_console",
    "panel_to_string",
    "print_env_summary",
    "table_to_string",
    "compare_dicts",
    "deprecated",
    "gs_rand_float",
    "set_seed",
    "setup_log_dir",
]
