"""Visualization exports.

The overlay classes ``import genesis`` at module load (Genesis is the only
backend that uses this GL rasterizer path), so they are exposed lazily via
``__getattr__`` — importing ``jaxrlworld.rl.vis`` (or one of its sim-agnostic
submodules, e.g. ``jaxrlworld.rl.vis.viser``) no longer drags Genesis into a
Newton- or MuJoCo-only process.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# name → (submodule, attr) for lazily-loaded, Genesis-importing names.
_LAZY: dict[str, tuple[str, str]] = {
    # Base classes (sim-agnostic, but hoisted here for a stable import path)
    "BaseOverlay": (".overlays", "BaseOverlay"),
    "Base2DOverlay": (".overlays", "Base2DOverlay"),
    "Base3DOverlay": (".overlays", "Base3DOverlay"),
    "OverlayConfig": (".overlays", "OverlayConfig"),
    "Overlay2DConfig": (".overlays", "Overlay2DConfig"),
    "Overlay3DConfig": (".overlays", "Overlay3DConfig"),
    "OverlayState": (".overlays", "OverlayState"),
    # Overlays (Genesis-importing)
    "CommandArrowOverlay": (".overlays", "CommandArrowOverlay"),
    "CommandArrowConfig": (".overlays", "CommandArrowConfig"),
    "TextHUDOverlay": (".overlays", "TextHUDOverlay"),
    "TextHUDConfig": (".overlays", "TextHUDConfig"),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        submod, attr = _LAZY[name]
        return getattr(importlib.import_module(submod, __name__), attr)  # dynamic-attr: lazy export named in _LAZY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # let type checkers / IDEs see the lazy names
    from .overlays import (
        Base2DOverlay,
        Base3DOverlay,
        BaseOverlay,
        CommandArrowConfig,
        CommandArrowOverlay,
        Overlay2DConfig,
        Overlay3DConfig,
        OverlayConfig,
        OverlayState,
        TextHUDConfig,
        TextHUDOverlay,
    )
