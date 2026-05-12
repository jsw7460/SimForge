"""Overlay exports.

``base`` is sim-agnostic and loads eagerly.  ``command_arrow`` and
``text_hud`` ``import genesis`` at module load (the GL rasterizer overlays are
Genesis-only), so they are exposed lazily via ``__getattr__`` — importing
``rlworld.rl.vis.overlays`` no longer drags Genesis in.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .base import (
    Base2DOverlay,
    Base3DOverlay,
    BaseOverlay,
    Overlay2DConfig,
    Overlay3DConfig,
    OverlayConfig,
)

# name → (submodule, attr) for lazily-loaded, Genesis-importing overlays.
_LAZY: dict[str, tuple[str, str]] = {
    "CommandArrowConfig": (".command_arrow", "CommandArrowConfig"),
    "CommandArrowOverlay": (".command_arrow", "CommandArrowOverlay"),
    "TextHUDConfig": (".text_hud", "TextHUDConfig"),
    "TextHUDOverlay": (".text_hud", "TextHUDOverlay"),
}

__all__ = [
    "BaseOverlay",
    "Base2DOverlay",
    "Base3DOverlay",
    "OverlayConfig",
    "Overlay2DConfig",
    "Overlay3DConfig",
    *_LAZY,
]


def __getattr__(name: str):
    if name in _LAZY:
        submod, attr = _LAZY[name]
        return getattr(importlib.import_module(submod, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # let type checkers / IDEs see the lazy names
    from .command_arrow import CommandArrowConfig, CommandArrowOverlay
    from .text_hud import TextHUDConfig, TextHUDOverlay
