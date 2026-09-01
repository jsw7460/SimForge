"""Sub-terrain configuration base (backend-agnostic, height-field).

Every sub-terrain declares a ``function(difficulty, cfg) -> np.ndarray``
returning a 2D height grid **in metres**. The generator tiles these into
one big grid; each backend's terrain importer feeds that grid to its
native heightfield API (Genesis / Newton / MuJoCo all collide a
heightfield non-convexly, which a single triangle mesh would not on
MuJoCo — hence height field, not mesh, is the cross-sim canonical form).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class SubTerrainCfg:
    """Base config for one height-field sub-terrain type.

    ``function`` returns a 2D height grid in metres of shape
    ``(size[0] / horizontal_scale, size[1] / horizontal_scale)``.
    ``size`` / ``horizontal_scale`` / ``vertical_scale`` are set by the
    :class:`TerrainGeneratorCfg` so all sub-terrains share one
    discretisation. ``difficulty`` / ``seed`` are assigned per sub-terrain
    by the generator.
    """

    function: Callable[[float, SubTerrainCfg], np.ndarray] = None
    proportion: float = 1.0
    size: tuple[float, float] = (8.0, 8.0)
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    difficulty: float = 0.0
    seed: int | None = None

    def copy(self) -> SubTerrainCfg:
        return replace(self)
