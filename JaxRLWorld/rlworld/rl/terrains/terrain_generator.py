"""Backend-agnostic height-field terrain generator.

Tiles a ``num_rows × num_cols`` grid of height-field sub-terrains into a
single height grid (in metres) centred on the origin, and records each
sub-terrain's spawn origin. The grid + curriculum structure is kept so
terrain-level curriculum / per-env origins can be layered on later.

The output (:class:`TerrainData`) is a pure numpy height grid + scales +
origins, with no simulator dependency. Each backend's terrain importer
feeds the grid to its native heightfield API — a heightfield is the only
representation that collides correctly (non-convex) on all three of
Genesis / Newton / MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sub_terrain_cfg import SubTerrainCfg


@dataclass
class TerrainData:
    """Canonical generated terrain shared across all backends."""

    heights_m: np.ndarray
    """``(nrow, ncol)`` elevation grid in **metres**."""
    horizontal_scale: float
    """Metres between adjacent grid cells."""
    vertical_scale: float
    """Height quantisation in metres (used by the Genesis backend)."""
    size_xy: tuple[float, float]
    """Total physical span ``(Lx, Ly)`` of the grid in metres, centred on the origin."""
    origins: np.ndarray
    """Per-sub-terrain spawn origins. Shape ``(num_rows, num_cols, 3)``."""

    @property
    def nrow(self) -> int:
        return int(self.heights_m.shape[0])

    @property
    def ncol(self) -> int:
        return int(self.heights_m.shape[1])

    @property
    def half_extent(self) -> tuple[float, float]:
        return (self.size_xy[0] / 2.0, self.size_xy[1] / 2.0)

    @property
    def max_height_m(self) -> float:
        return float(self.heights_m.max())

    @property
    def min_height_m(self) -> float:
        return float(self.heights_m.min())


@dataclass
class TerrainGeneratorCfg:
    """Configuration for :class:`TerrainGenerator`."""

    size: tuple[float, float] = (8.0, 8.0)
    """Width (x) and length (y) of each sub-terrain patch (m)."""
    num_rows: int = 1
    num_cols: int = 1
    sub_terrains: dict[str, SubTerrainCfg] = field(default_factory=dict)
    difficulty_range: tuple[float, float] = (0.0, 1.0)
    curriculum: bool = False
    border_width: float = 0.0
    """Width (m) of a flat (zero-height) border around the whole grid."""
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    seed: int | None = 0


class TerrainGenerator:
    """Tile sub-terrain height grids into one centred grid + origins."""

    def __init__(self, cfg: TerrainGeneratorCfg):
        if len(cfg.sub_terrains) == 0:
            raise ValueError("No sub-terrains specified! Add at least one to TerrainGeneratorCfg.sub_terrains.")
        self.cfg = cfg

        for sub_cfg in cfg.sub_terrains.values():
            sub_cfg.size = cfg.size
            sub_cfg.horizontal_scale = cfg.horizontal_scale
            sub_cfg.vertical_scale = cfg.vertical_scale

        seed = cfg.seed if cfg.seed is not None else np.random.get_state()[1][0]
        self.np_rng = np.random.default_rng(seed)

        # Per-sub-terrain pixel dims.
        self._rows_px = int(round(cfg.size[0] / cfg.horizontal_scale))
        self._cols_px = int(round(cfg.size[1] / cfg.horizontal_scale))
        self._border_px = int(round(cfg.border_width / cfg.horizontal_scale))

        big_rows = cfg.num_rows * self._rows_px + 2 * self._border_px
        big_cols = cfg.num_cols * self._cols_px + 2 * self._border_px
        heights = np.zeros((big_rows, big_cols), dtype=np.float32)  # flat border = 0
        origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)

        total_x = cfg.num_rows * cfg.size[0] + 2 * cfg.border_width
        total_y = cfg.num_cols * cfg.size[1] + 2 * cfg.border_width

        sub_cfgs = list(cfg.sub_terrains.values())
        proportions = np.array([c.proportion for c in sub_cfgs], dtype=np.float64)
        proportions /= proportions.sum()

        for r in range(cfg.num_rows):
            for c in range(cfg.num_cols):
                cell = r * cfg.num_cols + c
                if cfg.curriculum:
                    sub_idx = int(np.min(np.where(c / cfg.num_cols + 1e-3 < np.cumsum(proportions))[0]))
                    lower, upper = cfg.difficulty_range
                    difficulty = lower + (upper - lower) * (r + self.np_rng.uniform()) / cfg.num_rows
                else:
                    sub_idx = int(self.np_rng.choice(len(proportions), p=proportions))
                    difficulty = float(self.np_rng.uniform(*cfg.difficulty_range))

                grid = self._make_sub_grid(difficulty, sub_cfgs[sub_idx], cell)

                r0 = self._border_px + r * self._rows_px
                c0 = self._border_px + c * self._cols_px
                heights[r0 : r0 + self._rows_px, c0 : c0 + self._cols_px] = grid

                # World centre of this sub-terrain + surface height there.
                ox = -total_x / 2.0 + cfg.border_width + (r + 0.5) * cfg.size[0]
                oy = -total_y / 2.0 + cfg.border_width + (c + 0.5) * cfg.size[1]
                oz = float(grid[self._rows_px // 2, self._cols_px // 2])
                origins[r, c] = (ox, oy, oz)

        self.data = TerrainData(
            heights_m=heights,
            horizontal_scale=cfg.horizontal_scale,
            vertical_scale=cfg.vertical_scale,
            size_xy=(total_x, total_y),
            origins=origins,
        )

    def _make_sub_grid(self, difficulty: float, cfg: SubTerrainCfg, cell_index: int) -> np.ndarray:
        cfg = cfg.copy()
        cfg.difficulty = float(difficulty)
        cfg.seed = None if self.cfg.seed is None else self.cfg.seed + cell_index
        grid = np.asarray(cfg.function(difficulty, cfg), dtype=np.float32)
        if grid.shape != (self._rows_px, self._cols_px):
            raise ValueError(
                f"Sub-terrain '{cfg.function.__name__}' returned grid {grid.shape}, "
                f"expected {(self._rows_px, self._cols_px)}."
            )
        return grid
