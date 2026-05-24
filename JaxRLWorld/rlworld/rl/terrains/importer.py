"""Backend-agnostic terrain importer.

Owns the terrain's domain state — generated :class:`TerrainData`, the
sub-terrain origin grid, per-env spawn origins, and per-env terrain
levels (for curriculum). Each backend's scene manager constructs the
appropriate per-sim :class:`TerrainImporter` subclass via
:class:`ManagerRegistry` and asks it to inject the terrain into the
sim's native API.

Origin / curriculum logic is faithfully ported from IsaacLab's
``TerrainImporter`` (see ``terrain_importer.py``): when a generator
produces sub-terrain origins, ``env_origins`` is a (num_envs, 3) tensor
sampling those origins by ``(terrain_level, terrain_type)``;
``update_env_origins`` walks the levels for ``terrain_levels_vel``-style
curriculum. When there is no generator (flat plane) ``env_origins``
defaults to all-zeros — every env spawns at the world origin (matches
the prior Newton / MuJoCo behaviour).

The base class is pure (torch + numpy + our :mod:`rlworld.rl.terrains`),
and imports no simulator. Per-sim subclasses live next to each scene
manager and add the native-API injection method.
"""

from __future__ import annotations

import numpy as np
import torch

from rlworld.rl.configs.scene.terrain_config import TerrainCfg
from rlworld.rl.terrains.terrain_generator import TerrainData, TerrainGenerator


class TerrainImporter:
    """Owns terrain state + origin/curriculum logic. Subclasses inject into the sim."""

    cfg: TerrainCfg
    num_envs: int
    device: torch.device

    # Set when a generator runs.
    data: TerrainData | None
    terrain_origins: torch.Tensor | None  # (num_rows, num_cols, 3); None for plane / no grid
    # Per-env state.
    env_origins: torch.Tensor  # (num_envs, 3)
    terrain_levels: torch.Tensor | None  # (num_envs,) int — None for plane
    terrain_types: torch.Tensor | None  # (num_envs,) int — None for plane
    max_terrain_level: int

    def __init__(self, cfg: TerrainCfg, num_envs: int, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        self.data = None
        self.terrain_origins = None
        self.terrain_levels = None
        self.terrain_types = None
        self.max_terrain_level = 0
        # Default: every env at the origin (matches Newton stacking and
        # mjlab's _default_env_origins zeros). Subclasses override via
        # ``configure_env_origins`` when a generator provides terrain_origins.
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Internal generator helper (used by per-sim subclasses).
    # ------------------------------------------------------------------

    def _run_generator(self) -> TerrainData:
        """Run the configured :class:`TerrainGenerator`, store + return the data."""
        if self.cfg.terrain_generator is None:
            raise ValueError("TerrainCfg(terrain_type='generator') requires a terrain_generator.")
        self.data = TerrainGenerator(self.cfg.terrain_generator).data
        return self.data

    # ------------------------------------------------------------------
    # Origin / curriculum (sim-agnostic).
    # ------------------------------------------------------------------

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None) -> None:
        """Compute per-env spawn origins.

        When ``origins`` is the (num_rows, num_cols, 3) sub-terrain grid
        from the generator, sample by ``(terrain_level, terrain_type)``
        (curriculum-ready). When ``None`` (flat plane / no grid),
        ``env_origins`` stays at all-zeros from ``__init__`` — every env
        at the world origin.
        """
        if origins is None:
            # Keep the all-zeros default.
            self.terrain_origins = None
            return

        if isinstance(origins, np.ndarray):
            origins = torch.from_numpy(origins)
        self.terrain_origins = origins.to(self.device, dtype=torch.float32)
        self.env_origins = self._compute_env_origins_curriculum(self.num_envs, self.terrain_origins)

    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor) -> None:
        """Advance per-env terrain levels (curriculum) and re-pick spawn origins.

        Mirror of IsaacLab's ``TerrainImporter.update_env_origins`` — used
        by ``terrain_levels_vel``-style curriculum terms. No-op when there
        is no sub-terrain grid (plane / single-cell terrain).
        """
        if self.terrain_origins is None or self.terrain_levels is None or self.terrain_types is None:
            return
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        # Robots that clear the top level wrap to a random one; clip below at 0.
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        num_rows, num_cols = origins.shape[:2]
        cfg_cap = getattr(self.cfg, "max_init_terrain_level", None)
        max_init_level = (num_rows - 1) if cfg_cap is None else min(int(cfg_cap), num_rows - 1)
        self.max_terrain_level = num_rows
        self.terrain_levels = torch.randint(0, max_init_level + 1, (num_envs,), device=self.device)
        self.terrain_types = torch.div(
            torch.arange(num_envs, device=self.device),
            (num_envs / num_cols),
            rounding_mode="floor",
        ).to(torch.long)
        env_origins = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        return env_origins

    # ------------------------------------------------------------------
    # Read-only convenience accessors (forwarded from ``data``).
    # ------------------------------------------------------------------

    @property
    def has_terrain(self) -> bool:
        return self.data is not None

    @property
    def half_extent(self) -> tuple[float, float]:
        """In-plane half-extent (Lx/2, Ly/2). ``(inf, inf)`` for plane (no bound)."""
        return self.data.half_extent if self.data is not None else (float("inf"), float("inf"))

    @property
    def heights_m(self) -> np.ndarray | None:
        return self.data.heights_m if self.data is not None else None

    @property
    def horizontal_scale(self) -> float | None:
        return self.data.horizontal_scale if self.data is not None else None

    @property
    def vertical_scale(self) -> float | None:
        return self.data.vertical_scale if self.data is not None else None
