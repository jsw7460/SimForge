"""Unified terrain/ground scene-entity configuration (sim-agnostic).

:class:`TerrainCfg` is the single ground abstraction — a flat plane and a
generated rough terrain are the same config with a different
``terrain_type`` (mirrors IsaacLab's ``TerrainImporterCfg`` and mjlab's
``TerrainEntityCfg``). It replaces the older flat-vs-rough split.

For ``terrain_type="generator"`` the canonical geometry comes from
:class:`~rlworld.rl.terrains.TerrainGenerator` (a single centred
``trimesh`` + spawn origins); each backend's terrain importer turns that
mesh into native collision geometry, so the same terrain is identical
across Genesis / Newton / MuJoCo.

This module imports no simulator and (via a TYPE_CHECKING-only import of
``TerrainGeneratorCfg``) does not eagerly pull in ``trimesh`` — the
generator package is only imported by the per-backend importers and by
presets that actually build a generated terrain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rlworld.rl.terrains import TerrainGeneratorCfg


@dataclass
class TerrainCfg:
    """Ground specification: flat plane or generated heightfield terrain.

    Owned by the per-sim ``TerrainImporter`` (constructed inside each
    ``SceneManager``); not an entry in the ``entities`` dict.
    """

    terrain_type: Literal["plane", "generator"] = "plane"
    """``"plane"`` → flat ground; ``"generator"`` → mesh from ``terrain_generator``."""

    terrain_generator: TerrainGeneratorCfg | None = None
    """Required when ``terrain_type="generator"``; ignored for ``"plane"``."""

    # ── Contact material ──────────────────────────────────────────────
    contact_stiffness: float = 2.5e3
    contact_damping: float = 100.0
    friction: float = 1.0
    ground_kf: float = 1000.0
    ground_mu_rolling: float = 0.0001
    ground_mu_torsional: float = 0.005

    # ── Newton-specific ───────────────────────────────────────────────
    contact_margin: float = 0.005
    """Per-shape collision margin (m) for the Newton backend's mesh
    terrain. IsaacLab flags this as the single most important Newton
    rough-terrain setting; ignored by Genesis / MuJoCo."""

    # ── Curriculum (used by TerrainImporter when terrain_origins exists) ──
    max_init_terrain_level: int | None = None
    """Cap on the initial terrain level when sampling env_origins from a
    sub-terrain grid (mirror of IsaacLab's ``TerrainImporterCfg``). ``None``
    → use the top row (``num_rows - 1``)."""
