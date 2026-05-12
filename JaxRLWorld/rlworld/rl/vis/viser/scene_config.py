"""Appearance config for the simulator-agnostic Viser scene (eval viewer).

Kept dependency-free (just ``dataclasses``) so it can be referenced from
plain config classes without importing ``viser`` / ``trimesh``.  Applied
by ``rlworld.rl.vis.viser.scene.ViserScene`` for the Genesis/Newton
bridge-rendering path (mjlab brings its own scene).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ViserSceneConfig:
    """Tweakable look of the Viser eval viewer (Genesis/Newton).

    Defaults: a near-white matte ground plane + a near-black metallic
    robot.  Override any field to taste; ``robot_color=None`` keeps the
    simulator's own mesh colors instead.
    """

    # ── Ground ──────────────────────────────────────────────────────
    ground_kind: Literal["plane", "checkerboard", "none"] = "checkerboard"
    """``"checkerboard"`` (default — a subtle near-white grid, like mjlab's
    ground), ``"plane"`` (one flat color), or ``"none"``."""
    ground_color: tuple[int, int, int] = (245, 245, 245)
    """Plain-ground color (RGB 0-255). Also the light cell of the checkerboard."""
    ground_color_alt: tuple[int, int, int] = (225, 225, 225)
    """Dark cell color for ``ground_kind="checkerboard"`` (keep it close to
    ``ground_color`` for a faint-grid look; spread them apart for a chessboard)."""
    ground_size: float = 50.0
    ground_divisions: int = 100
    """Cells per side for the checkerboard (50 m / 100 = 0.5 m cells)."""
    ground_metalness: float = 0.0
    ground_roughness: float = 0.92
    ground_texture: str | None = "default"
    """When set, the ground is a tiled image (overrides ``ground_kind``).
    ``"default"`` → the bundled earthy ``ground_texture.png``; a file path →
    use that image; ``None`` → no texture, fall back to ``ground_kind`` + colors."""
    ground_texture_tiles: float = 25.0
    """How many times the texture repeats across ``ground_size`` (50 m / 25 = 2 m tile)."""

    # ── Robot ───────────────────────────────────────────────────────
    robot_color: tuple[int, int, int] | None = (35, 35, 35)
    """Override color for every robot mesh (RGB 0-255). ``None`` → keep the
    simulator's own mesh colors."""
    robot_metalness: float = 0.85
    robot_roughness: float = 0.35
    robot_opacity: float = 1.0

    # ── Shadows ─────────────────────────────────────────────────────
    cast_shadow: bool = True
    receive_shadow: bool = True
