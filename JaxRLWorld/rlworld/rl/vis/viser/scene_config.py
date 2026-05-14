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

    Default look is a "polished gallery": dark veined marble under glass-
    like polish (the bundled ``marble_texture.png`` shows through the
    gloss so the floor is never featureless) + a carbon-finish robot,
    lit by the built-in ``"studio"`` HDRI for clean specular reflections
    plus a soft directional sun for shadow.

    Override any field to taste; ``robot_color=None`` keeps the
    simulator's own mesh colors instead.  See ``looks.py`` for ready-
    made alternatives (``"earthy"``, ``"warehouse"``, ``"construction"`` …).
    """

    # ── Ground ──────────────────────────────────────────────────────
    ground_kind: Literal["plane", "checkerboard", "none"] = "plane"
    """Fallback ground when ``ground_texture`` is ``None``.  ``"plane"``
    (one flat colour), ``"checkerboard"`` (a subtle grid), or ``"none"``."""
    ground_color: tuple[int, int, int] = (28, 32, 38)
    """Plain-ground color (RGB 0-255). Also the light cell of the checkerboard."""
    ground_color_alt: tuple[int, int, int] = (22, 26, 32)
    """Dark cell color for ``ground_kind="checkerboard"`` (keep it close to
    ``ground_color`` for a faint-grid look; spread them apart for a chessboard)."""
    ground_size: float = 50.0
    ground_divisions: int = 100
    """Cells per side for the checkerboard (50 m / 100 = 0.5 m cells)."""
    ground_metalness: float = 0.35
    ground_roughness: float = 0.08
    """Very low roughness + the HDRI ``env_map`` → glass-polished specular.
    The bundled ``marble_texture.png`` shows through so the floor still
    has a pattern (it would otherwise read as a featureless mirror)."""
    ground_texture: str | None = "marble"
    """When set, the ground is a tiled image (overrides ``ground_kind``).
    ``"default"`` → the bundled earthy ``ground_texture.png``; ``"marble"``
    (default) → the bundled dark veined ``marble_texture.png``; ``"concrete"``
    → the bundled cool-gray ``concrete_texture.png``; a file path → use
    that image; ``None`` → no texture, fall back to ``ground_kind`` + colors."""
    ground_texture_tiles: float = 25.0
    """How many times the texture repeats across ``ground_size`` (50 m / 25 = 2 m tile)."""

    # ── Robot ───────────────────────────────────────────────────────
    robot_color: tuple[int, int, int] | None = (34, 38, 42)
    """Override color for every robot mesh (RGB 0-255). Default is a dark
    carbon-fibre finish.  ``None`` → keep the simulator's own mesh colors."""
    robot_metalness: float = 0.25
    robot_roughness: float = 0.4
    robot_opacity: float = 1.0

    # ── Shadows ─────────────────────────────────────────────────────
    cast_shadow: bool = True
    receive_shadow: bool = True

    # ── Image-based lighting (HDRI) ─────────────────────────────────
    env_map: str | None = "studio"
    """Built-in HDRI preset that drives image-based lighting (IBL) — what
    makes low-roughness materials show clean reflections (the glass /
    polished-floor look).  Options: ``'apartment'``, ``'city'``, ``'dawn'``,
    ``'forest'``, ``'lobby'``, ``'night'``, ``'park'``, ``'studio'``,
    ``'sunset'``, ``'warehouse'``.  ``None`` disables IBL — only the
    explicit lights below contribute."""
    env_map_intensity: float = 1.0
    """Multiplier on the HDRI's contribution to material shading."""
    env_map_as_background: bool = True
    """Show the HDRI itself as the canvas background (rotates with the
    camera).  When ``True``, the procedural ``sky_*`` backdrop is skipped."""
    env_map_blurriness: float = 0.25
    """0 = sharp HDRI behind the robot; 1 = heavily blurred (good when you
    don't want the backdrop competing for attention)."""

    # ── Lighting (outdoor rig) ──────────────────────────────────────
    lighting: bool = True
    """Add an explicit light rig on top of the HDRI: a soft ambient floor +
    a hemisphere fill + a shadow-casting directional "sun".  When
    ``env_map`` is set the HDRI already provides ambient/diffuse so the
    ambient/hemisphere intensities are kept low by default; the sun is
    still needed for crisp shadows.  Set ``lighting=False`` to skip."""
    sun_direction: tuple[float, float, float] = (-0.5, -0.35, -0.85)
    """Direction the sunlight *travels* (will be normalized) — down + from a corner."""
    sun_color: tuple[int, int, int] = (255, 240, 215)
    sun_intensity: float = 0.7
    sun_cast_shadow: bool = True
    ambient_intensity: float = 0.18
    hemisphere_intensity: float = 0.22
    hemisphere_ground_color: tuple[int, int, int] = (60, 64, 72)
    """Hemisphere-light bounce color from the ground (sky side uses ``sky_color``)."""

    # ── Sky background (only used when ``env_map_as_background=False``) ──
    sky_background: bool = True
    """Master switch for the flat canvas-background image (the backdrop
    does not rotate with the camera).  ``False`` disables it entirely."""
    sky_kind: str = "gradient"
    """Which backdrop to use when ``sky_background=True``.  ``"gradient"``
    (default) → procedural sky from ``sky_color`` / ``sky_horizon_color``
    / ``sky_sun_glow``; ``"construction"`` → the bundled hazy
    construction-site panorama (sky + crane + scaffolded buildings); a
    file path → load that image as the backdrop."""
    sky_color: tuple[int, int, int] = (138, 184, 235)
    """Top-of-sky color (also the hemisphere light's sky color)."""
    sky_horizon_color: tuple[int, int, int] = (226, 234, 240)
    """Hazy color near the horizon."""
    sky_sun_glow: bool = True
    """Add a soft warm glow in the upper part of the sky backdrop."""
