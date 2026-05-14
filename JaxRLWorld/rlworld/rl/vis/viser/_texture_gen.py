"""Procedural images for the Viser eval scene.

Pure numpy + PIL.  Run as a script to (re)bake the bundled PNGs under
``assets/``; tweak parameters here and re-run if you want a different
look.

Currently bakes:

* ``assets/ground_texture.png`` — tileable earthy ground (used by the
  ``"earthy"`` look).
* ``assets/marble_texture.png`` — tileable dark marble with veining
  (the default polished-glass ground — the veins read through the glass
  reflection so the floor doesn't look featureless).
* ``assets/concrete_texture.png`` — tileable cool-gray concrete slab.
* ``assets/construction_backdrop.png`` — wide hazy construction-site
  panorama (sky + crane + buildings); used as a flat canvas backdrop.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _tileable_fbm(size: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """A periodic (wrap-around) fractal-noise field in [0, 1].

    White noise → FFT → 1/f^beta radial spectrum → IFFT; the result is
    naturally tileable.  Larger ``beta`` → smoother / larger features.
    """
    spectrum = np.fft.fft2(rng.standard_normal((size, size)))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    radius[0, 0] = 1.0
    shaped = spectrum / (radius ** (beta / 2.0))
    shaped[0, 0] = 0.0  # zero mean
    field = np.fft.ifft2(shaped).real
    field -= field.min()
    field /= max(field.max(), 1e-9)
    return field


def generate_ground_texture(size: int = 512, seed: int = 7) -> Image.Image:
    """Earthy ground (dark soil → dry tan) with grain, grit and faint cracks — tileable."""
    rng = np.random.default_rng(seed)

    # Multi-octave height field: a few big patches + medium + fine grain.
    h = (
        0.62 * _tileable_fbm(size, beta=3.8, rng=rng)
        + 0.28 * _tileable_fbm(size, beta=2.6, rng=rng)
        + 0.16 * _tileable_fbm(size, beta=1.6, rng=rng)
        + 0.08 * _tileable_fbm(size, beta=0.8, rng=rng)
    )
    h -= h.min()
    h /= max(h.max(), 1e-9)
    h = np.clip((h - 0.5) * 1.35 + 0.5, 0.0, 1.0)  # punch up the contrast

    dark = np.array([74, 65, 53], dtype=np.float64)  # damp soil
    light = np.array([176, 160, 127], dtype=np.float64)  # dry tan
    rgb = dark[None, None, :] * (1.0 - h[..., None]) + light[None, None, :] * h[..., None]

    # Coarse "pebbles/clods": a medium-freq field pushed to high contrast → scattered
    # lighter/darker blobs that survive being viewed from far away.
    clods = _tileable_fbm(size, beta=1.3, rng=rng)
    clods = np.clip((clods - 0.5) * 4.5, -1.0, 1.0)
    rgb += clods[..., None] * 26.0

    # Per-pixel grit (i.i.d. → tileable) — fine soil grain up close.
    rgb += (rng.random((size, size, 1)) - 0.5) * 32.0

    # A network of darker cracks: ridges near the 0.5 level set of a mid-freq field.
    cracks = _tileable_fbm(size, beta=2.1, rng=rng)
    crack_mask = np.exp(-((cracks - 0.5) ** 2) / (2 * 0.022**2))
    rgb -= crack_mask[..., None] * 42.0

    # A whisper of mossy green in the lowest, dampest spots.
    moss = np.clip(0.45 - h, 0.0, 0.45) / 0.45
    rgb[..., 1] += moss * 10.0
    rgb[..., 0] -= moss * 4.0

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def generate_marble_texture(size: int = 1024, seed: int = 17) -> Image.Image:
    """Tileable dark marble — slate base + long flowing veins at two scales.

    Designed to sit under a low-roughness PBR material (the polished-glass
    default) — the veins remain visible through the gloss so the floor
    doesn't read as featureless plastic.  Veining uses high-beta (smooth)
    fields so the iso-level lines form long, continuous strands rather
    than spotty patches.
    """
    rng = np.random.default_rng(seed)

    # Deep slate base with slow color blotching for natural variation.
    base_blotch = _tileable_fbm(size, beta=3.6, rng=rng)
    base_blotch = (base_blotch - base_blotch.mean()) * 1.2
    base = np.array([28, 32, 38], dtype=np.float64)
    rgb = base[None, None, :] + base_blotch[..., None] * 16.0

    # Primary vein system: a very smooth field → its 0.5 iso-level is one
    # long sinuous strand winding across the tile.  Warm-tinted highlight.
    veins1 = _tileable_fbm(size, beta=3.3, rng=rng)
    vein_mask1 = np.exp(-((veins1 - 0.5) ** 2) / (2 * 0.018**2))
    rgb += vein_mask1[..., None] * 55.0 * np.array([1.00, 0.96, 0.88])[None, None, :]

    # Secondary, finer crossing veins on a slightly less smooth field.
    veins2 = _tileable_fbm(size, beta=2.6, rng=rng)
    vein_mask2 = np.exp(-((veins2 - 0.5) ** 2) / (2 * 0.011**2))
    rgb += vein_mask2[..., None] * 28.0 * np.array([0.95, 0.97, 1.05])[None, None, :]

    # Whisper of per-pixel grain — keeps it from reading as plastic.
    rgb += (rng.random((size, size, 1)) - 0.5) * 3.5

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def generate_concrete_texture(size: int = 512, seed: int = 11) -> Image.Image:
    """Tileable poured-concrete slab — cool gray base + dirt stains + grit + hairline cracks."""
    rng = np.random.default_rng(seed)

    # Base: a slow blotchy field around mid-gray; just enough variation to read
    # as "real" concrete from a distance.
    blotch = 0.7 * _tileable_fbm(size, beta=3.5, rng=rng) + 0.3 * _tileable_fbm(size, beta=2.2, rng=rng)
    blotch = (blotch - blotch.mean()) * 1.1  # roughly zero-mean, mild contrast
    base = np.array([172, 174, 176], dtype=np.float64)  # cool gray (slight blue lift)
    rgb = base[None, None, :] + blotch[..., None] * 28.0

    # Coarse darker stains (oil / water / pour seams) — a punched-up mid-freq field.
    stains = _tileable_fbm(size, beta=2.0, rng=rng)
    stain_mask = np.clip((stains - 0.62) * 4.0, 0.0, 1.0)
    rgb -= stain_mask[..., None] * np.array([20.0, 18.0, 16.0])[None, None, :]

    # Fine aggregate / grit — per-pixel iid noise (tileable trivially).
    rgb += (rng.random((size, size, 1)) - 0.5) * 18.0

    # Tiny darker speckles to suggest embedded aggregate.
    aggr = rng.random((size, size))
    aggr_mask = (aggr > 0.985).astype(np.float64)
    rgb -= aggr_mask[..., None] * 35.0

    # Hairline cracks: thin ridges near the 0.5 level set of a high-freq field
    # (narrower than the soil texture's cracks, and rarer).
    cracks = _tileable_fbm(size, beta=2.4, rng=rng)
    crack_mask = np.exp(-((cracks - 0.5) ** 2) / (2 * 0.013**2))
    rgb -= crack_mask[..., None] * 28.0

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _draw_scaffolded_block(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    *,
    cols: int = 6,
    rows: int = 8,
    grid_color: tuple[int, int, int] | None = None,
) -> None:
    """A boxy building silhouette overlaid with a faint scaffolding grid."""
    draw.rectangle((x0, y0, x1, y1), fill=color)
    if grid_color is None:
        gc = tuple(int(c * 0.55 + 30) for c in color)
    else:
        gc = grid_color
    for i in range(1, cols):
        x = x0 + int((x1 - x0) * i / cols)
        draw.line((x, y0, x, y1), fill=gc, width=1)
    for j in range(1, rows):
        y = y0 + int((y1 - y0) * j / rows)
        draw.line((x0, y, x1, y), fill=gc, width=1)


def _draw_tower_crane(
    draw: ImageDraw.ImageDraw,
    base_x: int,
    base_y: int,
    height: int,
    jib_len: int,
    counter_len: int,
    color: tuple[int, int, int],
) -> None:
    """A simple tower-crane silhouette: vertical mast + cross-shaped jib at the top."""
    mast_w = max(4, height // 80)
    top_y = base_y - height
    # Mast.
    draw.rectangle((base_x - mast_w // 2, top_y, base_x + mast_w // 2, base_y), fill=color)
    # Horizontal jib (long arm right + short counter-jib left).
    jib_thickness = max(3, height // 110)
    draw.rectangle(
        (base_x - counter_len, top_y - jib_thickness // 2, base_x + jib_len, top_y + jib_thickness // 2),
        fill=color,
    )
    # Operator cab (small box at the pivot).
    cab = max(6, height // 50)
    draw.rectangle((base_x - cab, top_y - cab, base_x + cab, top_y + cab // 2), fill=color)
    # A-frame triangle above the cab (the apex / pendant tower).
    apex_y = top_y - int(height * 0.18)
    draw.polygon(
        [(base_x - cab, top_y), (base_x + cab, top_y), (base_x, apex_y)],
        fill=color,
    )
    # Tension cables from apex to jib tip and counter-jib tip (thin lines).
    draw.line((base_x, apex_y, base_x + jib_len, top_y), fill=color, width=1)
    draw.line((base_x, apex_y, base_x - counter_len, top_y), fill=color, width=1)
    # A small hanging hook block under the jib.
    hook_x = base_x + int(jib_len * 0.78)
    hook_drop = int(height * 0.22)
    draw.line((hook_x, top_y, hook_x, top_y + hook_drop), fill=color, width=1)
    draw.rectangle(
        (hook_x - max(2, height // 200), top_y + hook_drop, hook_x + max(2, height // 200), top_y + hook_drop + 5),
        fill=color,
    )


def generate_construction_backdrop(width: int = 1024, height: int = 512, seed: int = 23) -> Image.Image:
    """A flat panoramic backdrop: hazy / dusty sky + city + construction silhouettes."""
    rng = np.random.default_rng(seed)

    # Sky gradient: pale dusty blue at top → warm haze near the horizon line.
    top = np.array([182, 188, 196], dtype=np.float64)
    horizon = np.array([214, 204, 188], dtype=np.float64)
    horizon_y = int(height * 0.66)
    img = np.empty((height, width, 3), dtype=np.float64)
    for y in range(height):
        if y < horizon_y:
            t = y / max(horizon_y - 1, 1)
            img[y] = top * (1.0 - t) + horizon * t
        else:
            # Below the horizon: a thin band of dust haze that fades to a
            # slightly cooler concrete-ground tone — the ground plane will
            # overdraw this anyway, but the band sells the haze.
            t = (y - horizon_y) / max(height - horizon_y - 1, 1)
            ground_tint = np.array([196, 192, 184], dtype=np.float64)
            img[y] = horizon * (1.0 - 0.5 * t) + ground_tint * (0.5 * t)

    # A diffuse sun glow up-left.
    sx, sy = int(width * 0.26), int(height * 0.22)
    yy, xx = np.mgrid[0:height, 0:width]
    sun_radius = height * 0.22
    glow = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * sun_radius**2))
    img += glow[..., None] * (np.array([252, 240, 218], dtype=np.float64) - img) * 0.55

    # Distant haze: very light rng noise across the sky to suggest atmospheric dust.
    haze = (rng.random((height, width, 1)) - 0.5) * 6.0
    img += haze

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    # ── Silhouettes ────────────────────────────────────────────────
    # Drawn directly on the PIL image so we can use ImageDraw's clean
    # rectangles / lines / polygons.  Two depth layers: far city (paler /
    # higher) and a nearer construction silhouette (darker / lower).
    draw = ImageDraw.Draw(pil)

    # Far city: pale silhouettes near the horizon.
    far_color = (118, 120, 128)
    far_top = horizon_y - int(height * 0.22)
    cursor = 0
    while cursor < width:
        w = int(rng.integers(width // 22, width // 11))
        h = int(rng.integers(height * 0.05, height * 0.18))
        x0 = cursor
        x1 = min(cursor + w, width)
        y1 = horizon_y
        y0 = max(far_top, horizon_y - h)
        draw.rectangle((x0, y0, x1, y1), fill=far_color)
        cursor = x1 + int(rng.integers(0, width // 80))

    # A tower crane on the far layer, off to the right.
    _draw_tower_crane(
        draw,
        base_x=int(width * 0.72),
        base_y=horizon_y,
        height=int(height * 0.55),
        jib_len=int(width * 0.16),
        counter_len=int(width * 0.05),
        color=(74, 78, 86),
    )

    # Nearer construction: a couple of taller blocks under scaffolding, darker.
    near_color = (62, 64, 72)
    grid_color = (92, 96, 104)
    near_blocks = [
        (int(width * 0.05), int(height * 0.46), int(width * 0.21), horizon_y),
        (int(width * 0.31), int(height * 0.36), int(width * 0.47), horizon_y),
        (int(width * 0.56), int(height * 0.50), int(width * 0.68), horizon_y),
    ]
    for x0, y0, x1, y1 in near_blocks:
        _draw_scaffolded_block(draw, x0, y0, x1, y1, near_color, cols=6, rows=10, grid_color=grid_color)
        # Crane on top of the tallest of these.
        if y0 == int(height * 0.36):
            _draw_tower_crane(
                draw,
                base_x=(x0 + x1) // 2,
                base_y=y0,
                height=int(height * 0.42),
                jib_len=int(width * 0.13),
                counter_len=int(width * 0.04),
                color=(46, 48, 54),
            )

    # A second smaller crane on the left, in front.
    _draw_tower_crane(
        draw,
        base_x=int(width * 0.12),
        base_y=horizon_y,
        height=int(height * 0.42),
        jib_len=int(width * 0.10),
        counter_len=int(width * 0.035),
        color=(54, 56, 62),
    )

    # A soft horizontal blur on the far layer to suggest atmospheric depth.
    haze_layer = pil.crop((0, 0, width, horizon_y)).filter(ImageFilter.GaussianBlur(radius=0.7))
    pil.paste(haze_layer, (0, 0))

    return pil


_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_DEFAULT_PATH = os.path.join(_ASSETS_DIR, "ground_texture.png")
_MARBLE_PATH = os.path.join(_ASSETS_DIR, "marble_texture.png")
_CONCRETE_PATH = os.path.join(_ASSETS_DIR, "concrete_texture.png")
_CONSTRUCTION_BACKDROP_PATH = os.path.join(_ASSETS_DIR, "construction_backdrop.png")


def default_texture_path() -> str:
    """Path to the bundled earthy ``ground_texture.png``."""
    return _DEFAULT_PATH


def marble_texture_path() -> str:
    """Path to the bundled ``marble_texture.png`` (dark veined marble)."""
    return _MARBLE_PATH


def concrete_texture_path() -> str:
    """Path to the bundled ``concrete_texture.png``."""
    return _CONCRETE_PATH


def construction_backdrop_path() -> str:
    """Path to the bundled ``construction_backdrop.png`` (flat sky backdrop)."""
    return _CONSTRUCTION_BACKDROP_PATH


if __name__ == "__main__":
    os.makedirs(_ASSETS_DIR, exist_ok=True)
    generate_ground_texture().save(_DEFAULT_PATH)
    print(f"wrote {_DEFAULT_PATH}")
    generate_marble_texture().save(_MARBLE_PATH)
    print(f"wrote {_MARBLE_PATH}")
    generate_concrete_texture().save(_CONCRETE_PATH)
    print(f"wrote {_CONCRETE_PATH}")
    generate_construction_backdrop().save(_CONSTRUCTION_BACKDROP_PATH)
    print(f"wrote {_CONSTRUCTION_BACKDROP_PATH}")
