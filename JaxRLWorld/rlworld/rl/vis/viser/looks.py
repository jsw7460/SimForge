"""A small palette of named ``ViserSceneConfig`` "looks".

Pick one by name (e.g. in an eval script: ``--look studio``) instead of
hand-writing a ``ViserSceneConfig`` every time.  Each entry is a complete
scene config — tweak / add freely (it's just data).

    from rlworld.rl.vis.viser import get_look, list_looks
    cfg = get_look("polished")              # a fresh, mutable copy
    print(list_looks())
"""

from __future__ import annotations

from dataclasses import replace

from .scene_config import ViserSceneConfig

# Glossy dark robot on a glass-polished light checkerboard + blue sky.
# Shared by the "ceramic_white" and "default" entries (``get_look`` hands
# out fresh copies, so sharing the instance is safe).
_CERAMIC_WHITE = ViserSceneConfig(
    robot_color=(18, 18, 20),
    robot_metalness=0.0,
    robot_roughness=0.18,
    ground_texture="checker",
    ground_texture_tiles=25.0,
    ground_metalness=0.15,
    ground_roughness=0.16,
    # The studio HDRI feeds the floor reflections, but its softbox
    # panels mirror as glaring white blobs on a near-mirror floor:
    # dim the map and lift the floor roughness a touch so the
    # panels spread into a soft sheen instead of hot spots.
    env_map="studio",
    env_map_intensity=0.45,
    env_map_as_background=False,
    sky_background=True,
    sky_kind="gradient",
    sky_color=(120, 172, 230),
    sky_horizon_color=(210, 226, 242),
    # No sun: the directional light's specular glint on the glass
    # floor (and the sky's sun-glow disc) read as a glaring hot spot;
    # ambient + hemisphere + the studio HDRI carry the lighting.
    sun_intensity=0.0,
    sky_sun_glow=False,
    ambient_intensity=0.55,
    hemisphere_intensity=0.5,
)

# name -> ViserSceneConfig.  Keep these as plain instances; ``get_look``
# hands out a fresh copy so callers can't mutate the shared template.
VISER_LOOKS: dict[str, ViserSceneConfig] = {
    # The package default — ceramic_white (also the fallback when no
    # ``viser_scene`` / ``--look`` is given; see ``ViserScene``).
    "default": _CERAMIC_WHITE,
    # Dark veined marble under glass-polished specular + carbon-finish
    # robot, lit by the "studio" HDRI (the former package default).
    "polished": ViserSceneConfig(),
    "carbon": ViserSceneConfig(),
    # Dark metal finish on the same polished marble (no carbon tint).
    "metal_dark": ViserSceneConfig(
        robot_color=(35, 35, 35),
        robot_metalness=0.9,
        robot_roughness=0.3,
    ),
    # Polished light metal — chrome / brushed-aluminium showpiece.
    "metal_polished": ViserSceneConfig(
        robot_color=(190, 195, 200),
        robot_metalness=1.0,
        robot_roughness=0.12,
    ),
    # Glossy black plastic / car paint.
    "plastic_glossy": ViserSceneConfig(
        robot_color=(18, 18, 20),
        robot_metalness=0.0,
        robot_roughness=0.18,
    ),
    # Matte dark-grey rubber / hard-anodised.
    "rubber_matte": ViserSceneConfig(
        robot_color=(40, 42, 46),
        robot_metalness=0.0,
        robot_roughness=0.95,
    ),
    # Satin / eggshell — soft sheen, mid-grey.
    "satin": ViserSceneConfig(robot_color=(120, 124, 130), robot_metalness=0.05, robot_roughness=0.5),
    # Industrial warehouse: dark concrete-feel ground + the "warehouse" HDRI.
    "warehouse": ViserSceneConfig(
        ground_color=(46, 48, 52),
        ground_metalness=0.10,
        ground_roughness=0.45,
        env_map="warehouse",
        env_map_blurriness=0.5,
    ),
    # Warm sunset glow — dramatic orange / pink rim light from the HDRI.
    "sunset": ViserSceneConfig(
        ground_color=(30, 28, 32),
        ground_roughness=0.20,
        env_map="sunset",
        env_map_blurriness=0.3,
        sun_color=(255, 200, 150),
        sun_intensity=1.3,
    ),
    # The previous outdoor look: earthy textured ground, blue sky + sun.
    # Kept here for callers who liked the natural-terrain vibe.
    "earthy": ViserSceneConfig(
        ground_texture="default",
        ground_texture_tiles=25.0,
        ground_metalness=0.0,
        ground_roughness=0.92,
        env_map=None,
        sky_background=True,
        sky_kind="gradient",
        sky_color=(138, 184, 235),
        sky_horizon_color=(226, 234, 240),
        ambient_intensity=0.45,
        hemisphere_intensity=0.55,
        hemisphere_ground_color=(150, 140, 120),
        sun_intensity=1.7,
    ),
    # The grid gives the eye a motion reference, the saturated sky
    # separates ground from background, and the "checker" TEXTURE (not
    # the face-colored checkerboard mesh) lets the ground carry a
    # glossy PBR material so it reflects.  See ``_CERAMIC_WHITE`` above.
    "ceramic_white": _CERAMIC_WHITE,
    # The MuJoCo / dm_control house style (as seen in OGBench renders):
    # dark-navy two-tone grid with light cross lines under a glassy
    # polish, a starfield night sky, and sim-native robot colors.  The
    # "night" HDRI feeds the floor reflections only (dark, so no glare
    # blobs) — ``env_map_as_background=False`` keeps the starfield as
    # the visible backdrop.
    "mujoco": ViserSceneConfig(
        robot_color=None,
        ground_texture="navy_grid",
        ground_texture_tiles=25.0,
        # Dielectric gloss: metalness must stay 0 — a metallic material
        # trades its diffuse color for reflections.  And the gloss must
        # stay moderate: a glossy floor SHOWS its environment, so over a
        # near-black night sky a mirror finish renders black no matter
        # the base color (Fresnel boosts reflection at the floor's
        # grazing view angles).  Satin roughness over the "city" HDRI
        # (night lights) keeps the navy readable with lit reflections.
        ground_metalness=0.0,
        ground_roughness=0.3,
        env_map="city",
        env_map_intensity=0.55,
        env_map_as_background=False,
        sky_background=True,
        sky_kind="starfield",
        sky_sun_glow=False,
        sun_intensity=0.9,
        ambient_intensity=0.4,
        hemisphere_intensity=0.35,
    ),
    # Keep the simulator's own per-link mesh colors / textures (the unitree
    # robots' black/grey parts, etc.) on the polished slate.
    "sim_native": ViserSceneConfig(robot_color=None),
    # Clean "product render" look: plain near-white ground, no busy sky,
    # soft fill, no harsh shadows, neutral matte robot.
    "studio": ViserSceneConfig(
        robot_color=(150, 152, 156),
        robot_metalness=0.0,
        robot_roughness=0.6,
        ground_texture=None,
        ground_kind="plane",
        ground_color=(244, 244, 246),
        ground_roughness=0.5,
        env_map="lobby",
        env_map_blurriness=0.6,
        cast_shadow=False,
        receive_shadow=False,
        ambient_intensity=0.7,
        hemisphere_intensity=0.7,
        sun_intensity=1.0,
    ),
    # Bare-bones diagram look: subtle checkerboard, viser's default lights, no
    # sky, no shadows, sim-native robot colors.
    "minimal": ViserSceneConfig(
        robot_color=None,
        ground_texture=None,
        ground_kind="checkerboard",
        ground_color=(245, 245, 245),
        ground_color_alt=(225, 225, 225),
        env_map=None,
        lighting=False,
        sky_background=False,
        cast_shadow=False,
        receive_shadow=False,
    ),
    # Construction-site vibe: poured-concrete slab + hazy panorama with tower
    # cranes and scaffolded buildings, overcast/dusty lighting, safety-orange
    # robot.  Backdrop is a flat image — it doesn't rotate with the camera, so
    # the cranes always sit roughly on the horizon.  HDRI off so it doesn't
    # override the painted backdrop.
    "construction": ViserSceneConfig(
        robot_color=(228, 120, 30),
        robot_metalness=0.35,
        robot_roughness=0.55,
        ground_texture="concrete",
        ground_texture_tiles=20.0,
        ground_metalness=0.0,
        ground_roughness=0.95,
        env_map=None,
        sky_background=True,
        sky_kind="construction",
        sky_color=(180, 184, 192),
        sky_horizon_color=(214, 204, 188),
        sky_sun_glow=False,
        sun_direction=(-0.4, -0.3, -0.86),
        sun_color=(248, 240, 220),
        sun_intensity=1.05,
        ambient_intensity=0.65,
        hemisphere_intensity=0.5,
        hemisphere_ground_color=(150, 145, 138),
    ),
}


def list_looks() -> list[str]:
    """Names of the available looks (sorted)."""
    return sorted(VISER_LOOKS)


def get_look(name: str) -> ViserSceneConfig:
    """A fresh (mutable) copy of the named look. Raises ``KeyError`` if unknown."""
    if name not in VISER_LOOKS:
        raise KeyError(f"Unknown viser look {name!r}. Available: {list_looks()}")
    return replace(VISER_LOOKS[name])
