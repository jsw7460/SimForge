"""A small palette of named ``ViserSceneConfig`` "looks".

Pick one by name (e.g. in an eval script: ``--look studio``) instead of
hand-writing a ``ViserSceneConfig`` every time.  Each entry is a complete
scene config — tweak / add freely (it's just data).

    from rlworld.rl.vis.viser import get_look, list_looks
    cfg = get_look("ceramic_white")          # a fresh, mutable copy
    print(list_looks())
"""

from __future__ import annotations

from dataclasses import replace

from .scene_config import ViserSceneConfig

# name -> ViserSceneConfig.  Keep these as plain instances; ``get_look``
# hands out a fresh copy so callers can't mutate the shared template.
VISER_LOOKS: dict[str, ViserSceneConfig] = {
    # The package default: near-black metallic robot, earthy textured ground,
    # outdoor "sun" + sky.
    "default": ViserSceneConfig(),
    "metal_dark": ViserSceneConfig(),
    # Polished light metal — chrome / brushed-aluminium showpiece.
    "metal_polished": ViserSceneConfig(robot_color=(190, 195, 200), robot_metalness=1.0, robot_roughness=0.12),
    # Glossy black plastic / car paint.
    "plastic_glossy": ViserSceneConfig(robot_color=(18, 18, 20), robot_metalness=0.0, robot_roughness=0.18),
    # Matte dark-grey rubber / hard-anodised.
    "rubber_matte": ViserSceneConfig(robot_color=(40, 42, 46), robot_metalness=0.0, robot_roughness=0.95),
    # Satin / eggshell — soft sheen, mid-grey.
    "satin": ViserSceneConfig(robot_color=(120, 124, 130), robot_metalness=0.05, robot_roughness=0.5),
    # Carbon-fibre-ish dark with a faint sheen.
    "carbon": ViserSceneConfig(robot_color=(34, 38, 42), robot_metalness=0.25, robot_roughness=0.4),
    # Glossy white ceramic on a light, plain ground + pale sky.
    "ceramic_white": ViserSceneConfig(
        robot_color=(232, 234, 238),
        robot_metalness=0.0,
        robot_roughness=0.35,
        ground_texture=None,
        ground_kind="plane",
        ground_color=(238, 238, 240),
        sky_color=(205, 216, 232),
    ),
    # Keep the simulator's own per-link mesh colors / textures (the unitree
    # robots' black/grey parts, etc.) on the earthy textured ground.
    "sim_native": ViserSceneConfig(robot_color=None),
    # Clean "product render" look: plain near-white ground, no sky, soft fill,
    # no shadows, neutral matte robot.
    "studio": ViserSceneConfig(
        robot_color=(150, 152, 156),
        robot_metalness=0.0,
        robot_roughness=0.6,
        ground_texture=None,
        ground_kind="plane",
        ground_color=(244, 244, 246),
        sky_background=False,
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
        lighting=False,
        sky_background=False,
        cast_shadow=False,
        receive_shadow=False,
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
