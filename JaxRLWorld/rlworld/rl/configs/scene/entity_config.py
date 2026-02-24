from dataclasses import dataclass, field

import genesis as gs


@dataclass
class GenesisSceneInitConfig:
    sim_options: gs.options.SimOptions = field(default_factory=gs.options.SimOptions)
    viewer_options: gs.options.ViewerOptions = field(default_factory=gs.options.ViewerOptions)
    vis_options: gs.options.VisOptions = field(default_factory=gs.options.VisOptions)
    rigid_options: gs.options.RigidOptions = field(default_factory=gs.options.RigidOptions)
    show_viewer: bool = False


@dataclass
class EntityConfig:
    """Configuration for a reward term."""

    entity_name: str
    morph: gs.morphs.Morph
    surface: gs.surfaces.Surface | None = None
    visualize_contact: bool = False

    p_gain: dict[str, float] | None = None
    d_gain: dict[str, float] | None = None
    armature: dict[str, float] | None = None