from dataclasses import dataclass, field
from typing import Literal

import newton
import warp as wp

RemeshingMethod = Literal["ftetwild", "alphashape", "quadratic", "convex_hull"]


@dataclass
class NewtonEntityConfig:
    """Configuration for a Newton entity (robot, object, or ground plane).

    Example (robot):
        NewtonEntityConfig(
            entity_name="robot",
            urdf_path="/path/to/robot.urdf",
            floating=True,
            joint_cfg=newton.ModelBuilder.JointDofConfig(...),
            shape_cfg=newton.ModelBuilder.ShapeConfig(...),
        )

    Example (ground plane):
        NewtonEntityConfig(
            entity_name="ground",
            entity_type="ground_plane",
            shape_cfg=newton.ModelBuilder.ShapeConfig(
                ke=5.0e3,
                kd=200.0,
                mu=1.0,
            ),
        )
    """

    # Identity
    entity_name: str

    body_label_prefix: str | None = None
    entity_type: Literal["urdf", "usd", "ground_plane"] = "urdf"

    # URDF specific
    urdf_path: str | None = None
    ignore_inertial_definitions: bool = False

    # USD specific
    usd_path: str | None = None
    hide_collision_shapes: bool = True
    skip_mesh_approximation: bool = True
    mesh_approximation: Literal["coacd", "vhacd", "bounding_sphere", "bounding_box"] | RemeshingMethod = "convex_hull"

    # Transform
    transform: wp.transform = field(default_factory=lambda: wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()))

    # Articulation settings
    floating: bool = False
    enable_self_collisions: bool = False
    collapse_fixed_joints: bool = False
    joints_to_keep: list[str] = field(default_factory=list)

    # Joint config
    # If joint_target_ke(kd)_map is given, pd-gain of joint_cfg will be ignored
    joint_cfg: newton.ModelBuilder.JointDofConfig | None = None

    # Shape config
    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None

    # Joint params (regex pattern -> value)
    joint_target_ke_map: dict[str, float] | None = None  # {"FR.*": 20.0, ".*_hip_joint": 30.0}
    joint_target_kd_map: dict[str, float] | None = None  # {"FR.*": 0.5}
    joint_armature_map: dict[str, float] | None = None  # {"FR.*": 0.01, ".*_hip_joint": 0.02}

    # Sites for sensors
    sites: dict[str, str] | None = None

    # Contact shapes
    contact_shapes: dict[str, tuple[str, tuple[float, float, float]]] | None = None


@dataclass
class NewtonGroundPlaneConfig:
    """Ground plane configuration."""

    shape_cfg: newton.ModelBuilder.ShapeConfig | None = None


@dataclass
class NewtonBoxConfig:
    """Configuration for a box primitive in Newton."""

    entity_name: str
    hx: float = 0.5  # Half-extent in x
    hy: float = 0.5  # Half-extent in y
    hz: float = 0.5  # Half-extent in z
    transform: wp.transform = field(default_factory=lambda: wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()))
    density: float = 1000.0
    ke: float = 1.0e4
    kd: float = 1.0e2
    kf: float = 1.0e2
    mu: float = 1.0
