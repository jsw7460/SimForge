"""Unified entity configuration for all simulators.

Provides a base :class:`EntityCfg` with common fields and per-simulator
subclasses (:class:`GenesisEntityCfg`, :class:`NewtonEntityCfg`,
:class:`MujocoEntityCfg`) for type-safe simulator-specific settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


# ---------------------------------------------------------------------------
# Actuator configuration (simulator-independent)
# ---------------------------------------------------------------------------

@dataclass
class ActuatorCfg:
    """Actuator definition for a group of joints.

    Attributes:
        target_names_expr: Regex patterns matching joint names that this
            actuator drives.
        control_type: ``"position"`` uses the simulator's built-in PD;
            ``"motor"`` bypasses it and accepts direct torques (required
            when using an explicit actuator model from
            :mod:`rlworld.rl.actuators`).
        stiffness: P-gain for position-mode actuators [N*m/rad].
        damping: D-gain for position-mode actuators [N*m*s/rad].
        effort_limit: Maximum torque [N*m].  None means no limit.
        armature: Reflected rotor inertia added to the joint [kg*m^2].
        frictionloss: Static friction at the joint [N*m].
    """

    target_names_expr: tuple[str, ...] = ()
    control_type: Literal["position", "motor"] = "position"
    stiffness: float = 0.0
    damping: float = 0.0
    effort_limit: float | None = None
    armature: float = 0.0
    frictionloss: float = 0.0


# ---------------------------------------------------------------------------
# Articulation configuration
# ---------------------------------------------------------------------------

@dataclass
class ArticulationCfg:
    """Articulation (joint drive) settings for an entity.

    Attributes:
        actuators: Tuple of :class:`ActuatorCfg` objects, one per
            actuator group.  Joints not matched by any actuator are
            passive.
        soft_joint_pos_limit_factor: Fraction of the physical joint
            limits used as "soft" limits (for observations / rewards).
    """

    actuators: tuple[ActuatorCfg, ...] = ()
    soft_joint_pos_limit_factor: float = 1.0


# ---------------------------------------------------------------------------
# Initial-state configuration
# ---------------------------------------------------------------------------

@dataclass
class InitialStateCfg:
    """Initial state of the entity when the environment resets.

    Attributes:
        pos: Root position in world frame [m].
        rot: Root orientation as (w, x, y, z) quaternion.
        lin_vel: Root linear velocity [m/s].
        ang_vel: Root angular velocity [rad/s].
        joint_pos: Regex-dict mapping joint name patterns to default
            positions [rad or m].
        joint_vel: Regex-dict mapping joint name patterns to default
            velocities [rad/s or m/s].
    """

    pos: tuple[float, float, float] = (0.0, 0.0, 0.5)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_pos: dict[str, float] = field(default_factory=dict)
    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})


# ---------------------------------------------------------------------------
# Base entity configuration
# ---------------------------------------------------------------------------

@dataclass
class EntityCfg:
    """Simulator-independent base entity configuration.

    Contains fields common to all simulators.  Use the subclasses
    :class:`GenesisEntityCfg`, :class:`NewtonEntityCfg`, or
    :class:`MujocoEntityCfg` for simulator-specific settings.
    """

    # -- Source ---------------------------------------------------------------
    urdf_path: str | None = None
    """Path to URDF file."""

    usd_path: str | None = None
    """Path to USD file (Newton only)."""

    # -- State ----------------------------------------------------------------
    init_state: InitialStateCfg = field(default_factory=InitialStateCfg)

    # -- Articulation ---------------------------------------------------------
    articulation: ArticulationCfg = field(default_factory=ArticulationCfg)

    floating: bool = False
    """True for mobile robots with a free-floating base."""

    enable_self_collisions: bool = False
    """Allow collisions between links of the same entity."""

    # -- Filtering ------------------------------------------------------------
    links_to_keep: list[str] = field(default_factory=list)
    """Links/joints to preserve when simplifying the model."""

    collapse_fixed_joints: bool = False
    """Merge bodies connected by fixed joints."""


# ---------------------------------------------------------------------------
# Per-simulator entity configurations
# ---------------------------------------------------------------------------

@dataclass
class GenesisEntityCfg(EntityCfg):
    """Genesis-specific entity configuration."""

    convexify: bool = False
    """Convex-decompose collision meshes."""

    visualize_contact: bool = False
    """Show contact forces in the viewer."""

    surface: Any = None
    """Genesis surface material (gs.surfaces.Surface)."""


@dataclass
class NewtonEntityCfg(EntityCfg):
    """Newton-specific entity configuration."""

    body_label_prefix: str | None = None
    """Prefix for body labels in the Newton model."""

    shape_cfg: Any = None
    """Newton contact material (newton.ModelBuilder.ShapeConfig)."""

    sites: dict[str, str] | None = None
    """Sensor site definitions: {site_name: body_name}."""

    contact_shapes: dict[str, Any] | None = None
    """Contact shape overrides."""

    mesh_approximation: str = "convex_hull"
    """Mesh simplification method for collision geometry."""

    ignore_inertial_definitions: bool = False
    """Ignore inertial properties from URDF."""


@dataclass
class MujocoEntityCfg(EntityCfg):
    """MuJoCo-specific entity configuration."""

    spec_fn: Callable | None = None
    """Factory callable returning an mujoco.MjSpec.  Required for MuJoCo
    since URDF must be converted to MjSpec."""

    entity_cfg: Any = None
    """Pre-built mjlab EntityCfg.  When provided, the actuator settings
    from :attr:`articulation` override its actuators but all other
    fields are taken from this object."""

    collisions: tuple = ()
    """mjlab CollisionCfg objects for contact customization."""

    cameras: tuple = ()
    """mjlab CameraCfg objects."""


# ---------------------------------------------------------------------------
# Ground plane configuration (convenience)
# ---------------------------------------------------------------------------

@dataclass
class GroundPlaneCfg:
    """Ground plane entity.

    Attributes:
        contact_stiffness: Normal contact stiffness [N/m].
        contact_damping: Normal contact damping [N*s/m].
        friction: Coulomb friction coefficient.
    """

    contact_stiffness: float = 5000.0
    contact_damping: float = 200.0
    friction: float = 1.0

    # Newton-specific ground options
    ground_kf: float = 100.0
    """Newton tangential contact stiffness."""

    ground_mu_rolling: float = 0.0
    """Newton rolling friction coefficient."""

    ground_mu_torsional: float = 0.0
    """Newton torsional friction coefficient."""
