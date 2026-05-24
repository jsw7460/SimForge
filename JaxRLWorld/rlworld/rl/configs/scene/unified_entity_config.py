"""Unified entity configuration for all simulators.

Provides a base :class:`EntityCfg` with common fields and per-simulator
subclasses (:class:`GenesisEntityCfg`, :class:`NewtonEntityCfg`,
:class:`MujocoEntityCfg`) for type-safe simulator-specific settings.

Actuator types are defined in :mod:`rlworld.rl.actuators.actuator_cfg`.
The actuator class determines the control mode:

- :class:`~rlworld.rl.actuators.ImplicitActuatorCfg` → simulator PD
- Any other actuator → explicit torque (motor/force mode)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from rlworld.rl.actuators.actuator_cfg import ActuatorBaseCfg


# ---------------------------------------------------------------------------
# Articulation configuration
# ---------------------------------------------------------------------------


@dataclass
class ArticulationCfg:
    """Articulation (joint drive) settings for an entity.

    Attributes:
        actuators: Tuple of actuator config objects (from
            :mod:`rlworld.rl.actuators`), one per actuator group.
            The actuator **type** determines the control mode:
            :class:`~rlworld.rl.actuators.ImplicitActuatorCfg` uses
            the simulator's built-in PD; all others compute torques
            explicitly.
        soft_joint_pos_limit_factor: Fraction of the physical joint
            limits used as "soft" limits (for observations / rewards).
    """

    actuators: tuple[ActuatorBaseCfg, ...] = ()
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

    mjcf_path: str | None = None
    """Path to MJCF (MuJoCo XML) file. Newton loads this via
    ``ModelBuilder.add_mjcf`` and preserves ``<site>`` declarations as
    first-class reference points on the shape arrays. Takes precedence
    over ``urdf_path`` when both are set (Newton backend)."""

    # -- State ----------------------------------------------------------------
    init_state: InitialStateCfg = field(default_factory=InitialStateCfg)

    # -- Articulation ---------------------------------------------------------
    articulation: ArticulationCfg = field(default_factory=ArticulationCfg)

    floating: bool = False
    """True for mobile robots with a free-floating base."""

    enable_self_collisions: bool = True
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

    spec_fn: Callable | str | None = None
    """Factory callable returning an ``mujoco.MjSpec``, or a string reference
    (``"module:func"``).  Automatically serialized to string by ``recursive_to_dict()``."""

    collisions: tuple = ()
    """mjlab CollisionCfg objects for contact customization."""

    cameras: tuple = ()
    """mjlab CameraCfg objects."""
