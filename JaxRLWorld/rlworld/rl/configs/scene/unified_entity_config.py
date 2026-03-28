"""Unified entity configuration for all simulators.

Provides a single EntityCfg that Genesis, Newton, and MuJoCo scene
managers all consume.  Simulator-specific details are handled by each
scene manager's adapter logic, not by the config itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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
# Unified entity configuration
# ---------------------------------------------------------------------------

@dataclass
class EntityCfg:
    """Simulator-independent entity configuration.

    Each scene manager interprets this config for its own simulator.
    Simulator-specific options that have no cross-simulator equivalent
    go into the ``genesis_options`` / ``newton_options`` /
    ``mujoco_options`` dictionaries.

    Example::

        EntityCfg(
            urdf_path="assets/go2/go2.urdf",
            init_state=InitialStateCfg(
                pos=(0, 0, 0.34),
                joint_pos={".*thigh": 0.9, ".*calf": -1.8},
            ),
            floating=True,
            articulation=ArticulationCfg(
                actuators=(
                    ActuatorCfg(
                        target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
                        stiffness=17.6, damping=5.6,
                        effort_limit=23.7, armature=0.0045,
                    ),
                    ActuatorCfg(
                        target_names_expr=(".*_calf_joint",),
                        stiffness=65.0, damping=20.7,
                        effort_limit=45.43, armature=0.016,
                    ),
                ),
            ),
        )
    """

    # -- Source ---------------------------------------------------------------
    urdf_path: str | None = None
    """Path to URDF file (Genesis and Newton)."""

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
    """Regex patterns for links to preserve when simplifying the model.
    Genesis uses this for foot-link retention; Newton uses joints_to_keep."""

    collapse_fixed_joints: bool = False
    """Merge bodies connected by fixed joints (Newton/MuJoCo)."""

    # -- Simulator-specific overrides -----------------------------------------
    genesis_options: dict[str, Any] = field(default_factory=dict)
    """Genesis-specific options passed to the scene manager.

    Commonly used keys:
        - ``convexify`` (bool): convex-decompose collision meshes.
        - ``surface`` (gs.surfaces.Surface): surface material.
        - ``visualize_contact`` (bool): show contact forces.
    """

    newton_options: dict[str, Any] = field(default_factory=dict)
    """Newton-specific options passed to the scene manager.

    Commonly used keys:
        - ``body_label_prefix`` (str): prefix for body labels.
        - ``shape_cfg`` (newton.ModelBuilder.ShapeConfig): contact material.
        - ``sites`` (dict[str, str]): sensor site definitions.
        - ``contact_shapes`` (dict): contact shape overrides.
        - ``mesh_approximation`` (str): mesh simplification method.
    """

    mujoco_options: dict[str, Any] = field(default_factory=dict)
    """MuJoCo-specific options passed to the scene manager.

    Commonly used keys:
        - ``collisions`` (tuple): mjlab CollisionCfg objects.
        - ``cameras`` (tuple): mjlab CameraCfg objects.
    """


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

    newton_options: dict[str, Any] = field(default_factory=dict)
    """Extra Newton ShapeConfig fields (kf, mu_rolling, etc.)."""
