from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Literal

import newton
import numpy as np
import torch
import warp as wp
from newton import ShapeFlags
from newton.selection import ArticulationView
from newton.sensors import SensorContact

from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg
from rlworld.rl.configs.newton_config_classes import SolverMuJoCoCfg
from rlworld.rl.configs.scene.newton_entity_config import (
    NewtonEntityConfig,
    NewtonGroundPlaneConfig,
)
from rlworld.rl.configs.scene.unified_entity_config import (
    EntityCfg,
    GroundPlaneCfg,
    NewtonEntityCfg,
)
from rlworld.rl.configs.sensors import ContactSensorCfg
from rlworld.rl.configs.sensors.newton_sensor_config import (
    NewtonContactSensorConfig,
    NewtonFrameTransformSensorConfig,
    NewtonIMUSensorConfig,
    NewtonSensorConfig,
)
from rlworld.rl.envs.indexing import ArticulationIndexing
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.managers.common.scene_helpers import build_kinematic_trees
from rlworld.rl.envs.managers.newton.contact_sensor import NewtonContactSensor
from rlworld.rl.envs.utils.newton.label import as_leaf_globs, leaf_name
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def apply_joint_params_by_pattern(
    builder,
    ke_map: Dict[str, float] | None = None,
    kd_map: Dict[str, float] | None = None,
    armature_map: Dict[str, float] | None = None,
    effort_limit_map: Dict[str, float] | None = None,
    friction_map: Dict[str, float] | None = None,
) -> None:
    """Apply joint parameters (target gains, armature, effort limit) using
    regex pattern matching.

    Uses ``re.fullmatch`` rather than ``re.match`` so a pattern like
    ``T1/.*Waist`` only fires on joint labels that *end* with
    ``Waist``, not any label whose XPath *contains* a ``Waist`` body
    segment. This matters for MJCF-loaded entities whose joint labels
    are hierarchical (e.g. ``T1/worldbody/Trunk/Waist/Hip_Pitch_Left/
    Left_Hip_Pitch``); with ``re.match`` a bare ``.*Waist`` pattern
    would silently claim every leg joint under the Waist subtree.
    Flat URDF labels behave identically under both functions because
    existing patterns already end in a specific suffix.

    ``effort_limit_map`` overrides ``builder.joint_effort_limit`` per DOF;
    ``SolverMuJoCo`` later turns this into ``joint.actfrcrange`` on each
    hinge (always ``actfrclimited=True``). Use this to relax the tight
    XML-declared ``actuatorfrcrange`` when the scene-file values disagree
    with the motor spec that training actually assumes (e.g. menagerie
    T1's 15/20 Nm ankle vs booster_t1's 50 Nm).
    """
    if ke_map is None and kd_map is None and armature_map is None and effort_limit_map is None and friction_map is None:
        return

    ke_map = ke_map or {}
    kd_map = kd_map or {}
    armature_map = armature_map or {}
    effort_limit_map = effort_limit_map or {}
    friction_map = friction_map or {}
    num_joints = len(builder.joint_label)

    for joint_idx, joint_label_raw in enumerate(builder.joint_label):
        # Canonicalize to bare leaf so regex patterns in ke_map etc. can
        # use bare joint names (``waist_yaw_joint``, ``.*_hip_pitch_joint``)
        # regardless of URDF-flat vs MJCF-XPath layout.
        joint_name = leaf_name(joint_label_raw)
        if joint_idx < num_joints - 1:
            dof_count = builder.joint_qd_start[joint_idx + 1] - builder.joint_qd_start[joint_idx]
        else:
            dof_count = builder.joint_dof_count - builder.joint_qd_start[joint_idx]

        if dof_count == 0:
            continue

        dof_start = builder.joint_qd_start[joint_idx]

        # Apply ke
        for pattern, value in ke_map.items():
            if re.fullmatch(pattern, joint_name):
                for d in range(dof_count):
                    builder.joint_target_ke[dof_start + d] = value
                break

        # Apply kd
        for pattern, value in kd_map.items():
            if re.fullmatch(pattern, joint_name):
                for d in range(dof_count):
                    builder.joint_target_kd[dof_start + d] = value
                break

        # Apply armature
        for pattern, value in armature_map.items():
            if re.fullmatch(pattern, joint_name):
                for d in range(dof_count):
                    builder.joint_armature[dof_start + d] = value

        # Apply effort limit (overrides XML-declared actuatorfrcrange)
        for pattern, value in effort_limit_map.items():
            if re.fullmatch(pattern, joint_name):
                for d in range(dof_count):
                    builder.joint_effort_limit[dof_start + d] = value
                break

        # Apply joint frictionloss (MuJoCo dof_frictionloss equivalent)
        for pattern, value in friction_map.items():
            if re.fullmatch(pattern, joint_name):
                for d in range(dof_count):
                    builder.joint_friction[dof_start + d] = value
                break


def _state_assign_full(
    dst: newton.State,
    src: newton.State,
    namespaces: tuple[str, ...] = ("mujoco",),
) -> None:
    r"""Assign ``src`` into ``dst`` including attribute-namespace arrays.

    Newton's :meth:`State.assign` iterates ``self.__dict__`` and copies
    top-level ``wp.array`` attributes, but descendants inside
    ``AttributeNamespace`` objects (e.g. ``state.mujoco.qfrc_actuator``)
    are skipped because the namespace itself is not a ``wp.array`` —
    the outer loop treats it as a non-array and ``continue``\s. This
    helper performs the standard ``dst.assign(src)`` then manually
    descends into each listed namespace and copies its ``wp.array``
    children.

    Callers rely on this when the ``need_state_copy`` branch of a
    substep loop fires (odd substeps under CUDA graph capture) so
    consumers reading ``state.mujoco.qfrc_actuator`` via
    :attr:`NewtonRobotData.applied_torque` don't see stale zeros.
    Remove (replace with plain ``dst.assign(src)``) once Newton's
    ``State.assign`` recurses into namespaces upstream.
    """
    dst.assign(src)
    for ns in namespaces:
        ns_dst = getattr(dst, ns, None)
        ns_src = getattr(src, ns, None)
        if ns_dst is None or ns_src is None:
            continue
        for attr in vars(ns_src):
            s = getattr(ns_src, attr, None)
            d = getattr(ns_dst, attr, None)
            if isinstance(s, wp.array) and isinstance(d, wp.array):
                d.assign(s)


def _force_collision_shape_priority(builder: newton.ModelBuilder, priority: int = 1) -> None:
    """Set ``geom_priority`` on every COLLIDE_SHAPES shape in this builder.

    Works around a Newton MJCF parser bug where XML ``priority`` attributes
    on ``<geom>`` elements are lost for most shapes when visuals are also
    parsed. We bypass the parser by mutating the custom_attribute values
    dict directly; Newton exposes no public per-shape setter for
    post-load custom attributes. Visual-only shapes
    (``COLLIDE_SHAPES`` flag cleared) are skipped because they don't
    participate in contact pairs.
    """
    attr = builder.custom_attributes.get("mujoco:geom_priority")
    if attr is None:
        return
    if attr.values is None:
        attr.values = {}
    for shape_idx in range(len(builder.shape_flags)):
        if builder.shape_flags[shape_idx] & ShapeFlags.COLLIDE_SHAPES:
            attr.values[shape_idx] = priority


@dataclass
class NewtonSceneManagerConfig:
    """Configuration for Newton scene management.

    This config defines the simulation parameters and lists of entities/sensors
    to be created in the scene.

    Example:
        config = NewtonSceneManagerConfig(
            num_worlds=4096,
            entities=[
                NewtonEntityConfig(
                    entity_name="robot",
                    urdf_path="/path/to/robot.urdf",
                    floating=True,
                    sites={"base_imu": "base"},  # Create IMU site on base body
                ),
            ],
            sensors=[
                NewtonIMUSensorConfig(
                    sensor_name="imu",
                    entity_name="robot",
                    site_names=["base_imu"],
                ),
            ],
            add_ground=True,
        )
    """

    num_worlds: int

    # Entity and sensor configurations
    entities: list[NewtonEntityConfig] | dict[str, EntityCfg | GroundPlaneCfg] = field(default_factory=dict)
    sensors: list[NewtonSensorConfig] | None = None
    # Simulator-agnostic contact sensors (ContactSensorCfg). Handled
    # separately from ``sensors`` (which only holds NewtonSensorConfig
    # subclasses); each becomes a ``NewtonContactSensor`` wrapper around a
    # native ``SensorContact``, with optional substep history.
    contact_sensors: list[ContactSensorCfg] | None = None

    # Ground plane
    add_ground: bool = True
    ground_config: NewtonGroundPlaneConfig | None = None

    # Simulation parameters
    dt: float = 1.0 / 100.0
    substeps: int = 10
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    env_spacing: tuple[float, float, float] = (2.0, 2.0, 0.0)

    # Solver
    solver_type: Literal["xpbd", "mujoco"] = "mujoco"  # "xpbd" or "mujoco"
    solver_cfg: SolverMuJoCoCfg = field(default_factory=SolverMuJoCoCfg)


class NewtonSceneManager(BaseManager):
    """Manages Newton scene creation and simulation.

    This manager handles:
    1. Creating entities (robots, objects) from configuration
    2. Setting up sensors (IMU, Contact, etc.)
    3. Building and stepping the simulation

    The scene is built in two phases:
    1. register_entities(): Add all entities to the builder
    2. build_scene(): Finalize and create solver/state

    Example:
        scene_manager = NewtonSceneManager(env, config)
        scene_manager.register_entities()
        scene_manager.build_scene()

        # In simulation loop:
        scene_manager.step()
    """

    def __init__(self, env: World, config: NewtonSceneManagerConfig):
        super().__init__(env)
        self.config = config

        # Newton objects
        self.model: newton.Model | None = None
        self.solver: Any = None
        self.state_0: newton.State | None = None
        self.state_1: newton.State | None = None
        self.control: newton.Control | None = None
        self.contacts: newton.Contacts | None = None
        self.collision_pipeline: Any = None

        # ArticulationView per entity (created in build_scene)
        self.articulation_views: dict[str, ArticulationView] = {}

        # Entity tracking
        self.entities: dict[str, Any] = {}  # entity_name -> entity info
        self._entity_builders: dict[str, newton.ModelBuilder] = {}  # Temporary during build
        self._body_name_to_idx: dict[str, dict[str, int]] = defaultdict(dict)  # entity_name -> {body_name: body_idx}

        # Sensor tracking
        self.sensors: dict[str, Any] = {}  # sensor_name -> sensor object
        # ContactSensorCfg-backed wrappers (subset of ``self.sensors`` —
        # the wrapper objects are also stored under their ``cfg.name`` in
        # ``self.sensors`` so generic sensor iteration sees them).
        self._contact_sensor_wrappers: dict[str, Any] = {}

        # Kinematic trees (for observation functions)
        self.trees: dict[str, Any] = {}

        # Internal
        self.substep_dt = config.dt / config.substeps

        # CUDA graph populated by ``capture()`` once the physics loop
        # is captured. Kept as an attribute from init so ``_step`` can
        # reference ``self.graph`` during the initial capture pass
        # without hitting AttributeError. ``use_cuda_graph`` mirrors
        # Newton's example convention (see
        # ``newton/examples/robot/example_robot_policy.py``) —
        # JaxRLWorld's NewtonSceneManager always captures in
        # ``_post_setup``, so the flag is effectively always True, but
        # exposing it lets ``_step``'s ``need_state_copy`` guard match
        # the Newton reference implementation line-for-line.
        self.use_cuda_graph = True
        self.graph = None

    @property
    def robot(self) -> Any:
        """For compatibility - returns model in Newton."""
        return self.model

    @property
    def state(self) -> newton.State:
        """Current state (state_0)."""
        return self.state_0

    def find_body_names(self, body_names: list[str], entity_name: str = "robot") -> list[str]:
        """Resolve regex body-name patterns to concrete body names.

        Cross-sim signature matches Genesis and mjlab. Newton currently
        supports a single robot, so ``entity_name`` is accepted for API
        symmetry but the lookup is always against the model-wide body
        label list.
        """
        # Source body names from ArticulationView — bare leaf names,
        # shared across worlds. Avoids per-Newton-loader XPath quirks.
        view = self.articulation_views.get(entity_name, self.articulation_views.get("robot"))
        if view is None:
            raise ValueError(f"No ArticulationView for entity {entity_name!r}; did build_scene() run?")
        _, names = string_utils.resolve_matching_names(body_names, list(view.link_names), preserve_order=True)
        return names

    def _prefix_names(
        self, entity_name: str, names: str | list[str] | list[int] | None
    ) -> str | list[str] | list[int] | None:
        if names is None:
            return None

        # Get actual prefix from entity's robot config
        entity_config = self.entities[entity_name]["config"]
        prefix = getattr(entity_config, "body_label_prefix", None)
        if prefix is None:
            return names

        if isinstance(names, list):
            if all(isinstance(n, int) for n in names):
                return names
            return [f"{prefix}/{n}" if "/" not in n else n for n in names]
        if isinstance(names, str):
            return f"{prefix}/{names}" if "/" not in names else names
        return names

    def register_entities(self) -> None:
        """Register all entities defined in config."""
        for entity_name, cfg in self.config.entities.items():
            self._register_entity(entity_name, cfg)

    def _register_entity(self, entity_name: str, cfg: EntityCfg | NewtonEntityCfg | GroundPlaneCfg) -> None:
        """Register a single entity from its unified config."""
        if entity_name in self.entities:
            raise ValueError(f"Entity '{entity_name}' already registered")

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        if isinstance(cfg, GroundPlaneCfg):
            shape_cfg = newton.ModelBuilder.ShapeConfig(
                ke=cfg.contact_stiffness,
                kd=cfg.contact_damping,
                mu=cfg.friction,
                kf=cfg.ground_kf,
                mu_rolling=cfg.ground_mu_rolling,
                mu_torsional=cfg.ground_mu_torsional,
            )
            builder.add_ground_plane(cfg=shape_cfg)
        elif cfg.usd_path:
            self._load_usd_entity(builder, cfg)
        elif cfg.mjcf_path:
            self._load_mjcf_entity(builder, cfg)
        elif cfg.urdf_path:
            self._load_urdf_entity(builder, cfg)
        else:
            raise ValueError(f"Entity '{entity_name}' has no mjcf_path, urdf_path, or usd_path")

        self._entity_builders[entity_name] = builder
        self.entities[entity_name] = {
            "config": cfg,
            "builder": builder,
            "shape_count": len(builder.shape_label),
        }

    def _load_urdf_entity(self, builder: newton.ModelBuilder, cfg: EntityCfg | NewtonEntityCfg) -> None:
        """Load URDF entity from unified config."""
        shape_cfg = getattr(cfg, "shape_cfg", None)
        if shape_cfg is not None:
            builder.default_shape_cfg = shape_cfg

        xform = wp.transform(wp.vec3(*cfg.init_state.pos), wp.quat(*cfg.init_state.rot))
        ignore_inertial = getattr(cfg, "ignore_inertial_definitions", False)
        builder.add_urdf(
            cfg.urdf_path,
            xform=xform,
            floating=cfg.floating,
            enable_self_collisions=cfg.enable_self_collisions,
            collapse_fixed_joints=False,
            ignore_inertial_definitions=ignore_inertial,
        )
        self._apply_entity_post_load(builder, cfg)

    def _load_mjcf_entity(self, builder: newton.ModelBuilder, cfg: EntityCfg | NewtonEntityCfg) -> None:
        """Load MJCF (MuJoCo XML) entity via ``ModelBuilder.add_mjcf``.

        Parallels :meth:`_load_urdf_entity`: the only format-specific
        work is the ``add_mjcf`` call (with ``parse_sites=True`` so
        MJCF ``<site>`` tags are preserved as first-class shapes);
        everything after that — collapsing fixed joints, applying
        actuator gains and armature, mesh approximation, site
        creation — is shared with the URDF loader via
        :meth:`_apply_entity_post_load`.

        The MJCF source is expected to contain an ``<actuator>``
        block; Newton's MJCF loader reads that block to set each
        DOF's ``joint_target_mode``, which ``SolverMuJoCo`` in turn
        uses to decide whether to create a MuJoCo actuator per DOF.
        MJCFs that ship without ``<actuator>`` (e.g. mjlab's
        booster_t1 ``t1.xml``, which wires actuators via runtime
        Python) leave every mode at ``NONE`` and produce ``nu=0``;
        use a robot asset that includes ``<actuator>`` instead.
        """
        shape_cfg = getattr(cfg, "shape_cfg", None)
        if shape_cfg is not None:
            builder.default_shape_cfg = shape_cfg

        xform = wp.transform(wp.vec3(*cfg.init_state.pos), wp.quat(*cfg.init_state.rot))
        ignore_inertial = getattr(cfg, "ignore_inertial_definitions", False)
        builder.add_mjcf(
            cfg.mjcf_path,
            xform=xform,
            floating=cfg.floating,
            enable_self_collisions=cfg.enable_self_collisions,
            collapse_fixed_joints=False,
            ignore_inertial_definitions=ignore_inertial,
            parse_sites=True,
            ignore_names=["floor", "ground"],
        )
        self._apply_entity_post_load(builder, cfg)

    def _apply_entity_post_load(
        self,
        builder: newton.ModelBuilder,
        cfg: EntityCfg | NewtonEntityCfg,
    ) -> None:
        """Shared post-load work for URDF and MJCF entities.

        Runs after the format-specific ``add_urdf`` / ``add_mjcf``
        call, in this order:

        1. ``collapse_fixed_joints(joints_to_keep=cfg.links_to_keep)``
           so fixed-joint merging respects user-requested link
           retention (e.g. foot frames needed for sensors).
        2. Actuator gain and armature pattern application.
           Implicit actuators (``ImplicitActuatorCfg``) set
           ``ke``/``kd`` so Newton's internal PD drives the joint;
           explicit actuators (IdealPD, Delayed, …) zero those so
           only the external torque loop is active. Armature is
           always set since it is a physical property.
        3. Mesh approximation via
           ``builder.approximate_meshes(cfg.mesh_approximation)``.
        4. Optional IMU / sensor ``sites`` declared on
           ``NewtonEntityCfg.sites``.
        """
        if cfg.collapse_fixed_joints:
            # Newton's ``collapse_fixed_joints`` uses literal string
            # equality for ``joints_to_keep`` (builder.collapse_fixed_joints
            # → ``joint["label"] in joints_to_keep``), with no
            # wildcard / regex support. We widen bare leaf names to
            # ``*/<name>`` (a no-op for patterns already containing
            # '/' or glob metachars) and expand against the loaded
            # ``builder.joint_label`` set via ``fnmatch``, then hand
            # Newton the resolved literal labels. This mirrors how
            # actuator ``target_names_expr`` (``apply_joint_params_by_pattern``
            # canonicalises via ``leaf_name``) and sensor body patterns
            # (scene.py sensor block uses ``fnmatch``) already resolve
            # cross-format labels — so the same bare leaf pattern in
            # cfg works for URDF flat labels (``entity/joint``) and
            # MJCF XPath labels (``entity/.../joint``) uniformly.
            if cfg.links_to_keep:
                from fnmatch import fnmatch

                patterns = as_leaf_globs(list(cfg.links_to_keep))
                all_labels = list(builder.joint_label)
                expanded = sorted({label for pat in patterns for label in all_labels if fnmatch(label, pat)})
                if not expanded:
                    raise ValueError(
                        f"links_to_keep={cfg.links_to_keep!r} (widened to "
                        f"{patterns!r}) matched zero joint labels. "
                        f"Sample available labels: {all_labels[:8]}"
                    )
            else:
                expanded = []
            builder.collapse_fixed_joints(joints_to_keep=expanded)

        # Auto-extract ``body_label_prefix`` from the loaded labels when the
        # user did not pin it explicitly. URDF body labels start with the
        # robot's ``<robot name="...">`` (e.g. ``go2_description/base``) while
        # MJCF labels start with the ``add_mjcf`` ``name`` argument plus the
        # XPath hierarchy (e.g. ``go2/worldbody/trunk``). Hardcoding a single
        # prefix in user config silently breaks one of the two formats —
        # sensor pattern ``"X/*"`` matches zero labels when the actual prefix
        # is ``"Y"``, leaving sensing_bodies=[] and the contact reward at 0.
        if getattr(cfg, "body_label_prefix", None) is None and builder.body_label:
            cfg.body_label_prefix = builder.body_label[0].split("/")[0]

        prefix = getattr(cfg, "body_label_prefix", None)
        ke_map: dict[str, float] = {}
        kd_map: dict[str, float] = {}
        armature_map: dict[str, float] = {}
        effort_limit_map: dict[str, float] = {}
        friction_map: dict[str, float] = {}

        # ``apply_joint_params_by_pattern`` canonicalises candidate
        # joint labels via ``leaf_name()`` before regex-matching, so
        # the maps here must hold *bare* patterns (IsaacLab convention):
        # ``".*_hip_joint"`` not ``"<entity>/.*_hip_joint"``.

        for act_cfg in cfg.articulation.actuators:
            is_explicit = not isinstance(act_cfg, ImplicitActuatorCfg)

            if is_explicit:
                for pattern in act_cfg.target_names_expr:
                    ke_map[pattern] = 0.0
                    kd_map[pattern] = 0.0
            else:
                if isinstance(act_cfg.stiffness, dict):
                    ke_map.update(act_cfg.stiffness)
                elif act_cfg.stiffness is not None and act_cfg.stiffness > 0:
                    for pattern in act_cfg.target_names_expr:
                        ke_map[pattern] = act_cfg.stiffness

                if isinstance(act_cfg.damping, dict):
                    kd_map.update(act_cfg.damping)
                elif act_cfg.damping is not None and act_cfg.damping > 0:
                    for pattern in act_cfg.target_names_expr:
                        kd_map[pattern] = act_cfg.damping

            if isinstance(act_cfg.armature, dict):
                armature_map.update(act_cfg.armature)
            elif isinstance(act_cfg.armature, int | float) and act_cfg.armature > 0:
                for pattern in act_cfg.target_names_expr:
                    armature_map[pattern] = act_cfg.armature

            # Effort limit — overrides XML's actuatorfrcrange (which becomes
            # joint.actfrcrange in Newton's MuJoCo solver).
            if isinstance(act_cfg.effort_limit, dict):
                effort_limit_map.update(act_cfg.effort_limit)
            elif isinstance(act_cfg.effort_limit, int | float) and act_cfg.effort_limit > 0:
                for pattern in act_cfg.target_names_expr:
                    effort_limit_map[pattern] = float(act_cfg.effort_limit)

            # Frictionloss — overrides XML-declared joint frictionloss.
            # Mirrors MuJoCo's dof_frictionloss / Genesis' set_dofs_frictionloss
            # so the three sims share one authoritative value at build time.
            if isinstance(act_cfg.frictionloss, int | float) and act_cfg.frictionloss > 0:
                for pattern in act_cfg.target_names_expr:
                    friction_map[pattern] = float(act_cfg.frictionloss)

        if ke_map or kd_map or armature_map or effort_limit_map or friction_map:
            apply_joint_params_by_pattern(
                builder,
                ke_map=ke_map or None,
                kd_map=kd_map or None,
                armature_map=armature_map or None,
                effort_limit_map=effort_limit_map or None,
                friction_map=friction_map or None,
            )
        # Workaround: force ``geom_priority=1`` on every collision shape of this
        # entity. Newton's MJCF parser loses the XML-declared ``priority``
        # attribute on most geoms when ``parse_visuals=True`` (only ~4/12
        # collision geoms end up with the intended value). Since priority gates
        # MuJoCo's friction combine rule (higher-priority geom's value wins;
        # otherwise element-wise MAX with ground/terrain), losing it here makes
        # per-robot friction DR silently ineffective — randomized foot μ gets
        # max()ed with the fixed ground μ and the policy never sees the
        # variation. We patch the builder's custom_attribute values dict
        # directly because Newton exposes no per-shape setter for this. Remove
        # once the upstream parser bug is fixed.
        _force_collision_shape_priority(builder)
        mesh_approx = getattr(cfg, "mesh_approximation", "bounding_box")
        builder.approximate_meshes(mesh_approx)

        sites = getattr(cfg, "sites", None)
        if sites:
            self._create_sites_from_dict(builder, sites, prefix)

    def _load_usd_entity(self, builder: newton.ModelBuilder, cfg: EntityCfg | NewtonEntityCfg) -> None:
        """Load USD entity from unified config."""
        shape_cfg = getattr(cfg, "shape_cfg", None)
        if shape_cfg is not None:
            builder.default_shape_cfg = shape_cfg

        xform = wp.transform(wp.vec3(*cfg.init_state.pos), wp.quat(*cfg.init_state.rot))

        builder.add_usd(
            cfg.usd_path,
            xform=xform,
            collapse_fixed_joints=cfg.collapse_fixed_joints,
            enable_self_collisions=cfg.enable_self_collisions,
        )

        # Mesh approximation
        mesh_approx = getattr(cfg, "mesh_approximation", "convex_hull")
        if mesh_approx is not None:
            builder.approximate_meshes(mesh_approx)

        # Apply gains from articulation actuators. Bare patterns only —
        # ``apply_joint_params_by_pattern`` canonicalises candidate
        # joint labels via ``leaf_name()``.
        prefix = getattr(cfg, "body_label_prefix", None)
        ke_map: dict[str, float] = {}
        kd_map: dict[str, float] = {}
        armature_map: dict[str, float] = {}
        effort_limit_map: dict[str, float] = {}
        friction_map: dict[str, float] = {}

        for act_cfg in cfg.articulation.actuators:
            for pattern in act_cfg.target_names_expr:
                if act_cfg.stiffness is not None and act_cfg.stiffness > 0:
                    ke_map[pattern] = act_cfg.stiffness
                if act_cfg.damping is not None and act_cfg.damping > 0:
                    kd_map[pattern] = act_cfg.damping
                if act_cfg.armature > 0:
                    armature_map[pattern] = act_cfg.armature
                if isinstance(act_cfg.effort_limit, int | float) and act_cfg.effort_limit > 0:
                    effort_limit_map[pattern] = float(act_cfg.effort_limit)
                if isinstance(act_cfg.frictionloss, int | float) and act_cfg.frictionloss > 0:
                    friction_map[pattern] = float(act_cfg.frictionloss)

        if ke_map or kd_map or armature_map or effort_limit_map or friction_map:
            apply_joint_params_by_pattern(
                builder,
                ke_map=ke_map or None,
                kd_map=kd_map or None,
                armature_map=armature_map or None,
                effort_limit_map=effort_limit_map or None,
                friction_map=friction_map or None,
            )

        # Sites
        sites = getattr(cfg, "sites", None)
        if sites:
            self._create_sites_from_dict(builder, sites, prefix)

    def _create_sites_from_dict(
        self, builder: newton.ModelBuilder, sites: dict[str, str], prefix: str | None = None
    ) -> None:
        """Create sensor sites from a {site_name: body_name} dict.

        ``body_name`` is matched as a regex against the builder's body
        leaf names (IsaacLab convention). ``prefix`` is unused here —
        Newton's MJCF XPath labels are canonicalised via
        :func:`leaf_name` inside :meth:`_find_body_by_name`, so callers
        pass bare names (``"torso_link"``) or bare regex patterns.
        """
        for site_name, body_name in sites.items():
            body_idx = self._find_body_by_name(builder, body_name)
            if body_idx is not None:
                builder.add_site(body_idx, label=site_name)
            else:
                raise ValueError(f"Body '{body_name}' not found for site '{site_name}'")

    @staticmethod
    def _find_body_by_name(builder: newton.ModelBuilder, body_name: str) -> int | None:
        """Find body index by name in the builder.

        Runs at builder-time (before ``builder.finalize()``) so
        ArticulationView is not yet available; we canonicalize the
        candidate labels to their leaf segments here via
        :func:`leaf_name` so downstream callers can pass bare body
        names (``"torso_link"``) regardless of whether the loader
        stored a flat ``g1_29dof/torso_link`` or a deep MJCF XPath
        like ``g1_29dof/worldbody/pelvis/.../torso_link``. The
        pattern itself is still ``re.fullmatch``'d so regex forms
        (``".*Trunk"``) keep working unchanged.
        """
        for i, name in enumerate(builder.body_label):
            if re.fullmatch(body_name, leaf_name(name)):
                return i
        return None

    def build_scene(self) -> None:
        """Build the complete scene with all entities replicated."""
        if not self._entity_builders:
            raise RuntimeError("No entities registered. Call register_entities() first.")

        # Create scene builder and replicate entities
        scene_builder = newton.ModelBuilder()

        for entity_name, entity_builder in self._entity_builders.items():
            cfg = self.entities[entity_name]["config"]
            if isinstance(cfg, GroundPlaneCfg):
                # Ground plane: add once (global)
                scene_builder.add_builder(entity_builder)
            else:
                # Other entities: replicate
                scene_builder.replicate(entity_builder, self.config.num_worlds)

        # Finalize model
        self.model = scene_builder.finalize()

        # Request state attributes needed by sensors
        self._request_sensor_state_attributes()

        # Create solver
        if self.config.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model)
        elif self.config.solver_type == "mujoco":
            scfg = self.config.solver_cfg
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                solver=scfg.solver,
                integrator=scfg.integrator,
                cone=scfg.cone,
                iterations=scfg.iterations,
                ls_iterations=scfg.ls_iterations,
                njmax=scfg.njmax,
                nconmax=scfg.nconmax,
                impratio=scfg.impratio,
                tolerance=scfg.tolerance,
                ls_tolerance=scfg.ls_tolerance,
                ccd_tolerance=scfg.ccd_tolerance,
                ccd_iterations=scfg.ccd_iterations,
                sdf_iterations=scfg.sdf_iterations,
                ls_parallel=scfg.ls_parallel,
                use_mujoco_contacts=scfg.use_mujoco_contacts,
                use_mujoco_cpu=scfg.use_mujoco_cpu,
                enable_multiccd=scfg.enable_multiccd,
                disable_contacts=scfg.disable_contacts,
            )
        else:
            raise ValueError(f"Unsupported solver type: {self.config.solver_type}")

        # Create state and control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # NOTE: FK is called later after state initialization in reset()
        # Initial FK with model defaults (will be updated in reset)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Create collision pipeline
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        # Update entity tracking with replicated info
        for entity_name in self.entities:
            self.entities[entity_name]["model"] = self.model

        # Build ArticulationView for each entity BEFORE sensor creation —
        # contact sensors now source their body-name filters from the
        # view (bare leaf names) instead of raw ``model.body_label``.
        self._build_robot_view()

        # Create sensors (legacy NewtonSensorConfig path)
        self._create_sensors()

        # Create simulator-agnostic ContactSensorCfg wrappers. Each builds
        # its own native ``SensorContact`` (which requests the ``force``
        # contact attribute on construction), so this MUST run before the
        # ``newton.Contacts`` allocation below. The wrapper is stored under
        # ``cfg.name`` in ``self.sensors`` (generic iteration, pretty
        # print) AND in ``self._contact_sensor_wrappers`` (per-step history
        # push, allocation gating).
        if self.config.contact_sensors:
            for cs_cfg in self.config.contact_sensors:
                if not isinstance(cs_cfg, ContactSensorCfg):
                    raise TypeError(
                        f"NewtonSceneManagerConfig.contact_sensors expects ContactSensorCfg, "
                        f"got {type(cs_cfg).__name__}"
                    )
                if cs_cfg.name in self.sensors:
                    raise ValueError(f"Sensor '{cs_cfg.name}' already exists")
                wrapper = NewtonContactSensor(self, cs_cfg)
                self.sensors[cs_cfg.name] = wrapper
                self._contact_sensor_wrappers[cs_cfg.name] = wrapper

        # Create sensor-specific contacts with extended attributes
        has_contact_sensor = any(isinstance(s, SensorContact) for s in self.sensors.values()) or bool(
            self._contact_sensor_wrappers
        )
        if has_contact_sensor:
            self.sensor_contacts = newton.Contacts(
                self.solver.get_max_contact_count(),
                0,
                device=self.model.device,
                requested_attributes=self.model.get_requested_contact_attributes(),
            )
        else:
            self.sensor_contacts = None

        # Clean up temporary builders
        self._entity_builders.clear()

        # Validate actuator parameters
        self._validate_mujoco_actuators()

        # Build kinematic trees for URDF entities
        self._set_kinematic_trees()

    def _validate_mujoco_actuators(self) -> None:
        """Validate MuJoCo actuator parameters after solver creation.

        Two actuator families are supported:

        * **Affine-bias** (``biastype == mjBIAS_AFFINE``): created by
          Newton's URDF loader and by MJCF ``<position>`` /
          ``<velocity>`` tags. Stores position gain ``kp`` in
          ``gainprm[0]`` and the matching ``-kp`` / ``-kd`` in
          ``biasprm[1:3]``. For these we sanity-check that
          ``gainprm[0] == -biasprm[1]`` so the position gain and
          bias are internally consistent.

        * **No-bias** (``biastype == mjBIAS_NONE``): created by MJCF
          ``<motor>`` tags (direct torque actuators). ``gainprm[0]``
          holds the gear ratio (typically 1.0) and ``biasprm`` is
          zero. For these the position/velocity gain checks do not
          apply — JaxRLWorld drives these joints via the external
          torque loop regardless — and the consistency assertion
          would spuriously fire if we reused the affine logic.
        """
        if self.config.solver_type != "mujoco":
            return

        mj_model = self.solver.mj_model
        num_actuators = mj_model.nu
        if num_actuators == 0:
            print("✓ MuJoCo actuators validated: 0 actuators (no external control)")
            return

        import mujoco  # noqa: PLC0415 — lazy to avoid mjwarp import at module load

        gainprm = mj_model.actuator_gainprm
        biasprm = mj_model.actuator_biasprm
        biastype = mj_model.actuator_biastype

        affine_mask = biastype == int(mujoco.mjtBias.mjBIAS_AFFINE)
        motor_mask = biastype == int(mujoco.mjtBias.mjBIAS_NONE)

        # Per-actuator ke / kd. Meaning depends on biastype — for
        # motor-type actuators gainprm[0] is the gear ratio, not a
        # position gain, and biasprm is zero.
        ke = gainprm[:, 0]
        kd = -biasprm[:, 2]
        ke_bias = -biasprm[:, 1]

        # Gains are always non-negative regardless of actuator type.
        assert (ke >= 0).all(), f"Negative actuator gains: {ke}"

        if affine_mask.any():
            assert (kd[affine_mask] >= 0).all(), f"Negative velocity gains on affine actuators: {kd[affine_mask]}"
            assert np.allclose(ke[affine_mask], ke_bias[affine_mask]), "Position gain/bias mismatch on affine actuators"

        num_affine = int(affine_mask.sum())
        num_motor = int(motor_mask.sum())
        num_other = num_actuators - num_affine - num_motor
        print(
            f"✓ MuJoCo actuators validated: {num_actuators} total "
            f"({num_affine} affine, {num_motor} motor, {num_other} other)"
        )
        if num_affine > 0:
            num_zero_affine = int((ke[affine_mask] == 0).sum())
            print(f"  affine ke range: [{ke[affine_mask].min():.2f}, {ke[affine_mask].max():.2f}]")
            print(f"  affine kd range: [{kd[affine_mask].min():.2f}, {kd[affine_mask].max():.2f}]")
            if num_zero_affine > 0:
                print(f"  ({num_zero_affine} affine joints with ke=0: explicit actuator mode)")
        if num_motor > 0:
            print(f"  motor gear range: [{ke[motor_mask].min():.2f}, {ke[motor_mask].max():.2f}]")

    def _build_robot_view(self) -> None:
        """Create an ArticulationView for each non-ground-plane entity.

        Uses each entity's body_label_prefix to select only that entity's
        articulation, excluding ground plane and other global shapes.

        The view is built **without** ``exclude_joint_types``: keeping
        the free (or fixed) root joint preserves alignment between the
        view's DOF space and ``model.joint_q`` / ``joint_qd`` so
        ``newton_q_indices`` / ``newton_qd_indices`` (computed off
        ``model.joint_q_start`` — full model coord space) index both
        ``view.get_dof_positions`` and the zero-copy
        ``state.joint_q.reshape(num_worlds, coords_per_world)`` views
        consistently. Free-joint DoFs are filtered out later via user
        pattern matching (bare actuated regexes like
        ``"FR_hip_joint"`` never fullmatch Newton's multi-DOF names
        like ``"root_joint:0"``).
        """
        for entity_name, entity_info in self.entities.items():
            cfg = entity_info["config"]
            if isinstance(cfg, GroundPlaneCfg):
                continue
            prefix = getattr(cfg, "body_label_prefix", None)
            pattern = f"{prefix}*" if prefix else "*"
            self.articulation_views[entity_name] = ArticulationView(
                self.model,
                pattern=pattern,
            )

    @property
    def robot_view(self) -> ArticulationView:
        """Shortcut for the 'robot' entity's ArticulationView."""
        return self.articulation_views["robot"]

    @property
    def robot_data(self):
        """NewtonRobotData from the environment (single instance)."""
        return self.env._robot_data

    @property
    def robot_state(self):
        """Alias for robot_data (backward compat)."""
        return self.robot_data

    @property
    def robot_state_writer(self):
        """NewtonRobotStateWriter from the environment (single instance).

        Mutation companion to ``robot_data`` — used by event terms and
        reset functions for joint/root writes and FK evaluation.
        """
        return self.env._robot_state_writer

    def _set_kinematic_trees(self) -> None:
        """Build kinematic trees for entities with URDF or MJCF source."""

        def _resolve(name: str):
            cfg = self.entities[name]["config"]
            mjcf_path = getattr(cfg, "mjcf_path", None)
            if mjcf_path:
                return ("mjcf_path", mjcf_path)
            urdf_path = getattr(cfg, "urdf_path", None)
            if urdf_path:
                return ("urdf", urdf_path)
            return None

        self.trees = build_kinematic_trees(self.entities.keys(), _resolve)

    def _request_sensor_state_attributes(self) -> None:
        """Request extended state attributes needed by sensors and data-API.

        Always requests ``mujoco:qfrc_actuator`` when using the MuJoCo
        solver so :attr:`NewtonRobotData.applied_torque` is readable
        uniformly (energy-based termination, actuator-torque rewards).
        The allocation is a single ``wp.array`` the size of
        ``joint_dof_count`` per state, so the cost is negligible even
        when nothing reads it.
        """
        if self.config.solver_type == "mujoco":
            self.model.request_state_attributes("mujoco:qfrc_actuator")

        if not self.config.sensors:
            return

        for sensor_config in self.config.sensors:
            if isinstance(sensor_config, NewtonIMUSensorConfig):
                # IMU needs body_qdd
                self.model.request_state_attributes("body_qdd")
                break  # Only need to request once

    def _create_sensors(self) -> None:
        """Create all sensors from config."""
        if not self.config.sensors:
            return

        for sensor_config in self.config.sensors:
            self._create_sensor(sensor_config)

    def _create_sensor(self, config: NewtonSensorConfig) -> None:
        """Create a single sensor from its config."""
        if config.sensor_name in self.sensors:
            raise ValueError(f"Sensor '{config.sensor_name}' already exists")

        if isinstance(config, NewtonIMUSensorConfig):
            all_site_indices = [idx for idx, key in enumerate(self.model.shape_label) if key in config.site_names]

            if not all_site_indices:
                raise ValueError(f"No sites found matching {config.site_names}")

            if config.sensor_name in self.sensors:
                raise ValueError(f"Sensor '{config.sensor_name}' already exists")

            sensor = newton.sensors.SensorIMU(self.model, all_site_indices)
            self.sensors[config.sensor_name] = sensor

        elif isinstance(config, NewtonContactSensorConfig):
            entity_name = config.entity_name
            # Widen bare leaf-name body patterns (e.g. ``"left_ankle_roll_link"``)
            # to ``*/<name>`` BEFORE prefix injection, so SensorContact's
            # internal fnmatch hits both URDF flat labels
            # (``<entity>/<name>``) and MJCF XPath labels
            # (``<entity>/<ancestor>/.../<name>``). Order matters:
            # ``_prefix_names`` skips patterns that already contain ``/``,
            # so widening first keeps bare-name configs (like
            # ``r.foot_names = ["left_ankle_roll_link"]``) working on both
            # loaders, while pre-scoped or glob / regex patterns pass
            # through unchanged.
            sensing_bodies = self._prefix_names(entity_name, as_leaf_globs(config.sensing_obj_bodies))
            sensing_shapes = self._prefix_names(entity_name, config.sensing_obj_shapes)
            counterpart_bodies = self._prefix_names(entity_name, as_leaf_globs(config.counterpart_bodies))

            # Apply exclude filter. Matching happens against world-0
            # slice of ``model.body_label`` (full labels — both the
            # entity prefix already injected by ``_prefix_names`` and
            # any XPath segments produced by Newton's MJCF loader).
            # ``sensing_bodies`` stays as a list of full labels so
            # SensorContact's internal fnmatch resolves them directly
            # against ``model.body_label``.
            if config.exclude_bodies and sensing_bodies is not None:
                from fnmatch import fnmatch

                all_labels = self.model.body_label
                world_count = self.model.world_count
                bodies_per_env = len(all_labels) // world_count
                first_env_labels = all_labels[:bodies_per_env]

                patterns = [sensing_bodies] if isinstance(sensing_bodies, str) else sensing_bodies
                matched_indices = [
                    idx for idx, label in enumerate(first_env_labels) if any(fnmatch(label, p) for p in patterns)
                ]
                matched_indices = [
                    idx
                    for idx in matched_indices
                    if not any(fnmatch(first_env_labels[idx], exc) for exc in config.exclude_bodies)
                ]
                sensing_bodies = [first_env_labels[idx] for idx in matched_indices]

            sensor = SensorContact(
                self.model,
                sensing_obj_bodies=sensing_bodies,
                sensing_obj_shapes=sensing_shapes,
                counterpart_bodies=counterpart_bodies,
                counterpart_shapes=config.counterpart_shapes,
                measure_total=config.include_total,
            )
            self.sensors[config.sensor_name] = sensor

        elif isinstance(config, NewtonFrameTransformSensorConfig):
            site_indices = [idx for idx, key in enumerate(self.model.shape_label) if key in config.site_names]

            if site_indices:
                sensor = config.create_sensor(self.model, site_indices)
                self.sensors[config.sensor_name] = sensor

    def capture(self):
        num_worlds = self.model.world_count
        base_zeros = torch.zeros((num_worlds, 6), device=self.device, dtype=torch.float32)
        dummy_actions = torch.zeros(
            (num_worlds, self.env.act_manager.num_actions), device=self.device, dtype=torch.float32
        )

        full_targets = torch.cat([base_zeros, dummy_actions], dim=-1).flatten()
        self.control.joint_target_pos = wp.from_torch(full_targets, dtype=wp.float32, requires_grad=False)

        with wp.ScopedCapture() as capture:
            self._step()
        self.graph = capture.graph

    def build_articulation_indexing(self, actuated_dof_names: list[str]):
        """Build ArticulationIndexing for the Newton model.

        Args:
            actuated_dof_names: Regex patterns for actuated joints (may include prefix).

        Returns:
            ArticulationIndexing with canonical ↔ simulator mappings.
        """
        # Source joint names from ArticulationView, which already emits
        # bare leaf names (``left_hip_pitch_joint``) with entity prefix
        # and XPath ancestors stripped — IsaacLab convention.
        view = self.articulation_views.get("robot")
        if view is None:
            raise ValueError(
                "build_articulation_indexing called before _build_robot_view; "
                "ArticulationView must be created before action-manager init."
            )
        # Use ``view.joint_names`` (one entry per JOINT) rather than
        # ``view.joint_dof_names`` (one entry per DOF, with ``:N``
        # suffixes for multi-DOF joints like free / spherical).
        # ``flat_world0`` lookups resolve one entry per joint, so the
        # DOF-level expansion would break the ``.index(n)`` below for
        # free-joint entries like ``"floating_base:0"``. User actuator
        # regexes match bare joint names (``"FR_hip_joint"``) so
        # multi-DOF joints are filtered out naturally by ``fullmatch``.
        all_names = list(view.joint_names)
        # Raw Newton joint-label list over all worlds is still needed
        # to recover q / qd indices into the flat model arrays.
        joint_names_raw = getattr(self.model, "joint_label", None) or getattr(self.model, "joint_key", None)
        if not joint_names_raw:
            raise ValueError("Newton model has no joint labels")
        num_worlds = self.model.world_count
        joints_per_world = len(joint_names_raw) // num_worlds
        flat_world0 = [leaf_name(n) for n in joint_names_raw[:joints_per_world]]

        # Indices of actuatable joints within world-0 of the flat array.
        # All of ``all_names`` entries must exist in ``flat_world0``.
        actuatable_names = all_names
        actuatable_indices = [flat_world0.index(n) for n in all_names]

        matched_indices, matched_names = string_utils.resolve_matching_names(
            actuated_dof_names, actuatable_names, preserve_order=True
        )
        original_indices = [actuatable_indices[i] for i in matched_indices]

        # Compute q and qd indices
        joint_q_start = wp.to_torch(self.model.joint_q_start).cpu().numpy()
        joint_qd_start = wp.to_torch(self.model.joint_qd_start).cpu().numpy()

        q_indices = torch.tensor([int(joint_q_start[j]) for j in original_indices], device=self.device)
        qd_indices = torch.tensor([int(joint_qd_start[j]) for j in original_indices], device=self.device)

        # sim_indices = qd_indices (used for _apply_force / _apply_position)
        sim_indices = qd_indices

        # sim_to_canonical: maps qd-space positions back to canonical
        # For Newton, RobotData uses q_indices/qd_indices directly
        sim_to_canonical = torch.zeros_like(sim_indices)
        for canonical_i, sim_i in enumerate(sim_indices):
            sim_to_canonical[canonical_i] = canonical_i  # identity since we index directly

        # Joint limits (in qd space)
        dofs_per_world = self.model.joint_dof_count // num_worlds
        lower_all = wp.to_torch(self.model.joint_limit_lower)[:dofs_per_world]
        upper_all = wp.to_torch(self.model.joint_limit_upper)[:dofs_per_world]
        lower = lower_all[qd_indices]
        upper = upper_all[qd_indices]

        return ArticulationIndexing(
            joint_names=tuple(matched_names),
            sim_indices=sim_indices,
            sim_to_canonical=sim_to_canonical,
            joint_limits_lower=lower,
            joint_limits_upper=upper,
            newton_q_indices=q_indices,
            newton_qd_indices=qd_indices,
        )

    def step(self) -> None:
        """Advance physics by one control step (multiple substeps)."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._step()

        # Update sensors
        self._update_sensors()

    def _step(self):
        # When substeps is odd AND the physics loop is wrapped in a
        # CUDA graph, a plain ``state_0, state_1 = state_1, state_0``
        # swap on every substep would leave the two state references
        # crossed (relative to the capture) on exit — the graph was
        # recorded against specific memory addresses, so replay would
        # read/write the wrong state. Newton's own example
        # ``newton/examples/robot/example_robot_policy.py:326-342``
        # handles this by copying state on the final odd iteration
        # instead of swapping. We mirror that here line-for-line
        # against Newton's reference.
        need_state_copy = self.use_cuda_graph and self.config.substeps % 2 == 1
        last_idx = self.config.substeps - 1

        self.contacts = self.model.collide(
            self.state_0, contacts=self.contacts, collision_pipeline=self.collision_pipeline
        )
        for i in range(self.config.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.substep_dt)

            if need_state_copy and i == last_idx:
                # Copy state_1 → state_0 so the final result lives in
                # the same buffer the graph capture started with.
                # Uses ``_state_assign_full`` (not bare ``assign``) so
                # extended attributes under namespaces like
                # ``state.mujoco.qfrc_actuator`` also propagate —
                # Newton's built-in assign skips them.
                _state_assign_full(self.state_0, self.state_1)
            else:
                # Swap states (reference rebind, no memory copy).
                self.state_0, self.state_1 = self.state_1, self.state_0

    def _update_sensors(self) -> None:
        """Update all sensors after physics step.

        Called once per ``scene.step()`` — i.e. once per physics step,
        ``decimation`` times per control step (and never from inside the
        captured CUDA graph, so dynamic torch ops in the ContactSensorCfg
        wrappers are safe here). The wrappers also push one substep frame
        into their ring buffer on every call (see ``NewtonContactSensor``).
        """
        has_contact_sensor = any(isinstance(s, SensorContact) for s in self.sensors.values()) or bool(
            self._contact_sensor_wrappers
        )
        if has_contact_sensor:
            self.solver.update_contacts(self.sensor_contacts, self.state_0)

        for sensor_name, sensor in self.sensors.items():
            if isinstance(sensor, NewtonContactSensor):
                # Wrapper: refreshes its native SensorContact + pushes history.
                sensor.update(self.state_0, self.sensor_contacts)
            elif isinstance(sensor, SensorContact):
                sensor.update(self.state_0, self.sensor_contacts)  # Contact sensor takes Contacts
            elif hasattr(sensor, "update"):
                sensor.update(self.state_0)  # IMU takes State

    def reset(self, env_ids=None) -> None:
        """Reset specified environments to model defaults (joint_q, joint_qd).

        Supports partial reset: only the given *env_ids* are overwritten while
        the remaining environments keep their current state.

        This bypasses the ``RobotStateWriterProtocol`` because the model
        defaults live as warp arrays inside ``view.get_dof_*(model)`` and
        we want to feed them straight into ``view.set_dof_*(state)``
        without a torch round-trip.
        """
        if env_ids is None or len(env_ids) == 0:
            return

        view = self.robot_view

        # Build warp mask inline (avoid torch round-trip).
        mask_torch = torch.zeros(self.model.world_count, dtype=torch.bool, device=self.env.device)
        mask_torch[env_ids] = True
        mask = wp.from_torch(mask_torch)

        # Copy model defaults into state for the reset environments
        view.set_dof_positions(self.state_0, view.get_dof_positions(self.model), mask=mask)
        view.set_dof_velocities(self.state_0, view.get_dof_velocities(self.model), mask=mask)

        # Re-evaluate FK for reset environments only
        view.eval_fk(self.state_0, mask=mask)

    def get_sensor(self, sensor_name: str) -> Any:
        """Get a sensor by name."""
        return self.sensors.get(sensor_name)

    def get_body_positions(self) -> wp.array:
        """Get body positions [num_bodies, 7] (pos + quat)."""
        return self.state_0.body_q

    def get_body_velocities(self) -> wp.array:
        """Get body velocities [num_bodies, 6] (linear + angular)."""
        return self.state_0.body_qd

    def get_joint_positions(self) -> wp.array:
        """Get joint positions [joint_coord_count]."""
        return self.state_0.joint_q

    def get_joint_velocities(self) -> wp.array:
        """Get joint velocities [joint_dof_count]."""
        return self.state_0.joint_qd
