from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Dict

import numpy as np
import torch
import warp as wp

import newton
from newton.sensors import SensorContact
from rlworld.rl.configs.robots.kinematic_tree import KinematicTree
from rlworld.rl.configs.scene.newton_entity_config import (
    NewtonEntityConfig,
    NewtonGroundPlaneConfig,
)
from rlworld.rl.configs.scene.unified_entity_config import (
    EntityCfg, NewtonEntityCfg, GroundPlaneCfg,
)
from rlworld.rl.configs.sensors.newton_sensor_config import (
    NewtonSensorConfig,
    NewtonIMUSensorConfig,
    NewtonContactSensorConfig,
    NewtonFrameTransformSensorConfig,
)
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def apply_joint_params_by_pattern(
    builder,
    ke_map: Dict[str, float] | None = None,
    kd_map: Dict[str, float] | None = None,
    armature_map: Dict[str, float] | None = None,
) -> None:
    """Apply joint parameters (target gains, armature) using regex pattern matching."""
    if ke_map is None and kd_map is None and armature_map is None:
        return

    ke_map = ke_map or {}
    kd_map = kd_map or {}
    armature_map = armature_map or {}
    num_joints = len(builder.joint_label)

    for joint_idx, joint_name in enumerate(builder.joint_label):
        if joint_idx < num_joints - 1:
            dof_count = builder.joint_qd_start[joint_idx + 1] - builder.joint_qd_start[joint_idx]
        else:
            dof_count = builder.joint_dof_count - builder.joint_qd_start[joint_idx]

        if dof_count == 0:
            continue

        dof_start = builder.joint_qd_start[joint_idx]

        # Apply ke
        for pattern, value in ke_map.items():
            if re.match(pattern, joint_name):
                for d in range(dof_count):
                    builder.joint_target_ke[dof_start + d] = value
                break

        # Apply kd
        for pattern, value in kd_map.items():
            if re.match(pattern, joint_name):
                for d in range(dof_count):
                    builder.joint_target_kd[dof_start + d] = value
                break

        # Apply armature
        for pattern, value in armature_map.items():
            if re.match(pattern, joint_name):
                for d in range(dof_count):
                    builder.joint_armature[dof_start + d] = value


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

    def __init__(self, env: "World", config: NewtonSceneManagerConfig):
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

        # Entity tracking
        self.entities: dict[str, Any] = {}  # entity_name -> entity info
        self._entity_builders: dict[str, newton.ModelBuilder] = {}  # Temporary during build
        self._body_name_to_idx: dict[str, dict[str, int]] = defaultdict(dict)  # entity_name -> {body_name: body_idx}

        # Sensor tracking
        self.sensors: dict[str, Any] = {}  # sensor_name -> sensor object

        # Kinematic trees (for observation functions)
        self.trees: dict[str, Any] = {}

        # Internal
        self.substep_dt = config.dt / config.substeps

    @property
    def robot(self) -> Any:
        """For compatibility - returns model in Newton."""
        return self.model

    @property
    def state(self) -> newton.State:
        """Current state (state_0)."""
        return self.state_0

    def find_body_names(self, body_names: list[str]):
        num_bodies_per_env = len(self.model.body_label) // self.env.num_envs
        bodies_key = self.model.body_label[:num_bodies_per_env]

        _, names = string_utils.resolve_matching_names(body_names, bodies_key)
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
        elif cfg.urdf_path:
            self._load_urdf_entity(builder, cfg)
        else:
            raise ValueError(f"Entity '{entity_name}' has no urdf_path or usd_path")

        self._entity_builders[entity_name] = builder
        self.entities[entity_name] = {
            "config": cfg,
            "builder": builder,
            "shape_count": len(builder.shape_label),
        }

    def _load_urdf_entity(self, builder: newton.ModelBuilder, cfg: EntityCfg | NewtonEntityCfg) -> None:
        """Load URDF entity from unified config."""
        # Shape config (Newton-specific)
        shape_cfg = getattr(cfg, "shape_cfg", None)
        if shape_cfg is not None:
            builder.default_shape_cfg = shape_cfg

        # Load URDF
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
        builder.collapse_fixed_joints(joints_to_keep=cfg.links_to_keep)

        # Apply gains and armature from articulation actuators.
        # Implicit actuators → set ke/kd so Newton's internal PD drives them.
        # Explicit actuators (IdealPD, Delayed, etc.) → set ke=0, kd=0 so
        #   Newton's internal PD is disabled; only our external torques apply.
        #   Armature is still set (it's a physical property, not a control gain).
        from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg

        prefix = getattr(cfg, "body_label_prefix", None)
        ke_map: dict[str, float] = {}
        kd_map: dict[str, float] = {}
        armature_map: dict[str, float] = {}

        def _prefixed(d: dict[str, float]) -> dict[str, float]:
            """Add body_label_prefix to regex keys."""
            if not prefix:
                return d
            return {f"{prefix}/{k}": v for k, v in d.items()}

        for act_cfg in cfg.articulation.actuators:
            is_explicit = not isinstance(act_cfg, ImplicitActuatorCfg)

            if is_explicit:
                # Explicit actuator: zero out solver PD gains for all target joints
                for pattern in act_cfg.target_names_expr:
                    key = f"{prefix}/{pattern}" if prefix else pattern
                    ke_map[key] = 0.0
                    kd_map[key] = 0.0
            else:
                # Implicit actuator: set solver PD gains
                if isinstance(act_cfg.stiffness, dict):
                    ke_map.update(_prefixed(act_cfg.stiffness))
                elif act_cfg.stiffness is not None and act_cfg.stiffness > 0:
                    for pattern in act_cfg.target_names_expr:
                        ke_map[f"{prefix}/{pattern}" if prefix else pattern] = act_cfg.stiffness

                if isinstance(act_cfg.damping, dict):
                    kd_map.update(_prefixed(act_cfg.damping))
                elif act_cfg.damping is not None and act_cfg.damping > 0:
                    for pattern in act_cfg.target_names_expr:
                        kd_map[f"{prefix}/{pattern}" if prefix else pattern] = act_cfg.damping

            # Armature is always set (physical property)
            if isinstance(act_cfg.armature, dict):
                armature_map.update(_prefixed(act_cfg.armature))
            elif isinstance(act_cfg.armature, (int, float)) and act_cfg.armature > 0:
                for pattern in act_cfg.target_names_expr:
                    armature_map[f"{prefix}/{pattern}" if prefix else pattern] = act_cfg.armature

        if ke_map or kd_map or armature_map:
            apply_joint_params_by_pattern(
                builder, ke_map=ke_map or None, kd_map=kd_map or None, armature_map=armature_map or None,
            )

        # Mesh approximation
        mesh_approx = getattr(cfg, "mesh_approximation", "bounding_box")
        builder.approximate_meshes(mesh_approx)

        # Sites (Newton-specific)
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

        # Apply gains from articulation actuators
        prefix = getattr(cfg, "body_label_prefix", None)
        ke_map: dict[str, float] = {}
        kd_map: dict[str, float] = {}
        armature_map: dict[str, float] = {}

        for act_cfg in cfg.articulation.actuators:
            for pattern in act_cfg.target_names_expr:
                key = f"{prefix}/{pattern}" if prefix else pattern
                if act_cfg.stiffness is not None and act_cfg.stiffness > 0:
                    ke_map[key] = act_cfg.stiffness
                if act_cfg.damping is not None and act_cfg.damping > 0:
                    kd_map[key] = act_cfg.damping
                if act_cfg.armature > 0:
                    armature_map[key] = act_cfg.armature

        if ke_map or kd_map or armature_map:
            apply_joint_params_by_pattern(
                builder, ke_map=ke_map or None, kd_map=kd_map or None, armature_map=armature_map or None,
            )

        # Sites
        sites = getattr(cfg, "sites", None)
        if sites:
            self._create_sites_from_dict(builder, sites, prefix)

    def _create_sites_from_dict(
        self, builder: newton.ModelBuilder, sites: dict[str, str], prefix: str | None = None
    ) -> None:
        """Create sensor sites from a {site_name: body_name} dict."""
        def _resolve(name: str) -> str:
            return f"{prefix}/{name}" if prefix and "/" not in name else name

        for site_name, body_name in sites.items():
            body_idx = self._find_body_by_name(builder, _resolve(body_name))
            if body_idx is not None:
                builder.add_site(body_idx, label=site_name)
            else:
                raise ValueError(f"Body '{body_name}' not found for site '{site_name}'")

    @staticmethod
    def _find_body_by_name(builder: newton.ModelBuilder, body_name: str) -> int | None:
        """Find body index by name in the builder."""
        for i, name in enumerate(builder.body_label):
            if name == body_name:
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
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                solver="newton",
                ls_parallel=True,
                njmax=1500,
                nconmax=150,
                impratio=100,
                iterations=100,
                ls_iterations=50,
                use_mujoco_contacts=True,

                # solver="newton",
                # integrator="implicitfast",
                # njmax=300,
                # nconmax=150,
                # cone="elliptic",
                # impratio=100,
                # iterations=100,
                # ls_iterations=50,
                # use_mujoco_contacts=True,
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
        # self.contacts = self.model.contacts()

        # Update entity tracking with replicated info
        for entity_name in self.entities:
            self.entities[entity_name]["model"] = self.model

        # Create sensors
        self._create_sensors()

        # Create sensor-specific contacts with extended attributes
        if any(isinstance(s, SensorContact) for s in self.sensors.values()):
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
        """Validate MuJoCo actuator parameters after solver creation."""
        if self.config.solver_type != "mujoco":
            return

        mj_model = self.solver.mj_model
        gainprm = mj_model.actuator_gainprm
        biasprm = mj_model.actuator_biasprm

        num_actuators = mj_model.nu

        # New structure: single actuator per joint with P+D combined
        ke = gainprm[:, 0]  # position gain
        kd = -biasprm[:, 2]  # velocity damping (stored as negative)
        ke_bias = -biasprm[:, 1]  # should equal ke

        # When explicit actuators set ke/kd=0, gains are intentionally zero.
        # Only validate that gains are non-negative and gain/bias are consistent.
        assert (ke >= 0).all(), f"Negative position gains: {ke}"
        assert (kd >= 0).all(), f"Negative velocity gains: {kd}"
        assert np.allclose(ke, ke_bias), f"Position gain/bias mismatch"

        num_zero = (ke == 0).sum()
        print(f"✓ MuJoCo actuators validated: {num_actuators} joints")
        print(f"  ke range: [{ke.min():.2f}, {ke.max():.2f}]")
        print(f"  kd range: [{kd.min():.2f}, {kd.max():.2f}]")
        if num_zero > 0:
            print(f"  ({num_zero} joints with ke=0: explicit actuator mode)")

    def _set_kinematic_trees(self) -> None:
        """Build kinematic trees for entities with URDF."""
        for entity_name, entity_info in self.entities.items():
            cfg = entity_info["config"]
            urdf_path = getattr(cfg, "urdf_path", None)
            if urdf_path is not None:
                self.trees[entity_name] = KinematicTree(urdf_path=urdf_path)

    def _request_sensor_state_attributes(self) -> None:
        """Request extended state attributes needed by sensors."""
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
            all_site_indices = [
                idx for idx, key in enumerate(self.model.shape_label)
                if key in config.site_names
            ]

            if not all_site_indices:
                raise ValueError(f"No sites found matching {config.site_names}")

            if config.sensor_name in self.sensors:
                raise ValueError(f"Sensor '{config.sensor_name}' already exists")

            sensor = newton.sensors.SensorIMU(self.model, all_site_indices)
            self.sensors[config.sensor_name] = sensor

        elif isinstance(config, NewtonContactSensorConfig):
            entity_name = config.entity_name
            sensor = SensorContact(
                self.model,
                sensing_obj_bodies=self._prefix_names(entity_name, config.sensing_obj_bodies),
                sensing_obj_shapes=self._prefix_names(entity_name, config.sensing_obj_shapes),
                counterpart_bodies=self._prefix_names(entity_name, config.counterpart_bodies),
                counterpart_shapes=config.counterpart_shapes,
                measure_total=config.include_total,
            )
            self.sensors[config.sensor_name] = sensor

        elif isinstance(config, NewtonFrameTransformSensorConfig):
            site_indices = [
                idx for idx, key in enumerate(self.model.shape_label)
                if key in config.site_names
            ]

            if site_indices:
                sensor = config.create_sensor(self.model, site_indices)
                self.sensors[config.sensor_name] = sensor

    def capture(self):
        num_worlds = self.model.world_count
        base_zeros = torch.zeros(
            (num_worlds, 6), device=self.device, dtype=torch.float32
        )
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
        from rlworld.rl.envs.indexing import ArticulationIndexing

        # Get all joint names from model (single world)
        joint_names_raw = getattr(self.model, "joint_label", None) or getattr(self.model, "joint_key", None)
        if not joint_names_raw:
            raise ValueError("Newton model has no joint labels")
        all_names = list(joint_names_raw)
        num_worlds = self.model.world_count
        joints_per_world = len(all_names) // num_worlds
        all_names = all_names[:joints_per_world]

        # Filter out floating_base
        actuatable = [(i, name) for i, name in enumerate(all_names) if name != "floating_base"]
        actuatable_names = [name for _, name in actuatable]
        actuatable_indices = [i for i, _ in actuatable]

        matched_indices, matched_names = string_utils.resolve_matching_names(
            actuated_dof_names, actuatable_names, preserve_order=True
        )
        original_indices = [actuatable_indices[i] for i in matched_indices]

        # Compute q and qd indices
        joint_q_start = wp.to_torch(self.model.joint_q_start).cpu().numpy()
        joint_qd_start = wp.to_torch(self.model.joint_qd_start).cpu().numpy()

        q_indices = torch.tensor(
            [int(joint_q_start[j]) for j in original_indices], device=self.device
        )
        qd_indices = torch.tensor(
            [int(joint_qd_start[j]) for j in original_indices], device=self.device
        )

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

        for _ in range(self.config.substeps):
            self.contacts = self.model.collide(
                self.state_0,
                contacts=self.contacts,
                collision_pipeline=self.collision_pipeline
            )
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.substep_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _update_sensors(self) -> None:
        """Update all sensors after physics step."""
        has_contact_sensor = any(
            isinstance(s, SensorContact) for s in self.sensors.values()
        )
        if has_contact_sensor:
            self.solver.update_contacts(self.sensor_contacts, self.state_0)

        for sensor_name, sensor in self.sensors.items():
            if isinstance(sensor, SensorContact):
                sensor.update(self.state_0, self.sensor_contacts)  # Contact sensor takes Contacts
            elif hasattr(sensor, 'update'):
                sensor.update(self.state_0)  # IMU takes State

    def reset(self, env_ids=None) -> None:
        """Reset environments to initial state."""
        # For now, full reset if all environments
        if env_ids is not None and len(env_ids) == self.config.num_worlds:
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

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
