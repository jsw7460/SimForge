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
    entities: list[NewtonEntityConfig] = field(default_factory=list)
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
        self._sim_dt = config.dt / config.substeps

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
        """Register all entities defined in config.

        This creates a ModelBuilder for each entity and prepares for replication.
        """
        for entity_config in self.config.entities:
            self._register_entity(entity_config)

    def _register_entity(self, config: NewtonEntityConfig) -> None:
        """Register a single entity from its config."""
        if config.entity_name in self.entities:
            raise ValueError(f"Entity '{config.entity_name}' already registered")

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        if config.entity_type == "ground_plane":
            if config.shape_cfg is not None:
                builder.add_ground_plane(cfg=config.shape_cfg)
            else:
                builder.add_ground_plane()

        elif config.entity_type == "urdf":
            self._load_urdf_entity(builder, config)

        elif config.entity_type == "usd":
            self._load_usd_entity(builder, config)

        else:
            raise ValueError(f"Unknown entity_type: {config.entity_type}")

        # Store builder for later
        self._entity_builders[config.entity_name] = builder
        self.entities[config.entity_name] = {
            "config": config,
            "builder": builder,
            "shape_count": len(builder.shape_label)
        }

    def _load_urdf_entity(self, builder: newton.ModelBuilder, config: NewtonEntityConfig) -> None:
        """Load URDF entity."""
        if config.joint_cfg is not None:
            builder.default_joint_cfg = config.joint_cfg
            builder.default_body_armature = config.joint_cfg.armature

        if config.shape_cfg is not None:
            builder.default_shape_cfg = config.shape_cfg

        if config.urdf_path is not None:
            builder.add_urdf(
                config.urdf_path,
                xform=config.transform,
                floating=config.floating,
                enable_self_collisions=config.enable_self_collisions,
                collapse_fixed_joints=False,
                ignore_inertial_definitions=config.ignore_inertial_definitions
            )
            builder.collapse_fixed_joints(joints_to_keep=config.joints_to_keep)

            # Apply joint params
            if config.joint_target_ke_map or config.joint_target_kd_map:
                apply_joint_params_by_pattern(
                    builder,
                    ke_map=config.joint_target_ke_map,
                    kd_map=config.joint_target_kd_map,
                    armature_map=config.joint_armature_map
                )
            elif config.joint_cfg is not None:
                for i in range(builder.joint_dof_count):
                    builder.joint_target_ke[i] = config.joint_cfg.target_ke
                    builder.joint_target_kd[i] = config.joint_cfg.target_kd

            # Todo: 여기 configuration으로 고치기
            builder.approximate_meshes("bounding_box")

        # Create sites
        self._create_sites(builder, config)

    def _load_usd_entity(self, builder: newton.ModelBuilder, config: NewtonEntityConfig) -> None:
        """Load USD entity."""
        if config.joint_cfg is not None:
            builder.default_joint_cfg = config.joint_cfg
            builder.default_body_armature = config.joint_cfg.armature

        if config.shape_cfg is not None:
            builder.default_shape_cfg = config.shape_cfg

        if config.usd_path is not None:
            builder.add_usd(
                config.usd_path,
                xform=config.transform,
                collapse_fixed_joints=config.collapse_fixed_joints,
                enable_self_collisions=config.enable_self_collisions,
                hide_collision_shapes=config.hide_collision_shapes,
                skip_mesh_approximation=config.skip_mesh_approximation,
            )

            # Mesh approximation (optional)
            if config.mesh_approximation is not None:
                builder.approximate_meshes(config.mesh_approximation)

            # Apply joint params
            if config.joint_target_ke_map or config.joint_target_kd_map:
                apply_joint_params_by_pattern(
                    builder,
                    ke_map=config.joint_target_ke_map,
                    kd_map=config.joint_target_kd_map,
                    armature_map=config.joint_armature_map
                )
            elif config.joint_cfg is not None:
                for i in range(builder.joint_dof_count):
                    builder.joint_target_ke[i] = config.joint_cfg.target_ke
                    builder.joint_target_kd[i] = config.joint_cfg.target_kd

        # Create sites
        self._create_sites(builder, config)

    def _create_sites(self, builder: newton.ModelBuilder, config: NewtonEntityConfig) -> None:
        prefix = config.body_label_prefix

        def _resolve(name: str) -> str:
            return f"{prefix}/{name}" if prefix and "/" not in name else name

        if config.sites:
            for site_name, body_name in config.sites.items():
                body_idx = self._find_body_by_name(builder, _resolve(body_name))
                if body_idx is not None:
                    builder.add_site(body_idx, label=site_name)
                else:
                    raise ValueError(f"Body '{body_name}' not found for site '{site_name}'")

        if config.contact_shapes:
            for shape_name, (body_name, local_pos) in config.contact_shapes.items():
                body_idx = self._find_body_by_name(builder, _resolve(body_name))
                if body_idx is not None:
                    xform = wp.transform(wp.vec3(*local_pos), wp.quat_identity())
                    builder.add_shape_sphere(body_idx, xform=xform, radius=0.02, label=shape_name)
                else:
                    raise ValueError(f"Body '{body_name}' not found for contact shape '{shape_name}'")

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
            config = self.entities[entity_name]["config"]
            if config.entity_type == "ground_plane":
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

        assert (ke > 0).all(), f"Zero/negative position gains: {ke}"
        assert (kd > 0).all(), f"Zero/negative velocity gains: {kd}"
        assert np.allclose(ke, ke_bias), f"Position gain/bias mismatch"

        print(f"✓ MuJoCo actuators validated: {num_actuators} joints")
        print(f"  ke range: [{ke.min():.2f}, {ke.max():.2f}]")
        print(f"  kd range: [{kd.min():.2f}, {kd.max():.2f}]")

    def _set_kinematic_trees(self) -> None:
        """Build kinematic trees for entities with URDF."""
        for entity_name, entity_info in self.entities.items():
            config = entity_info["config"]
            if config.urdf_path is not None:
                self.trees[entity_name] = KinematicTree(urdf_path=config.urdf_path)

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
                include_total=config.include_total,
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
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self._sim_dt)

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
