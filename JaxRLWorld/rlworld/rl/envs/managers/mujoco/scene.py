from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from rlworld.rl.configs.robots.kinematic_tree import KinematicTree
from rlworld.rl.configs.scene.unified_entity_config import (
    EntityCfg, MujocoEntityCfg, GroundPlaneCfg,
)
from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from mjlab.scene import Scene
    from mjlab.sim import Simulation, SimulationCfg
    from mjlab.entity import Entity


@dataclass
class MujocoSceneManagerConfig:
    """Internal config consumed by MujocoSceneManager.

    Populated from MujocoSceneConfig by mjlab_env at build time.
    Users should not construct this directly — use MujocoSceneConfig
    in presets instead.
    """
    device: str = "cuda:0"
    robot_entity_name: str = "robot"
    num_envs: int = 4096
    env_spacing: float = 2.0
    physics_dt: float = 0.002

    # Entities — unified EntityCfg dict (auto-converted to mjlab)
    entities: dict[str, EntityCfg | GroundPlaneCfg] | None = None

    # Sensors — mjlab sensor configs (passed through)
    sensors: tuple = ()

    # Terrain
    terrain_type: str = "plane"

    # Solver
    solver_iterations: int = 10
    solver_ls_iterations: int = 20
    ccd_iterations: int = 50
    nconmax: int = 35
    njmax: int = 1500
    contact_sensor_maxmatch: int = 64

    # Legacy — set by mjlab_env for backward compat
    mjlab_scene_cfg: Any = None
    mjlab_sim_cfg: Any = None
    unified_entities: Any = None


class MujocoSceneManager(BaseManager):
    """Manages mjlab Scene and Simulation lifecycle.

    This manager wraps mjlab's Scene and Simulation classes to provide
    a rlworld-compatible interface for MuJoCo-based environments.

    The scene is built in the following order:
    1. Create Scene from SceneCfg (builds MjSpec)
    2. Compile Scene to get MjModel
    3. Create Simulation with MjModel
    4. Initialize Scene with Simulation data

    Example:
        scene_manager = MujocoSceneManager(env, config)
        scene_manager.build_scene()

        # In simulation loop:
        scene_manager.step()
    """

    def __init__(self, env: "World", config: MujocoSceneManagerConfig):
        super().__init__(env)
        self.config = config

        # mjlab objects (initialized in build_scene)
        self._scene: "Scene" = None
        self._sim: "Simulation" = None

        # Kinematic trees for each entity
        self.trees: dict[str, KinematicTree] = {}

        # Timing
        self._physics_dt: float = 0.002  # Default, updated from sim config

    @property
    def scene(self) -> "Scene":
        """The mjlab Scene instance."""
        return self._scene

    @property
    def sim(self) -> "Simulation":
        """The mjlab Simulation instance."""
        return self._sim

    @property
    def model(self):
        """The mujoco-warp Model."""
        if self._sim is None:
            return None
        return self._sim.model

    @property
    def data(self):
        """The mujoco-warp Data."""
        if self._sim is None:
            return None
        return self._sim.data

    @property
    def mj_model(self):
        """The MuJoCo MjModel."""
        if self._sim is None:
            return None
        return self._sim.mj_model

    @property
    def physics_dt(self) -> float:
        """Physics timestep."""
        return self._physics_dt

    @property
    def robot(self) -> "Entity":
        """Get the main robot entity."""
        return self._scene[self.config.robot_entity_name]

    @property
    def entities(self) -> dict[str, "Entity"]:
        """Get all entities."""
        return self._scene.entities

    @property
    def sensors(self) -> dict[str, Any]:
        """Get all sensors."""
        return self._scene.sensors

    def build_scene(self) -> None:
        """Build the scene and simulation from config.

        Constructs mjlab SceneCfg and SimulationCfg internally from the
        rlworld config fields.  No mjlab imports needed at config level.
        """
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
        from mjlab.terrains import TerrainEntityCfg
        # Build mjlab SceneCfg
        if self.config.mjlab_scene_cfg is not None:
            # Legacy path — user provided mjlab SceneCfg directly
            scene_cfg = self.config.mjlab_scene_cfg
        else:
            scene_cfg = SceneCfg(
                num_envs=self.config.num_envs,
                env_spacing=self.config.env_spacing,
                terrain=TerrainEntityCfg(terrain_type=self.config.terrain_type),
                entities={},
                sensors=self.config.sensors,
            )
            self.config.mjlab_scene_cfg = scene_cfg

        # Convert unified entities → mjlab entities and merge
        if self.config.entities is not None:
            self._build_mjlab_entities()

        # Create scene
        self._scene = Scene(scene_cfg, device=self.config.device)

        # Compile to get MjModel
        mj_model = self._scene.compile()

        # Build SimulationCfg
        if self.config.mjlab_sim_cfg is not None:
            sim_cfg = self.config.mjlab_sim_cfg
        else:
            sim_cfg = SimulationCfg(
                nconmax=self.config.nconmax,
                njmax=self.config.njmax,
                mujoco=MujocoCfg(
                    timestep=self.config.physics_dt,
                    iterations=self.config.solver_iterations,
                    ls_iterations=self.config.solver_ls_iterations,
                    ccd_iterations=self.config.ccd_iterations,
                ),
                contact_sensor_maxmatch=self.config.contact_sensor_maxmatch,
            )

        # Update physics dt
        self._physics_dt = sim_cfg.mujoco.timestep

        # Create simulation
        self._sim = Simulation(
            num_envs=self._scene.num_envs,
            cfg=sim_cfg,
            model=mj_model,
            device=self.config.device,
        )

        # Initialize scene with simulation data
        self._scene.initialize(
            mj_model=self._sim.mj_model,
            model=self._sim.model,
            data=self._sim.data,
        )

        # Build kinematic trees for each entity
        self._set_kinematic_tree()

    def _build_mjlab_entities(self) -> None:
        """Convert unified entities to mjlab EntityCfg and merge into SceneCfg.

        Actuator type mapping (automatic):
          ImplicitActuatorCfg → BuiltinPositionActuatorCfg (simulator PD)
          Any other type      → BuiltinMotorActuatorCfg (direct torque)
        """
        from mjlab.entity import EntityCfg as MjlabEntityCfg, EntityArticulationInfoCfg
        from mjlab.actuator.builtin_actuator import (
            BuiltinPositionActuatorCfg,
            BuiltinMotorActuatorCfg,
        )
        from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg

        for entity_name, cfg in self.config.entities.items():
            if isinstance(cfg, GroundPlaneCfg):
                continue

            if not isinstance(cfg, MujocoEntityCfg):
                raise ValueError(
                    f"MuJoCo entity '{entity_name}' must be MujocoEntityCfg, "
                    f"got {type(cfg).__name__}"
                )
            if cfg.spec_fn is None:
                raise ValueError(
                    f"MuJoCo entity '{entity_name}' requires 'spec_fn'"
                )

            # Convert actuator configs → mjlab actuator configs.
            # When stiffness/damping/armature are dicts, we expand into one
            # mjlab actuator per regex key so each gets the correct value.
            mjlab_actuators = []
            for act_cfg in cfg.articulation.actuators:
                if isinstance(act_cfg, ImplicitActuatorCfg):
                    if isinstance(act_cfg.stiffness, dict):
                        # Expand: one BuiltinPositionActuator per gain key
                        stiff_dict = act_cfg.stiffness
                        damp_dict = act_cfg.damping if isinstance(act_cfg.damping, dict) else {}
                        arm_dict = act_cfg.armature if isinstance(act_cfg.armature, dict) else {}
                        for pattern, kp in stiff_dict.items():
                            kd = damp_dict.get(pattern, 0.0)
                            arm = arm_dict.get(pattern, 0.0)
                            mjlab_actuators.append(BuiltinPositionActuatorCfg(
                                target_names_expr=(pattern,),
                                stiffness=kp,
                                damping=kd,
                                effort_limit=act_cfg.effort_limit,
                                armature=arm,
                                frictionloss=act_cfg.frictionloss,
                            ))
                    else:
                        stiffness = act_cfg.stiffness if isinstance(act_cfg.stiffness, (int, float)) else 0.0
                        damping = act_cfg.damping if isinstance(act_cfg.damping, (int, float)) else 0.0
                        armature = act_cfg.armature if isinstance(act_cfg.armature, (int, float)) else 0.0
                        mjlab_actuators.append(BuiltinPositionActuatorCfg(
                            target_names_expr=act_cfg.target_names_expr,
                            stiffness=stiffness,
                            damping=damping,
                            effort_limit=act_cfg.effort_limit,
                            armature=armature,
                            frictionloss=act_cfg.frictionloss,
                        ))
                else:
                    # Explicit actuator (IdealPD, LSTM, etc.) → motor mode
                    if isinstance(act_cfg.armature, dict):
                        # Expand: one BuiltinMotorActuator per armature key
                        arm_dict = act_cfg.armature
                        for pattern, arm in arm_dict.items():
                            mjlab_actuators.append(BuiltinMotorActuatorCfg(
                                target_names_expr=(pattern,),
                                effort_limit=act_cfg.effort_limit or 1000.0,
                                armature=arm,
                                frictionloss=act_cfg.frictionloss,
                            ))
                    else:
                        armature = act_cfg.armature if isinstance(act_cfg.armature, (int, float)) else 0.0
                        mjlab_actuators.append(BuiltinMotorActuatorCfg(
                            target_names_expr=act_cfg.target_names_expr,
                            effort_limit=act_cfg.effort_limit or 1000.0,
                            armature=armature,
                            frictionloss=act_cfg.frictionloss,
                        ))

            articulation_info = EntityArticulationInfoCfg(
                actuators=tuple(mjlab_actuators),
                soft_joint_pos_limit_factor=cfg.articulation.soft_joint_pos_limit_factor,
            ) if mjlab_actuators else None

            # Build mjlab EntityCfg
            init_state = MjlabEntityCfg.InitialStateCfg(
                pos=cfg.init_state.pos,
                rot=cfg.init_state.rot,
                joint_pos=cfg.init_state.joint_pos or None,
                joint_vel=cfg.init_state.joint_vel,
            )
            spec_fn = cfg.spec_fn
            if isinstance(spec_fn, str):
                from rlworld.rl.utils.resolve import resolve_callable
                spec_fn = resolve_callable(spec_fn)

            mjlab_cfg = MjlabEntityCfg(
                init_state=init_state,
                spec_fn=spec_fn,
                articulation=articulation_info,
                collisions=cfg.collisions,
                sort_actuators=True,
            )

            self.config.mjlab_scene_cfg.entities[entity_name] = mjlab_cfg

    def build_articulation_indexing(
        self, actuated_dof_names: list[str], entity_name: str = "robot",
    ):
        """Build ArticulationIndexing for the given entity.

        Args:
            actuated_dof_names: Regex patterns for actuated joints.
            entity_name: Which entity to index.

        Returns:
            ArticulationIndexing with canonical ↔ simulator mappings.
        """
        from rlworld.rl.envs.indexing import ArticulationIndexing

        entity = self._scene[entity_name]

        if actuated_dof_names:
            indices, names = entity.find_joints(
                actuated_dof_names, preserve_order=True
            )
        else:
            names = list(entity.joint_names)
            indices = list(range(len(names)))

        sim_indices = torch.tensor(indices, device=self.config.device, dtype=torch.long)

        # sim_to_canonical: inverse permutation
        num_joints = len(indices)
        sim_to_canonical = torch.zeros(num_joints, device=self.config.device, dtype=torch.long)
        for canonical_i, sim_i in enumerate(sim_indices):
            sim_to_canonical[canonical_i] = canonical_i  # Will be used via sim_indices gather

        # Joint limits
        soft_limits = entity.data.soft_joint_pos_limits
        lower = soft_limits[0, sim_indices, 0]
        upper = soft_limits[0, sim_indices, 1]

        return ArticulationIndexing(
            joint_names=tuple(names),
            sim_indices=sim_indices,
            sim_to_canonical=sim_to_canonical,
            joint_limits_lower=lower,
            joint_limits_upper=upper,
        )

    def step(self) -> None:
        """Execute a single physics step."""
        self._sim.step()

    def forward(self) -> None:
        """Compute forward kinematics."""
        self._sim.forward()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset environments to initial state."""
        self._sim.reset(env_ids)
        self._scene.reset(env_ids)

    def update(self, dt: float) -> None:
        """Update scene (entities and sensors) after physics step."""
        self._scene.update(dt)

    def write_data_to_sim(self) -> None:
        """Write entity data to simulation."""
        self._scene.write_data_to_sim()

    def get_entity(self, entity_name: str) -> "Entity":
        """Get an entity by name."""
        return self._scene[entity_name]

    def get_sensor(self, sensor_name: str) -> Any:
        """Get a sensor by name."""
        return self._scene.sensors.get(sensor_name)

    def find_body_names(self, body_names: list[str], entity_name: str = "robot") -> list[str]:
        """Find body names matching patterns (for GaitManager compatibility).

        Args:
            body_names: List of body name patterns to match.
            entity_name: Name of the entity to search in.

        Returns:
            List of matched body names.
        """
        entity = self.get_entity(entity_name)
        # Use mjlab's find_bodies API
        _, matched_names = entity.find_bodies(body_names, preserve_order=True)
        return matched_names

    def _set_kinematic_tree(self) -> None:
        """Build kinematic trees for all entities from their MjSpec XML."""
        for entity_name, entity in self._scene.entities.items():
            try:
                # Get the XML representation from entity's spec
                xml_content = entity.spec.to_xml()

                # Write to a temporary file for KinematicTree to parse
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.xml', delete=False
                ) as tmp_file:
                    tmp_file.write(xml_content)
                    tmp_path = Path(tmp_file.name)

                try:
                    # Create KinematicTree from MJCF
                    tree = KinematicTree(mjcf_path=str(tmp_path))
                    self.trees[entity_name] = tree
                finally:
                    # Clean up temp file
                    tmp_path.unlink(missing_ok=True)

            except Exception as e:
                # Log warning but don't fail - tree is optional
                import warnings
                warnings.warn(
                    f"Could not build kinematic tree for entity '{entity_name}': {e}"
                )
