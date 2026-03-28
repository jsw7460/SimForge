from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from rlworld.rl.configs.robots.kinematic_tree import KinematicTree
from rlworld.rl.configs.scene.unified_entity_config import EntityCfg, GroundPlaneCfg, ActuatorCfg
from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from mjlab.scene import Scene
    from mjlab.sim import Simulation, SimulationCfg
    from mjlab.entity import Entity


@dataclass
class MjlabSceneManagerConfig:
    """Configuration for MuJoCo/mjlab scene management.

    This config wraps mjlab's Scene and Simulation configuration.

    Example:
        from mjlab.scene import SceneCfg
        from mjlab.sim import SimulationCfg

        config = MjlabSceneManagerConfig(
            mjlab_scene_cfg=SceneCfg(
                num_envs=4096,
                entities={"robot": robot_entity_cfg},
            ),
            mjlab_sim_cfg=SimulationCfg(),
            device="cuda:0",
        )
    """
    mjlab_scene_cfg: Any = None  # mjlab.SceneCfg (optional if unified_entities is set)
    mjlab_sim_cfg: Any = None  # mjlab.SimulationCfg (optional, uses default if None)
    device: str = "cuda:0"
    robot_entity_name: str = "robot"
    unified_entities: dict[str, EntityCfg | GroundPlaneCfg] | None = None
    """When set, entities are converted to mjlab EntityCfg at build time."""


class MjlabSceneManager(BaseManager):
    """Manages mjlab Scene and Simulation lifecycle.

    This manager wraps mjlab's Scene and Simulation classes to provide
    a rlworld-compatible interface for MuJoCo-based environments.

    The scene is built in the following order:
    1. Create Scene from SceneCfg (builds MjSpec)
    2. Compile Scene to get MjModel
    3. Create Simulation with MjModel
    4. Initialize Scene with Simulation data

    Example:
        scene_manager = MjlabSceneManager(env, config)
        scene_manager.build_scene()

        # In simulation loop:
        scene_manager.step()
    """

    def __init__(self, env: "World", config: MjlabSceneManagerConfig):
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
        """Build the scene and simulation from config."""
        from mjlab.scene import Scene
        from mjlab.sim import Simulation, SimulationCfg

        # Convert unified entities to mjlab SceneCfg if provided
        if self.config.unified_entities is not None:
            self._apply_unified_entities()

        # Create scene
        self._scene = Scene(self.config.mjlab_scene_cfg, device=self.config.device)

        # Compile to get MjModel
        mj_model = self._scene.compile()

        # Create simulation config if not provided
        sim_cfg = self.config.mjlab_sim_cfg
        if sim_cfg is None:
            sim_cfg = SimulationCfg()

        # Update physics dt from config
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

    def _apply_unified_entities(self) -> None:
        """Convert unified EntityCfg dict into mjlab SceneCfg.entities."""
        from mjlab.entity import EntityCfg as MjlabEntityCfg, EntityArticulationInfoCfg
        from mjlab.actuator.builtin_actuator import (
            BuiltinPositionActuatorCfg,
            BuiltinMotorActuatorCfg,
        )
        from mjlab.terrains import TerrainEntityCfg

        scene_cfg = self.config.mjlab_scene_cfg
        mjlab_entities: dict[str, Any] = {}

        for entity_name, cfg in self.config.unified_entities.items():
            if isinstance(cfg, GroundPlaneCfg):
                # Terrain handled separately via SceneCfg.terrain
                continue

            # Convert ActuatorCfg → mjlab actuator configs
            mjlab_actuators = []
            for act_cfg in cfg.articulation.actuators:
                if act_cfg.control_type == "motor":
                    mjlab_actuators.append(BuiltinMotorActuatorCfg(
                        target_names_expr=act_cfg.target_names_expr,
                        effort_limit=act_cfg.effort_limit or 1000.0,
                        armature=act_cfg.armature,
                        frictionloss=act_cfg.frictionloss,
                    ))
                else:
                    mjlab_actuators.append(BuiltinPositionActuatorCfg(
                        target_names_expr=act_cfg.target_names_expr,
                        stiffness=act_cfg.stiffness,
                        damping=act_cfg.damping,
                        effort_limit=act_cfg.effort_limit,
                        armature=act_cfg.armature,
                        frictionloss=act_cfg.frictionloss,
                    ))

            # Get the existing mjlab EntityCfg if available (from mujoco_options),
            # otherwise we need a spec_fn from mujoco_options
            mujoco_opts = cfg.mujoco_options
            if "entity_cfg" in mujoco_opts:
                # User provided a full mjlab EntityCfg — just swap actuators
                mjlab_cfg = mujoco_opts["entity_cfg"]
                if mjlab_actuators:
                    mjlab_cfg.articulation = EntityArticulationInfoCfg(
                        actuators=tuple(mjlab_actuators),
                        soft_joint_pos_limit_factor=cfg.articulation.soft_joint_pos_limit_factor,
                    )
            elif "spec_fn" in mujoco_opts:
                # User provided a spec_fn
                init_state = MjlabEntityCfg.InitialStateCfg(
                    pos=cfg.init_state.pos,
                    rot=cfg.init_state.rot,
                    joint_pos=cfg.init_state.joint_pos or None,
                    joint_vel=cfg.init_state.joint_vel,
                )
                mjlab_cfg = MjlabEntityCfg(
                    init_state=init_state,
                    spec_fn=mujoco_opts["spec_fn"],
                    articulation=EntityArticulationInfoCfg(
                        actuators=tuple(mjlab_actuators),
                        soft_joint_pos_limit_factor=cfg.articulation.soft_joint_pos_limit_factor,
                    ) if mjlab_actuators else None,
                    collisions=mujoco_opts.get("collisions", ()),
                )
            else:
                raise ValueError(
                    f"MuJoCo entity '{entity_name}' requires either "
                    "'entity_cfg' or 'spec_fn' in mujoco_options"
                )

            mjlab_entities[entity_name] = mjlab_cfg

        # Update the SceneCfg with converted entities
        if mjlab_entities:
            scene_cfg.entities.update(mjlab_entities)

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
        _, matched_names = entity.find_bodies(body_names)
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
