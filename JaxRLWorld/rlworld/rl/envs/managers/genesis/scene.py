from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.engine.entities import RigidEntity
from genesis.engine.sensors.base_sensor import Sensor
from rlworld.rl.configs.robots.kinematic_tree import KinematicTree
from rlworld.rl.configs.scene.unified_entity_config import (
    EntityCfg, GenesisEntityCfg, GroundPlaneCfg,
)
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import entity_utils
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class SceneManagerConfig:
    """Configuration for scene creation"""
    sim_options: gs.options.SimOptions
    viewer_options: gs.options.ViewerOptions
    vis_options: gs.options.VisOptions
    rigid_options: gs.options.RigidOptions
    entities: dict[str, EntityCfg | GroundPlaneCfg]
    sensors: list[SensorConfig] | None
    env_spacing: tuple
    show_viewer: bool


class SceneManager(BaseManager):
    """Manages scene creation and configuration"""

    def __init__(self, env: "World", config: SceneManagerConfig):
        BaseManager.__init__(self, env=env)
        self.config = config
        self.scene = None
        self.entities: dict[str, RigidEntity] = defaultdict()
        self.sensors: dict[str, dict[str, dict[str, Sensor]]] = defaultdict(lambda: defaultdict(dict))

        self.trees: dict = {}

        self.land_x_range = None
        self.land_y_range = None

        # Height map storage
        self.coords_x = None
        self.coords_y = None
        self.coords_x_clamped = None
        self.coords_y_clamped = None
        self.height_map = None  # World map ; [world_size, world_size]
        self.local_height_map = None  # Map around the robot; [n_envs, map_size, map_size]

    def __getattr__(self, item) -> RigidEntity:
        return self.entities[item]

    def __getitem__(self, item) -> RigidEntity:
        return self.entities[item]

    @property
    def terrain_info(self):
        return {
            "x_range": self.land_x_range,
            "y_range": self.land_y_range,
            "height_field": self.height_map,
            "horizontal_scale": self.horizontal_scale,
            "vertical_scale": self.vertical_scale,
        }

    def find_body_names(self, body_names: list[str], entity_name: str = "robot"):

        _, names = entity_utils.find_links(self.entities[entity_name], body_names)
        return names

    def register_entities(self) -> None:
        """Build complete scene with all components"""
        self._create_scene()
        self._add_entities()
        self._add_sensors()
        self._set_kinematic_tree()
        self.env.vis_manager._setup_visualization_cameras()

    def _add_entities(self):
        """Add entities from dict[str, EntityCfg/GenesisEntityCfg/GroundPlaneCfg]."""
        for entity_name, cfg in self.config.entities.items():
            if entity_name in self.entities:
                raise ValueError(f"Entity '{entity_name}' is already registered")

            if isinstance(cfg, GroundPlaneCfg):
                morph = gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
                entity = self.scene.add_entity(morph=morph)
            else:
                morph_kwargs = {"file": cfg.urdf_path, "fixed": not cfg.floating}
                if cfg.links_to_keep:
                    morph_kwargs["links_to_keep"] = cfg.links_to_keep

                # GenesisEntityCfg-specific fields
                if isinstance(cfg, GenesisEntityCfg):
                    morph_kwargs["convexify"] = cfg.convexify
                    surface = cfg.surface
                    visualize = cfg.visualize_contact
                else:
                    surface = None
                    visualize = False

                morph = gs.morphs.URDF(**morph_kwargs)
                entity = self.scene.add_entity(
                    morph=morph, surface=surface, visualize_contact=visualize,
                )
            self.entities[entity_name] = entity

    def _add_sensors(self):
        sensor_configs = self.config.sensors

        if not sensor_configs:
            return

        for sensor_config in sensor_configs:
            entity_name = sensor_config.entity_name
            link_name = sensor_config.link_name
            if entity_name not in self.entities:
                print(f"Entity {entity_name} not found for sensor. Skipping.")
                continue

            entity = self.entities[entity_name]
            sensor = sensor_config.create_sensor(
                scene=self.scene,
                entity=entity
            )

            sensor_class_name = sensor.__class__.__name__
            self.sensors[entity_name][link_name][sensor_class_name] = sensor

    def build_scene(self):
        self.scene.build(
            n_envs=self.env.num_envs,
            env_spacing=self.config.env_spacing,
            center_envs_at_origin=False
        )
        self._configure_robot_dynamics()
        self.env.vis_manager.inject_custom_context()

    def _set_kinematic_tree(self):
        for entity_name in self.entities:
            cfg = self.config.entities.get(entity_name)
            if cfg is None or isinstance(cfg, GroundPlaneCfg) or cfg.urdf_path is None:
                continue
            self.trees[entity_name] = KinematicTree(cfg.urdf_path)

    def _create_scene(self) -> None:
        """Initialize scene with basic settings"""
        self.scene = gs.Scene(
            sim_options=self.config.sim_options,
            viewer_options=self.config.viewer_options,
            vis_options=self.config.vis_options,
            rigid_options=self.config.rigid_options,
            show_viewer=self.config.show_viewer,
        )

    def _configure_robot_dynamics(self) -> None:
        """Apply gains/armature from ArticulationCfg actuators.

        For **implicit** actuators, we set the simulator's PD gains (Kp/Kd)
        so the simulator drives the joints internally.

        For **explicit** actuators (IdealPD, DelayedPD, LSTM, etc.), the
        simulator's PD gains are still set here but are effectively unused:
        Genesis switches a joint to force mode when ``control_dofs_force()``
        is called (the last-called control mode wins), so the Kp/Kd values
        have no effect once force mode is active.

        This differs from Newton, where PD forces are *always* summed with
        ``joint_f`` regardless of calling order, requiring explicit ke=0/kd=0
        to disable the internal PD.  (See ``_load_urdf_entity`` in
        ``newton/scene.py`` for that handling.)
        """
        from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg

        for entity_name, entity in self.entities.items():
            cfg = self.config.entities.get(entity_name)
            if cfg is None or isinstance(cfg, GroundPlaneCfg):
                continue

            for act_cfg in cfg.articulation.actuators:
                name_keys = list(act_cfg.target_names_expr)
                dof_ids, joint_names = entity_utils.find_dofs(
                    entity=entity, name_keys=name_keys
                )
                if not dof_ids:
                    continue

                num_dofs = len(dof_ids)

                # Only set Kp/Kd for implicit actuators (simulator PD)
                if isinstance(act_cfg, ImplicitActuatorCfg):
                    # Stiffness — float or dict[regex, float]
                    if isinstance(act_cfg.stiffness, dict):
                        sub_ids, sub_names = entity_utils.find_dofs(
                            entity=entity, name_keys=list(act_cfg.stiffness.keys())
                        )
                        if sub_ids:
                            _, _, vals = string_utils.resolve_matching_names_values(
                                act_cfg.stiffness, sub_names
                            )
                            entity.set_dofs_kp(vals, sub_ids)
                    elif act_cfg.stiffness is not None and act_cfg.stiffness > 0:
                        entity.set_dofs_kp([act_cfg.stiffness] * num_dofs, dof_ids)

                    # Damping — float or dict[regex, float]
                    if isinstance(act_cfg.damping, dict):
                        sub_ids, sub_names = entity_utils.find_dofs(
                            entity=entity, name_keys=list(act_cfg.damping.keys())
                        )
                        if sub_ids:
                            _, _, vals = string_utils.resolve_matching_names_values(
                                act_cfg.damping, sub_names
                            )
                            entity.set_dofs_kv(vals, sub_ids)
                    elif act_cfg.damping is not None and act_cfg.damping > 0:
                        entity.set_dofs_kv([act_cfg.damping] * num_dofs, dof_ids)

                # Armature — float or dict[regex, float]
                if isinstance(act_cfg.armature, dict):
                    sub_ids, sub_names = entity_utils.find_dofs(
                        entity=entity, name_keys=list(act_cfg.armature.keys())
                    )
                    if sub_ids:
                        _, _, vals = string_utils.resolve_matching_names_values(
                            act_cfg.armature, sub_names
                        )
                        entity.set_dofs_armature(vals, sub_ids)
                elif isinstance(act_cfg.armature, (int, float)) and act_cfg.armature > 0:
                    entity.set_dofs_armature([act_cfg.armature] * num_dofs, dof_ids)

    def step(self):
        self.scene.step()

    def get_local_height_map(self) -> torch.Tensor | None:
        """Get current height map"""
        return self.local_height_map

    def reset(self, env_indices: torch.Tensor | None = None) -> None:
        pass
