from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity
from genesis.engine.sensors.base_sensor import Sensor

from rlworld.rl.configs.scene import EntityConfig
from rlworld.rl.configs.scene.unified_entity_config import (
    EntityCfg, GenesisEntityCfg, GroundPlaneCfg,
)
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import entity_utils
from rlworld.rl.utils import string as string_utils

from rlworld.rl.configs.robots.kinematic_tree import KinematicTree

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class SceneManagerConfig:
    """Configuration for scene creation"""
    sim_options: gs.options.SimOptions
    viewer_options: gs.options.ViewerOptions
    vis_options: gs.options.VisOptions
    rigid_options: gs.options.RigidOptions
    entities: list[EntityConfig] | dict[str, EntityCfg | GroundPlaneCfg]
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
        if isinstance(self.config.entities, dict):
            self._add_entities_from_unified_cfg()
        else:
            self._add_entities_from_legacy_cfg()

    def _add_entities_from_legacy_cfg(self):
        """Add entities from legacy list[EntityConfig]."""
        for entity_config in self.config.entities:
            entity_name = entity_config.entity_name
            if entity_name in self.entities:
                raise ValueError(f"Entity '{entity_name}' is already registered")
            entity = self.scene.add_entity(
                morph=entity_config.morph,
                surface=entity_config.surface,
                visualize_contact=entity_config.visualize_contact
            )
            self.entities[entity_name] = entity

    def _add_entities_from_unified_cfg(self):
        """Add entities from unified dict[str, EntityCfg/GenesisEntityCfg/GroundPlaneCfg]."""
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
        for entity_name, entity in self.entities.items():
            if isinstance(self.config.entities, dict):
                cfg = self.config.entities.get(entity_name)
                if cfg is None or isinstance(cfg, GroundPlaneCfg) or cfg.urdf_path is None:
                    continue
                urdf_path = cfg.urdf_path
            else:
                urdf_path = self.entities["robot"].morph.file
            urdf_tree = KinematicTree(urdf_path)
            self.trees[entity_name] = urdf_tree

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
        """Configure robot dynamic properties from entity configs."""
        if isinstance(self.config.entities, dict):
            self._configure_dynamics_from_unified_cfg()
        else:
            self._configure_dynamics_from_legacy_cfg()

    def _configure_dynamics_from_legacy_cfg(self) -> None:
        """Apply gains/armature from legacy list[EntityConfig]."""
        for entity_name, entity in self.entities.items():
            entity_config = next(
                (cfg for cfg in self.config.entities if cfg.entity_name == entity_name), None
            )
            if entity_config is None:
                continue

            if entity_config.p_gain is not None:
                dof_ids, joint_names = entity_utils.find_dofs(
                    entity=entity, name_keys=list(entity_config.p_gain.keys())
                )
                _, _, p_values = string_utils.resolve_matching_names_values(
                    entity_config.p_gain, joint_names
                )
                entity.set_dofs_kp(p_values, dof_ids)

            if entity_config.d_gain is not None:
                dof_ids, joint_names = entity_utils.find_dofs(
                    entity=entity, name_keys=list(entity_config.d_gain.keys())
                )
                _, _, d_values = string_utils.resolve_matching_names_values(
                    entity_config.d_gain, joint_names
                )
                entity.set_dofs_kv(d_values, dof_ids)

            if entity_config.armature is not None:
                dof_ids, joint_names = entity_utils.find_dofs(
                    entity=entity, name_keys=list(entity_config.armature.keys())
                )
                _, _, armature_values = string_utils.resolve_matching_names_values(
                    entity_config.armature, joint_names
                )
                entity.set_dofs_armature(armature_values, dof_ids)

    def _configure_dynamics_from_unified_cfg(self) -> None:
        """Apply gains/armature from unified dict[str, EntityCfg]."""
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

                # Stiffness (Kp)
                if act_cfg.stiffness > 0:
                    entity.set_dofs_kp([act_cfg.stiffness] * num_dofs, dof_ids)

                # Damping (Kd)
                if act_cfg.damping > 0:
                    entity.set_dofs_kv([act_cfg.damping] * num_dofs, dof_ids)

                # Armature
                if act_cfg.armature > 0:
                    entity.set_dofs_armature([act_cfg.armature] * num_dofs, dof_ids)

    def step(self):
        self.scene.step()

    def get_local_height_map(self) -> torch.Tensor | None:
        """Get current height map"""
        return self.local_height_map

    def reset(self, env_indices: torch.Tensor | None = None) -> None:
        pass
