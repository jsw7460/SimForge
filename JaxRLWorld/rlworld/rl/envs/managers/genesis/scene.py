from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import genesis as gs
import torch

from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg
from rlworld.rl.configs.scene.terrain_config import TerrainCfg
from rlworld.rl.configs.scene.unified_entity_config import (
    EntityCfg,
    GenesisEntityCfg,
    GroundPlaneCfg,
)
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.indexing import ArticulationIndexing
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.managers.common.canonical_joint_order import filter_canonical_to_actuated
from rlworld.rl.envs.managers.common.scene_helpers import build_kinematic_trees
from rlworld.rl.envs.managers.genesis.terrain_importer import import_terrain_genesis
from rlworld.rl.utils import entity_utils, string as string_utils

if TYPE_CHECKING:
    # ``genesis.engine.entities`` / ``genesis.engine.sensors`` evaluate
    # ``gs.qd_float`` at module load (i.e. need ``genesis.init()``), so they
    # are imported for type hints only — not on the runtime import path.
    from genesis.engine.entities import RigidEntity
    from genesis.engine.sensors.base_sensor import Sensor

    from rlworld.rl.envs import World


def _canonical_joint_order_genesis(entity: RigidEntity) -> list[str]:
    """Canonical joint name list — DFS walk of ``entity.links`` with
    siblings sorted alphabetically by bare link name at each node,
    collecting each link's inbound joint(s) when visited.

    Genesis enumerates bodies in BFS-by-depth internally, which gave a
    different action order than Newton/mjlab for humanoids that mix arm
    and leg chains at the same depth. This DFS-with-sorted-siblings walk
    pins the canonical order solely to the kinematic structure + names,
    independent of any sim's parser order.
    """
    links = list(entity.links)
    if not links:
        return []
    by_idx = {link.idx: link for link in links}
    roots: list = []
    children: dict[int, list] = {}
    for link in links:
        p = link.parent_idx
        if p == -1 or p not in by_idx:
            roots.append(link)
        else:
            children.setdefault(p, []).append(link)
    # Sort siblings alphabetically at every level (roots included).
    roots.sort(key=lambda lk: lk.name)
    for k in children:
        children[k].sort(key=lambda lk: lk.name)

    out: list[str] = []
    stack = list(reversed(roots))
    while stack:
        link = stack.pop()
        # ``link.joints`` are the joints connecting this link to its parent.
        # Sort by name for determinism when a body has multiple inbound joints
        # (rare — typically zero or one).
        for joint in sorted(link.joints, key=lambda j: j.name):
            if joint.n_dofs > 0:
                out.append(joint.name)
        kids = children.get(link.idx, [])
        for kid in reversed(kids):
            stack.append(kid)
    return out


@dataclass
class SceneManagerConfig:
    """Configuration for scene creation"""

    sim_options: gs.options.SimOptions
    viewer_options: gs.options.ViewerOptions
    vis_options: gs.options.VisOptions
    rigid_options: gs.options.RigidOptions
    entities: dict[str, EntityCfg | GroundPlaneCfg | TerrainCfg]
    sensors: list[SensorConfig] | None
    env_spacing: tuple
    show_viewer: bool


class SceneManager(BaseManager):
    """Manages scene creation and configuration"""

    def __init__(self, env: World, config: SceneManagerConfig):
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
        self.horizontal_scale = None
        self.vertical_scale = None

        # Generated terrain (TerrainData) when a TerrainCfg generator is
        # used; None for flat ground. Read by the out-of-bounds termination.
        self._terrain_data = None

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
        _, names = entity_utils.find_links(self.entities[entity_name], body_names, preserve_order=True)
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

            if isinstance(cfg, TerrainCfg):
                entity, terrain_data = import_terrain_genesis(self.scene, cfg)
                if terrain_data is not None:
                    self._set_terrain_scaffolding(terrain_data)
            elif isinstance(cfg, GroundPlaneCfg):
                morph = gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
                entity = self.scene.add_entity(morph=morph)
            else:
                # GenesisEntityCfg-specific fields
                if isinstance(cfg, GenesisEntityCfg):
                    convexify = cfg.convexify
                    surface = cfg.surface
                    visualize = cfg.visualize_contact
                else:
                    convexify = False
                    surface = None
                    visualize = False

                mjcf_path = getattr(cfg, "mjcf_path", None)
                if mjcf_path:
                    mjcf_kwargs = {
                        "file": mjcf_path,
                        "convexify": convexify,
                        "batch_fixed_verts": True,
                    }
                    if cfg.init_state.pos != (0.0, 0.0, 0.0):
                        mjcf_kwargs["pos"] = cfg.init_state.pos
                    morph = gs.morphs.MJCF(**mjcf_kwargs)
                else:
                    urdf_kwargs = {
                        "file": cfg.urdf_path,
                        "fixed": not cfg.floating,
                        "convexify": convexify,
                    }
                    if cfg.links_to_keep:
                        urdf_kwargs["links_to_keep"] = cfg.links_to_keep
                    morph = gs.morphs.URDF(**urdf_kwargs)

                entity = self.scene.add_entity(
                    morph=morph,
                    surface=surface,
                    visualize_contact=visualize,
                )
            self.entities[entity_name] = entity

    def _set_terrain_scaffolding(self, data) -> None:
        """Populate the terrain scaffolding from generated TerrainData.

        Fills the (previously unset) ``terrain_info`` fields so height-scan
        observations / the out-of-bounds termination can read the world
        height map and extent.
        """
        self._terrain_data = data
        self.height_map = data.heights_m
        self.horizontal_scale = data.horizontal_scale
        self.vertical_scale = data.vertical_scale
        lx, ly = data.size_xy
        self.land_x_range = (-lx / 2.0, lx / 2.0)
        self.land_y_range = (-ly / 2.0, ly / 2.0)

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
            sensor = sensor_config.create_sensor(scene=self.scene, entity=entity)

            sensor_class_name = sensor.__class__.__name__
            self.sensors[entity_name][link_name][sensor_class_name] = sensor

    def build_scene(self):
        self.scene.build(n_envs=self.env.num_envs, env_spacing=self.config.env_spacing, center_envs_at_origin=False)
        self._configure_robot_dynamics()
        self.env.vis_manager.inject_custom_context()

    def _set_kinematic_tree(self):
        def _resolve(name: str):
            cfg = self.config.entities.get(name)
            if cfg is None or isinstance(cfg, GroundPlaneCfg | TerrainCfg):
                return None
            mjcf_path = getattr(cfg, "mjcf_path", None)
            if mjcf_path:
                return ("mjcf_path", mjcf_path)
            urdf_path = getattr(cfg, "urdf_path", None)
            if urdf_path:
                return ("urdf", urdf_path)
            return None

        self.trees = build_kinematic_trees(self.entities.keys(), _resolve)

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
        for entity_name, entity in self.entities.items():
            cfg = self.config.entities.get(entity_name)
            if cfg is None or isinstance(cfg, GroundPlaneCfg | TerrainCfg):
                continue

            for act_cfg in cfg.articulation.actuators:
                name_keys = list(act_cfg.target_names_expr)
                dof_ids, joint_names = entity_utils.find_dofs(entity=entity, name_keys=name_keys)
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
                            _, _, vals = string_utils.resolve_matching_names_values(act_cfg.stiffness, sub_names)
                            entity.set_dofs_kp(vals, sub_ids)
                    elif act_cfg.stiffness is not None and act_cfg.stiffness > 0:
                        entity.set_dofs_kp([act_cfg.stiffness] * num_dofs, dof_ids)

                    # Damping — float or dict[regex, float]
                    if isinstance(act_cfg.damping, dict):
                        sub_ids, sub_names = entity_utils.find_dofs(
                            entity=entity, name_keys=list(act_cfg.damping.keys())
                        )
                        if sub_ids:
                            _, _, vals = string_utils.resolve_matching_names_values(act_cfg.damping, sub_names)
                            entity.set_dofs_kv(vals, sub_ids)
                    elif act_cfg.damping is not None and act_cfg.damping > 0:
                        entity.set_dofs_kv([act_cfg.damping] * num_dofs, dof_ids)

                # Armature — float or dict[regex, float]
                if isinstance(act_cfg.armature, dict):
                    sub_ids, sub_names = entity_utils.find_dofs(entity=entity, name_keys=list(act_cfg.armature.keys()))
                    if sub_ids:
                        _, _, vals = string_utils.resolve_matching_names_values(act_cfg.armature, sub_names)
                        entity.set_dofs_armature(vals, sub_ids)
                elif isinstance(act_cfg.armature, int | float) and act_cfg.armature > 0:
                    entity.set_dofs_armature([act_cfg.armature] * num_dofs, dof_ids)

                # Effort limit — symmetric force range [-limit, +limit]. Applied
                # to both implicit and explicit actuators so Genesis enforces the
                # same motor saturation as Newton/Mjlab. For explicit actuators
                # this is redundant with the Python-side _clip_effort in
                # IdealPDActuator but keeps cross-sim behavior identical when
                # URDF-declared limits differ from cfg.
                if isinstance(act_cfg.effort_limit, dict):
                    sub_ids, sub_names = entity_utils.find_dofs(
                        entity=entity, name_keys=list(act_cfg.effort_limit.keys())
                    )
                    if sub_ids:
                        _, _, vals = string_utils.resolve_matching_names_values(act_cfg.effort_limit, sub_names)
                        neg = [-float(v) for v in vals]
                        pos = [float(v) for v in vals]
                        entity.set_dofs_force_range(neg, pos, sub_ids)
                elif act_cfg.effort_limit is not None and act_cfg.effort_limit > 0:
                    limit = float(act_cfg.effort_limit)
                    entity.set_dofs_force_range(
                        [-limit] * num_dofs,
                        [limit] * num_dofs,
                        dof_ids,
                    )

                # Friction loss — static joint friction [N*m]. Scalar only.
                if act_cfg.frictionloss > 0:
                    entity.set_dofs_frictionloss(
                        [float(act_cfg.frictionloss)] * num_dofs,
                        dof_ids,
                    )

    def build_articulation_indexing(
        self,
        actuated_dof_names: list[str],
        entity_name: str = "robot",
    ):
        """Build ArticulationIndexing for the given entity in canonical joint order.

        Joint order is computed by a DFS walk of the kinematic body tree
        (siblings sorted alphabetically by bare body name at each level),
        emitting each body's inbound joint when visited. This order depends
        only on the kinematic structure + joint/body names, so it is identical
        across simulators when the same robot is loaded — regardless of how
        Genesis / Newton / mjlab happen to enumerate bodies internally
        (Genesis uses BFS by depth, Newton/mjlab follow MJCF declaration
        order, etc.). The user's ``actuated_dof_names`` regexes filter that
        canonical list while preserving canonical order.

        Args:
            actuated_dof_names: Regex patterns for actuated joints.
            entity_name: Which entity to index.

        Returns:
            ArticulationIndexing with canonical ↔ simulator mappings.
        """
        entity = self.entities[entity_name]
        canonical_names = _canonical_joint_order_genesis(entity)
        matched_names, _ = filter_canonical_to_actuated(canonical_names, actuated_dof_names)
        # An entity with zero actuated joints (e.g. a free-flying drone)
        # is a legitimate case — return an empty ArticulationIndexing
        # so the action manager can still operate via term-based actions.
        if not matched_names:
            empty_long = torch.zeros(0, device=self.env.device, dtype=torch.long)
            empty_float = torch.zeros(0, device=self.env.device, dtype=torch.float32)
            return ArticulationIndexing(
                joint_names=(),
                sim_indices=empty_long,
                sim_to_canonical=empty_long.clone(),
                joint_limits_lower=empty_float,
                joint_limits_upper=empty_float.clone(),
            )

        # Resolve each matched joint name to its Genesis-local DOF id(s).
        dof_ids: list[int] = []
        for name in matched_names:
            joint = entity.get_joint(name)
            ids = joint.dofs_idx_local
            if hasattr(ids, "__iter__"):
                dof_ids.extend(int(i) for i in ids)
            else:
                dof_ids.append(int(ids))
        sim_indices = torch.tensor(dof_ids, device=self.env.device)

        # sim_to_canonical: identity since RobotData indexes by sim_indices.
        sim_to_canonical = torch.arange(len(dof_ids), device=self.env.device)

        # Joint limits in canonical order.
        dof_lower, dof_upper = entity.get_dofs_limit(dofs_idx_local=sim_indices)

        return ArticulationIndexing(
            joint_names=tuple(matched_names),
            sim_indices=sim_indices,
            sim_to_canonical=sim_to_canonical,
            joint_limits_lower=dof_lower[0],
            joint_limits_upper=dof_upper[0],
        )

    def step(self):
        self.scene.step()

    def get_local_height_map(self) -> torch.Tensor | None:
        """Get current height map"""
        return self.local_height_map

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Genesis manages scene state internally; nothing to do here
        # for partial reset. Kept for cross-sim API symmetry.
        pass
