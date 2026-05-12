"""MuJoCo/mjlab environment for rlworld.

This module provides MujocoEnv, which wraps mjlab's Scene and Simulation
while following rlworld's World interface and manager pattern.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg as _MjlabSceneEntityCfg

from rlworld.rl.configs import CommandConfig, CurriculumManagerConfig, EventConfig, RewardConfig
from rlworld.rl.configs.common_config_classes import VisualizationConfig
from rlworld.rl.configs.mujoco_config_classes import (
    MujocoActionConfig,
    MujocoEnvConfig,
    MujocoObservationConfig,
    MujocoSceneConfig,
)
from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.managers.registry import ManagerRegistry
from rlworld.rl.envs.world import World
from rlworld.rl.utils import set_seed


class MujocoEnv(World):
    """MuJoCo/mjlab environment following rlworld's World interface.

    This environment wraps mjlab's Scene and Simulation while providing
    the standard rlworld manager-based architecture.

    Example:
        from mjlab.scene import SceneCfg
        from mjlab.entity import EntityCfg

        scene_cfg = SceneCfg(
            num_envs=4096,
            entities={"robot": robot_entity_cfg},
        )

        mujoco_scene_cfg = MujocoSceneConfig(
            mjlab_scene_cfg=scene_cfg,
        )

        env = MujocoEnv(
            num_envs=4096,
            env_cfg=env_cfg,
            scene_cfg=mujoco_scene_cfg,
            ...
        )
    """

    sim_name: str = "Mujoco"
    sim_type: str = "mujoco"

    def __init__(
        self,
        num_envs: int,
        env_cfg: MujocoEnvConfig,
        scene_cfg: MujocoSceneConfig,
        visualization_cfg: VisualizationConfig,
        obs_cfg: MujocoObservationConfig,
        act_cfg: MujocoActionConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        event_cfg: EventConfig,
        curriculum_cfg: CurriculumManagerConfig,
    ):
        set_seed(env_cfg.seed)
        super().__init__()

        self.seed = env_cfg.seed
        self.num_envs = num_envs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Store high-level configs
        self.env_cfg = env_cfg
        self.scene_cfg = scene_cfg
        self.visualization_cfg = visualization_cfg
        self.obs_cfg = obs_cfg
        self.act_cfg = act_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.event_cfg = event_cfg
        self.curriculum_cfg = curriculum_cfg

        # Timing (will be updated after scene is built)
        self.decimation = env_cfg.decimation
        self.physics_dt = scene_cfg.physics_dt
        self.control_dt = self.physics_dt * self.decimation

        # Initialize buffers
        self._init_buffers()

        # Setup environment
        self._setup_environment()

    @property
    def robot(self):
        """Get the main robot entity."""
        return self.scene_manager.robot

    @property
    def robot_data(self):
        return self.get_robot_data("robot")

    def get_robot_data(self, entity_name: str = "robot"):
        if hasattr(self, "_robot_data_cache") and entity_name in self._robot_data_cache:
            return self._robot_data_cache[entity_name]
        # Fallback before cache is built (e.g. during setup)
        return self.scene_manager.get_entity(entity_name).data

    def get_robot_state_writer(self, entity_name: str = "robot"):
        """Return the write-API companion to ``get_robot_data``.

        Mirrors NewtonEnv / GenesisEnv: callers can use a single
        cross-sim accessor to mutate joint and root state via the
        ``RobotStateWriterProtocol`` shape (see
        ``managers/common/robot_state_writer_protocol.py``).
        """
        return self._robot_state_writer_cache[entity_name]

    def resolve_selector(self, selector: SceneEntitySelector) -> ResolvedEntity:
        # Idempotent: managers pre-resolve selectors in term params, so a
        # DR/reward term that re-calls resolve_selector may pass the already-
        # resolved object back in.
        if isinstance(selector, ResolvedEntity):
            return selector
        # Build a transient mjlab SceneEntityCfg purely so we can reuse
        # mjlab's resolver — never expose this object to JaxRLWorld
        # callers (the mjlab type is an internal backend detail).
        mjlab_cfg = _MjlabSceneEntityCfg(
            name=selector.name,
            joint_names=tuple(selector.joint_names) if selector.joint_names else None,
            body_names=tuple(selector.body_names) if selector.body_names else None,
            geom_names=tuple(selector.geom_names) if selector.geom_names else None,
            site_names=tuple(selector.site_names) if selector.site_names else None,
            actuator_names=(tuple(selector.actuator_names) if selector.actuator_names else None),
            preserve_order=selector.preserve_order,
        )
        mjlab_cfg.resolve(self.scene_manager.scene)

        joint_ids, joint_names_resolved = self._resolve_canonical_joint_ids(
            selector.joint_names, preserve_order=selector.preserve_order
        )

        def _to_tensor(ids, requested) -> torch.Tensor | None:
            if not requested:
                return None
            if isinstance(ids, slice):
                return None
            return torch.as_tensor(list(ids), device=self.device, dtype=torch.long)

        def _names(attr_value, requested) -> list[str] | None:
            """mjlab populates ``names_attr`` after resolve when requested."""
            if not requested:
                return None
            if attr_value is None:
                return None
            return list(attr_value)

        return ResolvedEntity(
            source_selector=selector,
            name=selector.name,
            backend_handle=self.scene_manager.get_entity(selector.name),
            joint_ids=joint_ids,
            joint_ids_native=_to_tensor(mjlab_cfg.joint_ids, selector.joint_names),
            body_ids=_to_tensor(mjlab_cfg.body_ids, selector.body_names),
            geom_ids=_to_tensor(mjlab_cfg.geom_ids, selector.geom_names),
            site_ids=_to_tensor(mjlab_cfg.site_ids, selector.site_names),
            actuator_ids=_to_tensor(mjlab_cfg.actuator_ids, selector.actuator_names),
            joint_names=joint_names_resolved if selector.joint_names is not None else None,
            body_names=_names(mjlab_cfg.body_names, selector.body_names),
            geom_names=_names(mjlab_cfg.geom_names, selector.geom_names),
            site_names=_names(mjlab_cfg.site_names, selector.site_names),
            actuator_names=_names(mjlab_cfg.actuator_names, selector.actuator_names),
        )

    def _build_scene(self) -> None:
        """Create MuJoCo/mjlab scene via ManagerRegistry."""
        # Sync num_envs (eval env may override env_cfg.num_envs)
        self.scene_cfg.num_envs = self.num_envs

        SceneCls = ManagerRegistry.get_class(self.sim_type, "scene")
        SceneCfgCls = ManagerRegistry.get_config_class(self.sim_type, "scene")
        self.scene_manager = SceneCls(
            env=self,
            config=SceneCfgCls(
                device=str(self.device),
                robot_entity_name=self.scene_cfg.robot_entity_name,
                num_envs=self.scene_cfg.num_envs,
                env_spacing=self.scene_cfg.env_spacing,
                physics_dt=self.scene_cfg.physics_dt,
                substeps=getattr(self.scene_cfg, "substeps", 1),
                entities=getattr(self.scene_cfg, "entities", None),
                sensors=getattr(self.scene_cfg, "sensors", ()),
                terrain_type=getattr(self.scene_cfg, "terrain_type", "plane"),
                solver_iterations=getattr(self.scene_cfg, "solver_iterations", 10),
                solver_ls_iterations=getattr(self.scene_cfg, "solver_ls_iterations", 20),
                ccd_iterations=getattr(self.scene_cfg, "ccd_iterations", 50),
                nconmax=getattr(self.scene_cfg, "nconmax", 35),
                njmax=getattr(self.scene_cfg, "njmax", 1500),
                impratio=getattr(self.scene_cfg, "impratio", 1.0),
                cone=getattr(self.scene_cfg, "cone", "pyramidal"),
                contact_sensor_maxmatch=getattr(self.scene_cfg, "contact_sensor_maxmatch", 64),
                # Legacy fallbacks
                mjlab_scene_cfg=getattr(self.scene_cfg, "mjlab_scene_cfg", None),
                mjlab_sim_cfg=getattr(self.scene_cfg, "mjlab_sim_cfg", None),
                unified_entities=getattr(self.scene_cfg, "unified_entities", None),
            ),
        )
        self.scene_manager.build_scene()

        # physics_dt stays at the config-level dt (before substep
        # division), matching Newton's convention. The MuJoCo solver's
        # actual timestep is dt/substeps, but that's internal to the
        # scene manager — the env and reward/obs code only sees the
        # config-level dt × decimation = control_dt.
        self.physics_dt = self.scene_cfg.physics_dt
        self.control_dt = self.physics_dt * self.decimation

    def _build_sim_managers(self) -> None:
        """Create MuJoCo-specific managers via ManagerRegistry."""
        ActCls = ManagerRegistry.get_class(self.sim_type, "action")
        ActCfgCls = ManagerRegistry.get_config_class(self.sim_type, "action")
        self.act_manager = ActCls(
            env=self,
            config=ActCfgCls(
                actuated_dof_names=self.act_cfg.actuated_dof_names,
                scale=self.act_cfg.action_scale,
                clip=self.act_cfg.clip_actions,
                offset=self.act_cfg.offset,
                settle_steps=self.act_cfg.settle_steps,
                action_terms=self.act_cfg.action_terms,
            ),
        )

        # Build MujocoRobotData using ArticulationIndexing
        from rlworld.rl.envs.mujoco.robot_data import MujocoRobotData
        from rlworld.rl.envs.mujoco.robot_state_writer import MujocoRobotStateWriter

        self._robot_data_cache = {}
        self._robot_state_writer_cache = {}
        entity = self.scene_manager.robot
        indexing = self.act_manager.indexing
        self._robot_data_cache["robot"] = MujocoRobotData(
            entity=entity,
            joint_ids=indexing.sim_indices,
            num_envs=self.num_envs,
            device=self.device,
            env=self,
            default_joint_pos=self._resolve_default_joint_pos(),
        )
        self._robot_state_writer_cache["robot"] = MujocoRobotStateWriter(
            env=self,
            entity=entity,
            joint_ids=indexing.sim_indices,
        )

        ObsCls = ManagerRegistry.get_class(self.sim_type, "observation")
        self.obs_manager = ObsCls(
            env=self,
            config=self.obs_cfg,
        )

        ContactCls = ManagerRegistry.get_class(self.sim_type, "contact")
        self.contact_manager = ContactCls(env=self)
        self.contact_manager.register_sensors()

        viewer_type = getattr(self.visualization_cfg, "viewer_type", None)
        if viewer_type == "viser":
            from rlworld.rl.envs.managers.mujoco.visualization import (
                MujocoVisualizationManager,
                MujocoVisualizationManagerConfig,
            )

            viz_config = MujocoVisualizationManagerConfig(
                viewer_type="viser",
                viser_port=self.visualization_cfg.viser_port,
            )
            self.visualization_manager = MujocoVisualizationManager(env=self, config=viz_config)
            self.visualization_manager.setup()
        else:
            self.visualization_manager = None

    def _post_setup(self) -> None:
        """Expand model fields for per-env domain randomization."""
        from rlworld.rl.configs.base_config import iter_terms
        from rlworld.rl.configs.events.event_term_config import EventTermConfig

        dr_fields = []
        for name, term in iter_terms(self.event_cfg, EventTermConfig).items():
            if term.mode == "startup" and "field" in term.params:
                dr_fields.append(term.params["field"])
        if dr_fields:
            self.scene_manager.sim.expand_model_fields(dr_fields)

    def _step_physics(self) -> None:
        for _ in range(self.decimation):
            self.act_manager.apply_actions(self.act_manager.processed_actions)
            self.scene_manager.write_data_to_sim()
            self.scene_manager.step()
            self.scene_manager.update(dt=self.physics_dt)

        # Update visualization
        if self.visualization_manager is not None:
            self.visualization_manager.advance()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset with mjlab-specific write to sim."""
        self.scene_manager.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) > 0:
            self.scene_manager.write_data_to_sim()
            # forward() for reset envs is now handled by the
            # consolidated _post_reset_forward() hook which runs
            # sim.forward() for ALL envs (reset + non-reset).

    def _post_reset_forward(self) -> None:
        """Refresh all MuJoCo derived quantities for every environment.

        mjlab's native env calls ``sim.forward()`` once after resets
        to recompute xpos, xquat, site positions, cvel, and sensor
        data from the current qpos/qvel — covering both freshly-reset
        envs and non-reset envs whose kinematics were last updated
        inside the decimation loop's ``scene.update(dt)`` call.
        """
        self.scene_manager.forward()
        self.scene_manager.sim.sense()
