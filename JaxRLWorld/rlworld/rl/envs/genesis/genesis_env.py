import genesis as gs
import torch
from genesis.utils.misc import set_random_seed as _gs_set_random_seed

from rlworld.rl.configs import (
    ActionConfig,
    CommandConfig,
    CurriculumManagerConfig,
    EnvConfig,
    EventConfig,
    ObservationConfig,
    RewardConfig,
    SceneConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.managers import (
    VisualizationManager,
    VisualizationManagerConfig,
)
from rlworld.rl.envs.managers.genesis.contact_sensor import GenesisContactSensor
from rlworld.rl.envs.managers.registry import ManagerRegistry
from rlworld.rl.envs.world import World
from rlworld.rl.utils import entity_utils as _eu, set_seed


class GenesisEnv(World):
    sim_name: str = "Genesis"
    sim_type: str = "genesis"

    def __init__(
        self,
        num_envs: int,
        env_cfg: EnvConfig,
        scene_cfg: SceneConfig,
        visualization_cfg: VisualizationConfig,
        obs_cfg: ObservationConfig,
        act_cfg: ActionConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        event_cfg: EventConfig,
        curriculum_cfg: CurriculumManagerConfig,
    ):
        # Initialise the Genesis runtime on first use.  Kept here (rather than
        # at package import) so a Newton-/MuJoCo-only process never imports or
        # initialises Genesis.
        #
        # ``gs.init(seed=...)`` seeds Python/NumPy/Torch RNGs *and* the
        # Quadrants (warp) kernel RNG. We also want to reseed on every
        # subsequent env construction (curriculum / re-init paths), so
        # call ``set_random_seed`` directly when Genesis is already up.
        if not gs._initialized:
            gs.init(logging_level="warning", seed=env_cfg.seed)
        else:
            _gs_set_random_seed(env_cfg.seed)

        set_seed(env_cfg.seed)
        super().__init__()

        self.seed = env_cfg.seed
        self.num_envs = num_envs
        self.device = gs.device

        # Store configs
        self.env_cfg = env_cfg
        self.scene_cfg = scene_cfg
        self.visualization_cfg = visualization_cfg
        self.obs_cfg = obs_cfg
        self.act_cfg = act_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.event_cfg = event_cfg
        self.curriculum_cfg = curriculum_cfg

        # Timing
        self.physics_dt = scene_cfg.sim_options.dt
        self.decimation = env_cfg.decimation
        self.control_dt = self.physics_dt * self.decimation

        # Initialize buffers
        self._init_buffers()

        # Setup
        self._setup_environment()

    @property
    def robot(self):
        return self.scene_manager.robot

    @property
    def robot_data(self):
        return self.get_robot_data("robot")

    def get_robot_data(self, entity_name: str = "robot"):
        return self._robot_data_cache[entity_name]

    def get_robot_state_writer(self, entity_name: str = "robot"):
        """Return the write-API companion to ``get_robot_data``.

        Mirrors NewtonEnv / MujocoEnv: callers can use a single
        cross-sim accessor to mutate joint and root state via the
        ``RobotStateWriterProtocol`` shape (see
        ``managers/common/robot_state_writer_protocol.py``).
        """
        return self._robot_state_writer_cache[entity_name]

    def resolve_selector(self, selector: SceneEntitySelector) -> ResolvedEntity:
        entity = self.scene_manager[selector.name]

        joint_ids, joint_names_resolved = self._resolve_canonical_joint_ids(
            selector.joint_names, preserve_order=selector.preserve_order
        )

        body_ids = None
        body_names_resolved = None
        if selector.body_names is not None:
            link_ids_local, link_names_matched = _eu.find_links(
                entity,
                list(selector.body_names),
                global_ids=False,
                preserve_order=selector.preserve_order,
            )
            body_ids = torch.tensor(link_ids_local, device=self.device, dtype=torch.long)
            body_names_resolved = list(link_names_matched)

        # Genesis has no first-class actuator concept; act_manager owns
        # the actuator ↔ joint mapping (1:1 with actuated joints), so
        # actuator_ids reuses the canonical joint resolution.
        actuator_ids = None
        actuator_names_resolved = None
        if selector.actuator_names is not None:
            actuator_ids, actuator_names_resolved = self._resolve_canonical_joint_ids(
                selector.actuator_names, preserve_order=selector.preserve_order
            )

        if selector.geom_names is not None:
            raise NotImplementedError(
                "Genesis geoms have no names; use SceneEntitySelector.body_names "
                "and let the backend expand to the link's collision geoms."
            )
        if selector.site_names is not None:
            raise NotImplementedError("Genesis has no site concept; use body_names instead.")

        return ResolvedEntity(
            source_selector=selector,
            name=selector.name,
            joint_ids=joint_ids,
            joint_ids_native=None,
            body_ids=body_ids,
            geom_ids=None,
            site_ids=None,
            actuator_ids=actuator_ids,
            joint_names=joint_names_resolved if selector.joint_names is not None else None,
            body_names=body_names_resolved,
            actuator_names=actuator_names_resolved,
        )

    @property
    def scene(self) -> gs.Scene:
        return self.scene_manager.scene

    def _build_scene(self) -> None:
        """Create Genesis scene and visualization manager."""
        SceneCls = ManagerRegistry.get_class(self.sim_type, "scene")
        SceneCfgCls = ManagerRegistry.get_config_class(self.sim_type, "scene")

        self.scene_manager = SceneCls(
            env=self,
            config=SceneCfgCls(
                sim_options=self.scene_cfg.sim_options,
                viewer_options=self.scene_cfg.viewer_options,
                vis_options=self.scene_cfg.vis_options,
                rigid_options=self.scene_cfg.rigid_options,
                entities=self.scene_cfg.entities,
                sensors=self.scene_cfg.sensors,
                env_spacing=self.scene_cfg.env_spacing,
                show_viewer=self.visualization_cfg.show_viewer,
                num_envs=self.num_envs,
                device=str(self.device),
                terrain_cfg=self.scene_cfg.terrain_cfg,
            ),
        )

        # Visualization (created before register_entities which references it).
        self.vis_manager = VisualizationManager(
            env=self,
            config=VisualizationManagerConfig(
                show_viewer=self.visualization_cfg.show_viewer,
                record_video=self.visualization_cfg.record_video,
                video_dir=self.visualization_cfg.video_dir,
                video_fps=self.visualization_cfg.video_fps,
                record_env_ids=self.visualization_cfg.record_env_ids,
                grid_layout=self.visualization_cfg.grid_layout,
                enable_command_arrow=self.visualization_cfg.enable_command_arrow,
                command_arrow_radius=self.visualization_cfg.command_arrow_radius,
                command_arrow_length_scale=self.visualization_cfg.command_arrow_length_scale,
                max_arrow_length=self.visualization_cfg.max_arrow_length,
                enable_text_hud=self.visualization_cfg.enable_text_hud,
                hud_position=self.visualization_cfg.hud_position,
                feet_names=self.visualization_cfg.feet_names,
                extra_hud_items=self.visualization_cfg.extra_hud_items,
            ),
        )

        self.scene_manager.register_entities()
        # Native Genesis contact sensors (gs.sensors.Contact / ContactForce) must be
        # added to the scene *before* scene.build() (scene.add_sensor is
        # @gs.assert_unbuilt). ContactManager.register_sensor() runs post-build, so we
        # pre-create the ContactSensorCfg-backed sensors here and stash them for the
        # contact manager to adopt.
        self._genesis_contact_sensors: dict = {}
        contact_sensors = getattr(self.scene_cfg, "contact_sensors", None)
        if contact_sensors:
            for sensor_cfg in contact_sensors:
                sensor = GenesisContactSensor(self, sensor_cfg)
                sensor.create_native_sensors()
                self._genesis_contact_sensors[sensor_cfg.name] = sensor
        self.scene_manager.build_scene()

        # Replace with unified Viser viewer after scene is built.
        if self.visualization_cfg.viewer_type == "viser":
            from rlworld.rl.vis.viser import ViserVisualizationManager
            from rlworld.rl.vis.viser.bridges import GenesisBridge
            from rlworld.rl.vis.viser.viewer import ViserViewerConfig

            bridge = GenesisBridge(self.scene_manager)
            viser_cfg = ViserViewerConfig(
                port=self.visualization_cfg.viser_port,
                share=self.visualization_cfg.viser_share,
                enable_reward_plots=self.visualization_cfg.viser_enable_reward_plots,
                enable_debug_viz=self.visualization_cfg.viser_enable_debug_viz,
                scene=self.visualization_cfg.viser_scene,
            )
            self.vis_manager = ViserVisualizationManager(env=self, bridge=bridge, config=viser_cfg)

    def _build_sim_managers(self) -> None:
        """Create Genesis-specific managers via ManagerRegistry.

        Order matters: the ActionManager must exist before the
        ObservationManager because the latter resolves SceneEntitySelector
        params in ``__init__`` and ``resolve_selector`` needs
        ``act_manager.actuated_joint_names`` for the canonical joint order.
        (Newton / MuJoCo already build act → obs.)
        """
        ActCls = ManagerRegistry.get_class(self.sim_type, "action")
        ActCfgCls = ManagerRegistry.get_config_class(self.sim_type, "action")
        self.act_manager = ActCls(
            env=self,
            config=ActCfgCls(
                actuated_dof_names=self.act_cfg.actuated_dof_names,
                clip=self.act_cfg.clip_actions,
                scale=self.act_cfg.action_scale,
                offset=self.act_cfg.offset,
                settle_steps=self.act_cfg.settle_steps,
                action_terms=self.act_cfg.action_terms,
            ),
        )

        ObsCls = ManagerRegistry.get_class(self.sim_type, "observation")
        self.obs_manager = ObsCls(
            env=self,
            config=self.obs_cfg,
        )

        ContactCls = ManagerRegistry.get_class(self.sim_type, "contact")
        self.contact_manager = ContactCls(env=self)
        contact_sensors = getattr(self.scene_cfg, "contact_sensors", None)
        if contact_sensors:
            for sensor_cfg in contact_sensors:
                self.contact_manager.register_sensor(sensor_cfg)

        from rlworld.rl.envs.genesis.robot_data import GenesisRobotData
        from rlworld.rl.envs.genesis.robot_state_writer import GenesisRobotStateWriter

        self._robot_data_cache = {}
        self._robot_state_writer_cache = {}
        indexing = self.act_manager.indexing
        _default_jp = self._resolve_default_joint_pos()
        for name, entity in self.scene_manager.entities.items():
            self._robot_data_cache[name] = GenesisRobotData(
                entity=entity,
                actuated_dof_ids=indexing.sim_indices,
                num_envs=self.num_envs,
                device=self.device,
                default_joint_pos=_default_jp,
            )
            self._robot_state_writer_cache[name] = GenesisRobotStateWriter(
                env=self,
                entity=entity,
                actuated_dof_ids=indexing.sim_indices,
            )

    def _step_physics(self) -> None:
        """Genesis physics step with decimation.

        When an actuator model is active, torques are recomputed every
        substep using the latest joint state (matching IsaacLab behavior).
        For position control without an actuator, re-applying the same
        target each substep is a harmless no-op.
        """
        prof = self._step_profiler
        for _ in range(self.decimation):
            with prof.section("  phys:apply_actions"):
                self.act_manager.apply_actions(self.act_manager.processed_actions)
            with prof.section("  phys:scene.step"):
                self.scene_manager.step()
            with prof.section("  phys:contact_manager.advance"):
                self.contact_manager.advance(dt=self.physics_dt)
        with prof.section("  phys:vis_manager.advance"):
            self.vis_manager.advance()
