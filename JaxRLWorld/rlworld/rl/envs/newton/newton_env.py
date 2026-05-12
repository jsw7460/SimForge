import torch
import warp as wp

from rlworld.rl.configs import CommandConfig, CurriculumManagerConfig, EventConfig, RewardConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.managers.newton import (
    NewtonVisualizationManager,
    NewtonVisualizationManagerConfig,
)
from rlworld.rl.envs.managers.registry import ManagerRegistry
from rlworld.rl.envs.world import World
from rlworld.rl.utils import set_seed, string as _su


class NewtonEnv(World):
    sim_name: str = "Newton"
    sim_type: str = "newton"

    def __init__(
        self,
        num_envs: int,
        env_cfg: NewtonEnvConfig,
        scene_cfg: NewtonSceneConfig,
        visualization_cfg: VisualizationConfig,
        obs_cfg: NewtonObservationConfig,
        act_cfg: NewtonActionConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        event_cfg: EventConfig,
        curriculum_cfg: CurriculumManagerConfig,
    ):
        set_seed(env_cfg.seed)
        super().__init__()

        self.seed = env_cfg.seed
        self.num_envs = num_envs
        self.device = wp.device_to_torch(wp.get_device())

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

        # Timing
        self.physics_dt = scene_cfg.dt
        self.decimation = getattr(env_cfg, "decimation", 1)
        self.control_dt = self.physics_dt * self.decimation

        # Initialize buffers
        self._init_buffers()

        # Setup
        self._setup_environment()

    @property
    def robot(self):
        return self.scene_manager.model

    @property
    def robot_data(self):
        return self.get_robot_data("robot")

    def get_robot_data(self, entity_name: str = "robot"):
        # Newton currently supports single entity only
        return self._robot_data

    def get_robot_state_writer(self, entity_name: str = "robot"):
        """Return the write-API companion to ``get_robot_data``.

        Used by event terms / reset functions to mutate joint and root
        state via ``NewtonRobotStateWriter`` (see
        ``rlworld/rl/envs/newton/robot_state_writer.py``). Newton
        currently supports a single entity only.
        """
        return self._robot_state_writer

    def resolve_selector(self, selector: SceneEntitySelector) -> ResolvedEntity:
        view = self.scene_manager.articulation_views[selector.name]

        joint_ids, joint_names_resolved = self._resolve_canonical_joint_ids(
            selector.joint_names, preserve_order=selector.preserve_order
        )

        body_ids: torch.Tensor | None = None
        body_names_resolved: list[str] | None = None
        if selector.body_names is not None:
            link_idx, link_names_matched = _su.resolve_matching_names(
                list(selector.body_names),
                list(view.link_names),
                preserve_order=selector.preserve_order,
            )
            body_ids = torch.tensor(link_idx, device=self.device, dtype=torch.long)
            body_names_resolved = list(link_names_matched)

        # In Newton, "geom" maps to a per-shape entry under
        # ``ArticulationView.shape_names`` (bare names) /
        # ``model.shape_label`` (XPath).
        geom_ids: torch.Tensor | None = None
        geom_names_resolved: list[str] | None = None
        if selector.geom_names is not None:
            shape_names = list(view.shape_names)
            try:
                shape_idx, shape_names_matched = _su.resolve_matching_names(
                    list(selector.geom_names),
                    shape_names,
                    preserve_order=selector.preserve_order,
                )
            except ValueError:
                # Fall back to full XPath labels.
                all_labels = list(self.scene_manager.model.shape_label)
                world_count = self.scene_manager.model.world_count
                per_world = len(all_labels) // world_count
                first_env_labels = all_labels[:per_world]
                shape_idx, shape_names_matched = _su.resolve_matching_names(
                    list(selector.geom_names),
                    first_env_labels,
                    preserve_order=selector.preserve_order,
                )
            geom_ids = torch.tensor(shape_idx, device=self.device, dtype=torch.long)
            geom_names_resolved = list(shape_names_matched)

        # Newton actuators are 1:1 with act_manager.actuated_joint_names.
        actuator_ids: torch.Tensor | None = None
        actuator_names_resolved: list[str] | None = None
        if selector.actuator_names is not None:
            actuator_ids, actuator_names_resolved = self._resolve_canonical_joint_ids(
                selector.actuator_names, preserve_order=selector.preserve_order
            )

        if selector.site_names is not None:
            raise NotImplementedError(
                "Newton does not expose sites as a first-class concept; "
                "use SceneEntitySelector.body_names or geom_names instead."
            )

        return ResolvedEntity(
            source_selector=selector,
            name=selector.name,
            joint_ids=joint_ids,
            joint_ids_native=None,
            body_ids=body_ids,
            geom_ids=geom_ids,
            site_ids=None,
            actuator_ids=actuator_ids,
            joint_names=joint_names_resolved if selector.joint_names is not None else None,
            body_names=body_names_resolved,
            geom_names=geom_names_resolved,
            actuator_names=actuator_names_resolved,
        )

    def _build_scene(self) -> None:
        """Create Newton scene and visualization manager."""
        SceneCls = ManagerRegistry.get_class(self.sim_type, "scene")
        SceneCfgCls = ManagerRegistry.get_config_class(self.sim_type, "scene")

        self.scene_manager = SceneCls(
            env=self,
            config=SceneCfgCls(
                num_worlds=self.num_envs,
                entities=self.scene_cfg.entities,
                sensors=self.scene_cfg.sensors,
                contact_sensors=getattr(self.scene_cfg, "contact_sensors", None),
                add_ground=self.scene_cfg.add_ground,
                dt=self.scene_cfg.dt,
                substeps=self.scene_cfg.substeps,
                gravity=self.scene_cfg.gravity,
                solver_type=self.scene_cfg.solver_type,
                solver_cfg=self.scene_cfg.solver_cfg,
                env_spacing=self.scene_cfg.env_spacing,
            ),
        )
        self.scene_manager.register_entities()
        self.scene_manager.build_scene()

        # Visualization Manager (after scene build)
        # viewer_type="viser" → always use ViserVisualizationManager (matches Genesis pattern)
        if self.visualization_cfg.viewer_type == "viser":
            from rlworld.rl.vis.viser import ViserVisualizationManager
            from rlworld.rl.vis.viser.bridges import NewtonBridge
            from rlworld.rl.vis.viser.viewer import ViserViewerConfig

            bridge = NewtonBridge(self.scene_manager)
            viser_cfg = ViserViewerConfig(
                port=self.visualization_cfg.viser_port,
                share=self.visualization_cfg.viser_share,
                enable_reward_plots=self.visualization_cfg.viser_enable_reward_plots,
                enable_debug_viz=self.visualization_cfg.viser_enable_debug_viz,
                scene=self.visualization_cfg.viser_scene,
            )
            self.vis_manager = ViserVisualizationManager(env=self, bridge=bridge, config=viser_cfg)
        elif self.visualization_cfg.show_viewer or self.visualization_cfg.record_video:
            # Non-viser viewer (GL, rerun, usd, file) — only when actively viewing or recording
            vis_manager_cfg = NewtonVisualizationManagerConfig(
                show_viewer=self.visualization_cfg.show_viewer,
                record_video=self.visualization_cfg.record_video,
                video_dir=self.visualization_cfg.video_dir,
                video_fps=self.visualization_cfg.video_fps or 60,
                viewer_type=self.visualization_cfg.viewer_type,
                viser_port=self.visualization_cfg.viser_port,
                viser_share=self.visualization_cfg.viser_share,
                rerun_web_port=self.visualization_cfg.rerun_web_port,
            )
            self.vis_manager = NewtonVisualizationManager(env=self, config=vis_manager_cfg)
            self.vis_manager.setup()
        else:
            # Headless — no visualization overhead
            self.vis_manager = None

    def _build_sim_managers(self) -> None:
        """Create Newton-specific managers via ManagerRegistry."""
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

        ObsCls = ManagerRegistry.get_class(self.sim_type, "observation")
        self.obs_manager = ObsCls(
            env=self,
            config=self.obs_cfg,
        )

        ContactCls = ManagerRegistry.get_class(self.sim_type, "contact")
        self.contact_manager = ContactCls(env=self)
        self.contact_manager.register_sensors()

        from rlworld.rl.envs.newton.robot_data import NewtonRobotData
        from rlworld.rl.envs.newton.robot_state_writer import NewtonRobotStateWriter

        self._robot_data = NewtonRobotData(
            self,
            self.scene_manager.robot_view,
            default_joint_pos=self._resolve_default_joint_pos(),
        )
        self._robot_state_writer = NewtonRobotStateWriter(self, self.scene_manager.robot_view)

    def _post_setup(self) -> None:
        """Capture CUDA graph for Newton performance."""
        self.scene_manager.capture()

    def _step_physics(self) -> None:
        """Newton physics step with decimation.

        Each decimation iteration recomputes actuator torques using the
        latest joint state, then runs scene_manager.step() which
        executes the internal substep loop (potentially via CUDA graph).
        """
        for _ in range(self.decimation):
            self.act_manager.apply_actions(self.act_manager.processed_actions)
            self.scene_manager.step()

        # Update visualization
        if self.vis_manager is not None:
            self.vis_manager.advance()
