import torch

import genesis as gs
from rlworld.rl.configs import (
    EnvConfig, SceneConfig, ObservationConfig, VisualizationConfig,
    ActionConfig, RewardConfig, CommandConfig, EventConfig
)
from rlworld.rl.envs.mdp.configs import CurriculumManagerConfig
from rlworld.rl.envs.managers import (
    VisualizationManagerConfig, VisualizationManager,
)
from rlworld.rl.envs.managers.registry import ManagerRegistry
from rlworld.rl.envs.world import World
from rlworld.rl.utils import set_seed


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
            )
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
            )
        )

        self.scene_manager.register_entities()
        self.scene_manager.build_scene()

        # Replace with unified Viser viewer after scene is built.
        if self.visualization_cfg.viewer_type == "viser":
            from rlworld.rl.vis.viser import ViserVisualizationManager
            from rlworld.rl.vis.viser.viewer import ViserViewerConfig
            from rlworld.rl.vis.viser.bridges import GenesisBridge

            bridge = GenesisBridge(self.scene_manager)
            viser_cfg = ViserViewerConfig(
                port=self.visualization_cfg.viser_port,
                share=self.visualization_cfg.viser_share,
                enable_reward_plots=self.visualization_cfg.viser_enable_reward_plots,
                enable_debug_viz=self.visualization_cfg.viser_enable_debug_viz,
            )
            self.vis_manager = ViserVisualizationManager(
                env=self, bridge=bridge, config=viser_cfg
            )

    def _build_sim_managers(self) -> None:
        """Create Genesis-specific managers via ManagerRegistry."""
        ObsCls = ManagerRegistry.get_class(self.sim_type, "observation")
        self.obs_manager = ObsCls(
            env=self,
            config=self.obs_cfg,
        )

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
            )
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
        for name, entity in self.scene_manager.entities.items():
            self._robot_data_cache[name] = GenesisRobotData(
                entity=entity,
                actuated_dof_ids=indexing.sim_indices,
                num_envs=self.num_envs,
                device=self.device,
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
        for _ in range(self.decimation):
            self.act_manager.apply_actions(self.act_manager.processed_actions)
            self.scene_manager.step()
        self.vis_manager.advance()

    def _apply_actions(self, processed_actions: torch.Tensor) -> None:
        """Apply actions via the action manager's configured control mode."""
        self.act_manager.apply_actions(processed_actions)
