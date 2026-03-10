import torch

import genesis as gs
from genesis.utils.geom import quat_to_xyz
from rlworld.rl.configs import (
    EnvConfig, SceneConfig, ObservationConfig, VisualizationConfig,
    ActionConfig, RewardConfig, CommandConfig, EventConfig
)
from rlworld.rl.envs.managers import (
    SceneManagerConfig, SceneManager,
    CommandManagerConfig, CommandManager,
    VisualizationManagerConfig, VisualizationManager,
    ObsManagerConfig, ObservationManager,
    ActionManagerConfig, ActionManager,
    TerminationConfig, TerminationManager,
    RewardManagerConfig, RewardManager,
    EventManagerConfig, EventManager,
    ContactManager
)
from rlworld.rl.envs.world import World
from rlworld.rl.utils import set_seed


class GenesisEnv(World):
    sim_name: str = "Genesis"

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
    def heading_w(self) -> torch.Tensor:
        """Get robot heading (yaw) in world frame.

        Genesis get_quat() returns wxyz convention.

        Returns:
            Tensor of shape [num_envs] in radians.
        """
        quat_wxyz = self.robot.get_quat()  # [num_envs, 4] wxyz
        euler = quat_to_xyz(quat_wxyz)  # [num_envs, 3] -> (roll, pitch, yaw)
        return euler[:, 2]

    @property
    def scene(self) -> gs.Scene:
        return self.scene_manager.scene

    def _setup_environment(self) -> None:
        # Scene
        self.scene_manager = SceneManager(
            env=self,
            config=SceneManagerConfig(
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

        # Other managers
        self.command_manager = CommandManager(
            env=self,
            config=CommandManagerConfig(
                command_terms=self.command_cfg.sampler,
                resampling_time_s=self.command_cfg.resampling_time_s,
                rel_standing_envs=self.command_cfg.rel_standing_envs,
                heading_command=self.command_cfg.heading_command,
                heading_control_stiffness=self.command_cfg.heading_control_stiffness,
                heading_range=self.command_cfg.heading_range,
                rel_heading_envs=self.command_cfg.rel_heading_envs,
            )
        )

        self.obs_manager = ObservationManager(
            env=self,
            config=ObsManagerConfig(
                num_envs=self.num_envs,
                obs_group=self.obs_cfg.obs_group,
                enable_noise=getattr(self.obs_cfg, 'enable_noise', True),
            )
        )

        self.act_manager = ActionManager(
            env=self,
            config=ActionManagerConfig(
                actuated_dof_names=self.act_cfg.actuated_dof_names,
                clip=self.act_cfg.clip_actions,
                scale=self.act_cfg.action_scale,
                offset=self.act_cfg.offset,
                control_mode=self.act_cfg.control_mode
            )
        )

        self.termination_manager = TerminationManager(
            env=self,
            config=TerminationConfig(
                num_envs=self.num_envs,
                termination_criteria=self.env_cfg.termination_criteria,
                episode_length_s=self.env_cfg.episode_length_s,
            )
        )

        self.reward_manager = RewardManager(
            env=self,
            config=RewardManagerConfig(reward_terms=self.reward_cfg.reward_terms)
        )

        self.event_manager = EventManager(
            env=self,
            config=EventManagerConfig(event_terms=self.event_cfg.event_terms)
        )

        self.contact_manager = ContactManager(env=self)

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

        # Pretty print environment summary
        from rlworld.rl.utils.pretty import print_env_summary
        print_env_summary(self)

    def _step_physics(self) -> None:
        """Genesis physics step with decimation."""
        for _ in range(self.decimation):
            self.scene_manager.step()
        self.vis_manager.advance()

    def _apply_actions(self, processed_actions: torch.Tensor) -> None:
        """Apply actions via the action manager's configured control mode."""
        self.act_manager.apply_actions(processed_actions)
