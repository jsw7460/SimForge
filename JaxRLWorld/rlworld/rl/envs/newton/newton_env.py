import torch
import warp as wp
from warp.torch import device_to_torch

from genesis.utils.geom import quat_to_xyz
from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonEnvConfig,
    NewtonSceneConfig,
    NewtonObservationConfig,
    NewtonActionConfig,
    VisualizationConfig
)
from rlworld.rl.envs.managers import (
    CommandManager, CommandManagerConfig,
    RewardManager, RewardManagerConfig,
    TerminationManager, TerminationConfig,
    EventManager, EventManagerConfig,
)
from rlworld.rl.envs.managers.newton import (
    NewtonSceneManager, NewtonSceneManagerConfig,
    NewtonActionManager, NewtonActionManagerConfig,
    NewtonObservationManager, NewtonObsManagerConfig,
    NewtonVisualizationManager, NewtonVisualizationManagerConfig,
    NewtonContactManager
)
from rlworld.rl.envs.mdp.observations.newton.proprioception import base_quat
from rlworld.rl.envs.world import World
from rlworld.rl.utils import set_seed


class NewtonEnv(World):
    sim_name: str = "Newton"

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
    ):
        set_seed(env_cfg.seed)
        super().__init__()

        self.seed = env_cfg.seed
        self.num_envs = num_envs
        self.device = device_to_torch(wp.get_device())

        # Store high-level configs
        self.env_cfg = env_cfg
        self.scene_cfg = scene_cfg
        self.visualization_cfg = visualization_cfg
        self.obs_cfg = obs_cfg
        self.act_cfg = act_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.event_cfg = event_cfg

        # Timing
        self.physics_dt = scene_cfg.dt
        self.control_dt = self.physics_dt

        # Initialize buffers
        self._init_buffers()

        # Setup
        self._setup_environment()

    @property
    def robot(self):
        return self.scene_manager.model

    @property
    def heading_w(self) -> torch.Tensor:
        """Get robot heading (yaw) in world frame.

        Newton joint_q[:, 3:7] is xyzw convention,
        quat_to_xyz expects wxyz.

        Returns:
            Tensor of shape [num_envs] in radians.
        """
        quat_xyzw = base_quat(self)  # [num_envs, 4] xyzw
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        euler = quat_to_xyz(quat_wxyz)  # [num_envs, 3]
        return euler[:, 2]

    def _setup_environment(self) -> None:
        """Setup all managers by converting high-level configs to manager configs."""

        # Scene Manager
        scene_manager_cfg = NewtonSceneManagerConfig(
            num_worlds=self.num_envs,
            entities=self.scene_cfg.entities,
            sensors=self.scene_cfg.sensors,
            add_ground=self.scene_cfg.add_ground,
            dt=self.scene_cfg.dt,
            substeps=self.scene_cfg.substeps,
            gravity=self.scene_cfg.gravity,
            solver_type=self.scene_cfg.solver_type,
            env_spacing=self.scene_cfg.env_spacing,
        )
        self.scene_manager = NewtonSceneManager(env=self, config=scene_manager_cfg)
        self.scene_manager.register_entities()
        self.scene_manager.build_scene()

        # Visualization Manager (after scene build)
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

        # Action Manager
        act_manager_cfg = NewtonActionManagerConfig(
            actuated_dof_names=self.act_cfg.actuated_dof_names,
            scale=self.act_cfg.action_scale,
            clip=self.act_cfg.clip_actions,
            offset=self.act_cfg.offset,
        )
        self.act_manager = NewtonActionManager(env=self, config=act_manager_cfg)

        # Observation Manager
        obs_manager_cfg = NewtonObsManagerConfig(
            num_envs=self.num_envs,
            obs_group=self.obs_cfg.obs_group,
            enable_noise=getattr(self.obs_cfg, 'enable_noise', True),
        )
        self.obs_manager = NewtonObservationManager(env=self, config=obs_manager_cfg)

        # Contact Manager
        self.contact_manager = NewtonContactManager(env=self)
        self.contact_manager.register_sensors()

        # Command Manager (shared with Genesis)
        command_manager_cfg = CommandManagerConfig(
                command_terms=self.command_cfg.sampler,
                resampling_time_s=self.command_cfg.resampling_time_s,
                rel_standing_envs=self.command_cfg.rel_standing_envs,
                heading_command=self.command_cfg.heading_command,
                heading_control_stiffness=self.command_cfg.heading_control_stiffness,
                heading_range=self.command_cfg.heading_range,
                rel_heading_envs=self.command_cfg.rel_heading_envs,
            )
        self.command_manager = CommandManager(env=self, config=command_manager_cfg)

        # Reward Manager (shared with Genesis)
        reward_manager_cfg = RewardManagerConfig(
            reward_terms=self.reward_cfg.reward_terms,
        )
        self.reward_manager = RewardManager(env=self, config=reward_manager_cfg)

        # Termination Manager (shared with Genesis)
        termination_cfg = TerminationConfig(
            num_envs=self.num_envs,
            termination_criteria=self.env_cfg.termination_criteria,
            episode_length_s=self.env_cfg.episode_length_s,
        )
        self.termination_manager = TerminationManager(env=self, config=termination_cfg)

        # Event Manager (shared with Genesis)
        event_manager_cfg = EventManagerConfig(
            event_terms=self.event_cfg.event_terms,
        )
        self.event_manager = EventManager(env=self, config=event_manager_cfg)

        # Capture graph for performance
        self.scene_manager.capture()

        # Apply startup events
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

        # Pretty print environment summary
        from rlworld.rl.utils.pretty import print_env_summary
        print_env_summary(self)

    def _step_physics(self) -> None:
        """Newton physics step (substeps handled internally by scene manager)."""
        self.scene_manager.step()

        # Update visualization
        if self.vis_manager is not None:
            self.vis_manager.advance()

    def _apply_actions(self, processed_actions: torch.Tensor) -> None:
        """Apply actions to Newton control."""
        self.act_manager.apply_actions(processed_actions)
