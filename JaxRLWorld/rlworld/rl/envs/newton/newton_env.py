import jax
import jax.numpy as jnp
import warp as wp

from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonEnvConfig,
    NewtonSceneConfig,
    NewtonObservationConfig,
    NewtonActionConfig,
    VisualizationConfig
)
from rlworld.rl.envs.managers.common.command_jax import (
    JaxCommandManager, CommandManagerConfig,
)
from rlworld.rl.envs.managers.common.reward_jax import (
    JaxRewardManager, RewardManagerConfig,
)
from rlworld.rl.envs.managers.common.termination_jax import (
    JaxTerminationManager, TerminationConfig,
)
from rlworld.rl.envs.managers.common.event_jax import (
    JaxEventManager, EventManagerConfig,
)
from rlworld.rl.envs.managers.newton import (
    NewtonSceneManager, NewtonSceneManagerConfig,
    NewtonActionManager, NewtonActionManagerConfig,
    NewtonObservationManager, NewtonObsManagerConfig,
    NewtonVisualizationManager, NewtonVisualizationManagerConfig,
    NewtonContactManager
)
from rlworld.rl.envs.mdp.observations.newton.proprioception import base_quat
from rlworld.rl.envs.world_jax import JaxWorld
from rlworld.rl.utils import set_seed


def _quat_to_yaw_jax(quat_wxyz: jax.Array) -> jax.Array:
    """Extract yaw (heading) from wxyz quaternion. Returns shape [num_envs]."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jnp.arctan2(siny_cosp, cosy_cosp)


class NewtonEnv(JaxWorld):
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
        # Store device as torch-compatible string for stats collector compatibility
        wp_device = wp.get_device()
        if wp_device.is_cuda:
            self.device = f"cuda:{wp_device.ordinal}"
        else:
            self.device = "cpu"

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
    def heading_w(self) -> jax.Array:
        """Get robot heading (yaw) in world frame.

        Newton joint_q[:, 3:7] is xyzw convention.
        Convert to wxyz for yaw extraction.

        Returns:
            JAX array of shape [num_envs] in radians.
        """
        quat_xyzw = base_quat(self)  # [num_envs, 4] xyzw
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        return _quat_to_yaw_jax(quat_wxyz)

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

        # Action Manager (already JAX-native from Phase 4)
        act_manager_cfg = NewtonActionManagerConfig(
            actuated_dof_names=self.act_cfg.actuated_dof_names,
            scale=self.act_cfg.action_scale,
            clip=self.act_cfg.clip_actions,
            offset=self.act_cfg.offset,
        )
        self.act_manager = NewtonActionManager(env=self, config=act_manager_cfg)

        # Observation Manager (JAX-native via Newton alias)
        obs_manager_cfg = NewtonObsManagerConfig(
            num_envs=self.num_envs,
            obs_group=self.obs_cfg.obs_group,
        )
        self.obs_manager = NewtonObservationManager(env=self, config=obs_manager_cfg)

        # Contact Manager (already JAX-native from Phase 4)
        self.contact_manager = NewtonContactManager(env=self)
        self.contact_manager.register_sensors()

        # Command Manager (JAX-native)
        command_manager_cfg = CommandManagerConfig(
                command_terms=self.command_cfg.sampler,
                resampling_time_s=self.command_cfg.resampling_time_s,
                rel_standing_envs=self.command_cfg.rel_standing_envs,
                heading_command=self.command_cfg.heading_command,
                heading_control_stiffness=self.command_cfg.heading_control_stiffness,
                heading_range=self.command_cfg.heading_range,
                rel_heading_envs=self.command_cfg.rel_heading_envs,
            )
        self.command_manager = JaxCommandManager(env=self, config=command_manager_cfg)

        # Reward Manager (JAX-native)
        reward_manager_cfg = RewardManagerConfig(
            reward_terms=self.reward_cfg.reward_terms,
        )
        self.reward_manager = JaxRewardManager(env=self, config=reward_manager_cfg)

        # Termination Manager (JAX-native)
        termination_cfg = TerminationConfig(
            num_envs=self.num_envs,
            termination_criteria=self.env_cfg.termination_criteria,
            episode_length_s=self.env_cfg.episode_length_s,
        )
        self.termination_manager = JaxTerminationManager(env=self, config=termination_cfg)

        # Event Manager (JAX-native)
        event_manager_cfg = EventManagerConfig(
            event_terms=self.event_cfg.event_terms,
        )
        self.event_manager = JaxEventManager(env=self, config=event_manager_cfg)

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

    def _apply_actions(self, processed_actions: jax.Array) -> None:
        """Apply actions to Newton control."""
        self.act_manager.apply_actions(processed_actions)
