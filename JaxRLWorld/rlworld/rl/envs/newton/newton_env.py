import torch
import warp as wp
from warp.torch import device_to_torch

from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig
from rlworld.rl.envs.mdp.configs import CurriculumManagerConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonEnvConfig,
    NewtonSceneConfig,
    NewtonObservationConfig,
    NewtonActionConfig,
    VisualizationConfig
)
from rlworld.rl.envs.managers.newton import (
    NewtonVisualizationManager, NewtonVisualizationManagerConfig,
)
from rlworld.rl.envs.managers.registry import ManagerRegistry
from rlworld.rl.envs.world import World
from rlworld.rl.utils import set_seed


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
                add_ground=self.scene_cfg.add_ground,
                dt=self.scene_cfg.dt,
                substeps=self.scene_cfg.substeps,
                gravity=self.scene_cfg.gravity,
                solver_type=self.scene_cfg.solver_type,
                solver_cfg=self.scene_cfg.solver_cfg,
                env_spacing=self.scene_cfg.env_spacing,
            )
        )
        self.scene_manager.register_entities()
        self.scene_manager.build_scene()

        # Visualization Manager (after scene build)
        # viewer_type="viser" → always use ViserVisualizationManager (matches Genesis pattern)
        if self.visualization_cfg.viewer_type == "viser":
            from rlworld.rl.vis.viser import ViserVisualizationManager
            from rlworld.rl.vis.viser.viewer import ViserViewerConfig
            from rlworld.rl.vis.viser.bridges import NewtonBridge

            bridge = NewtonBridge(self.scene_manager)
            viser_cfg = ViserViewerConfig(
                port=self.visualization_cfg.viser_port,
                share=self.visualization_cfg.viser_share,
                enable_reward_plots=self.visualization_cfg.viser_enable_reward_plots,
                enable_debug_viz=self.visualization_cfg.viser_enable_debug_viz,
            )
            self.vis_manager = ViserVisualizationManager(
                env=self, bridge=bridge, config=viser_cfg
            )
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
            )
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
        self._robot_state_writer = NewtonRobotStateWriter(
            self, self.scene_manager.robot_view
        )

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

    def _apply_actions(self, processed_actions: torch.Tensor) -> None:
        """Apply actions to Newton control."""
        self.act_manager.apply_actions(processed_actions)
