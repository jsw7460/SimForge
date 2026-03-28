"""MuJoCo/mjlab environment for rlworld.

This module provides MjlabEnv, which wraps mjlab's Scene and Simulation
while following rlworld's World interface and manager pattern.
"""
from __future__ import annotations

import torch

from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig
from rlworld.rl.configs.common_config_classes import VisualizationConfig
from rlworld.rl.configs.mujoco_config_classes import (
    MujocoEnvConfig,
    MujocoSceneConfig,
    MujocoObservationConfig,
    MujocoActionConfig,
)
from rlworld.rl.envs.managers.registry import ManagerRegistry
from rlworld.rl.envs.world import World
from rlworld.rl.utils import set_seed


class MjlabEnv(World):
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

        env = MjlabEnv(
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
        """mjlab's EntityData already satisfies the RobotData protocol."""
        return self.get_robot_data("robot")

    def get_robot_data(self, entity_name: str = "robot"):
        return self.scene_manager.get_entity(entity_name).data

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
                entities=getattr(self.scene_cfg, "entities", None),
                sensors=getattr(self.scene_cfg, "sensors", ()),
                terrain_type=getattr(self.scene_cfg, "terrain_type", "plane"),
                solver_iterations=getattr(self.scene_cfg, "solver_iterations", 10),
                solver_ls_iterations=getattr(self.scene_cfg, "solver_ls_iterations", 20),
                ccd_iterations=getattr(self.scene_cfg, "ccd_iterations", 50),
                nconmax=getattr(self.scene_cfg, "nconmax", 35),
                njmax=getattr(self.scene_cfg, "njmax", 1500),
                contact_sensor_maxmatch=getattr(self.scene_cfg, "contact_sensor_maxmatch", 64),
                # Legacy fallbacks
                mjlab_scene_cfg=getattr(self.scene_cfg, "mjlab_scene_cfg", None),
                mjlab_sim_cfg=getattr(self.scene_cfg, "mjlab_sim_cfg", None),
                unified_entities=getattr(self.scene_cfg, "unified_entities", None),
            )
        )
        self.scene_manager.build_scene()

        # Update physics_dt from simulation
        self.physics_dt = self.scene_manager.physics_dt
        self.control_dt = self.physics_dt * self.decimation

    def _build_sim_managers(self) -> None:
        """Create MuJoCo-specific managers via ManagerRegistry."""
        ActCls = ManagerRegistry.get_class(self.sim_type, "action")
        ActCfgCls = ManagerRegistry.get_config_class(self.sim_type, "action")
        self.act_manager = ActCls(
            env=self,
            config=ActCfgCls(
                entity_name=self.act_cfg.entity_name,
                actuated_dof_names=self.act_cfg.actuated_dof_names,
                scale=self.act_cfg.action_scale,
                clip=self.act_cfg.clip_actions,
                offset=self.act_cfg.offset,
            )
        )

        ObsCls = ManagerRegistry.get_class(self.sim_type, "observation")
        ObsCfgCls = ManagerRegistry.get_config_class(self.sim_type, "observation")
        self.obs_manager = ObsCls(
            env=self,
            config=ObsCfgCls(
                num_envs=self.num_envs,
                obs_group=self.obs_cfg.obs_group,
                enable_noise=getattr(self.obs_cfg, 'enable_noise', True),
            )
        )

        ContactCls = ManagerRegistry.get_class(self.sim_type, "contact")
        self.contact_manager = ContactCls(env=self)
        self.contact_manager.register_sensors()

        viewer_type = getattr(self.visualization_cfg, "viewer_type", None)
        if viewer_type == "viser":
            from rlworld.rl.envs.managers.mujoco.visualization import (
                MjlabVisualizationManager,
                MjlabVisualizationManagerConfig,
            )
            viz_config = MjlabVisualizationManagerConfig(
                viewer_type="viser",
                viser_port=self.visualization_cfg.viser_port,
            )
            self.visualization_manager = MjlabVisualizationManager(
                env=self, config=viz_config
            )
            self.visualization_manager.setup()
        else:
            self.visualization_manager = None

    def _post_setup(self) -> None:
        """Expand model fields for per-env domain randomization."""
        dr_fields = []
        for term in self.event_cfg.event_terms:
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

    def _apply_actions(self, processed_actions: torch.Tensor) -> None:
        """Apply processed actions to mjlab Entity."""
        self.act_manager.apply_actions(processed_actions)

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset with mjlab-specific write + forward."""
        super()._reset_idx(env_ids)

        if len(env_ids) > 0:
            self.scene_manager.write_data_to_sim()
            self.scene_manager.forward()