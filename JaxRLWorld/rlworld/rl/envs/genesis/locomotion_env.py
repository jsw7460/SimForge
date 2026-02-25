from __future__ import annotations
from typing import TYPE_CHECKING

from rlworld.rl.envs import GenesisEnv
from rlworld.rl.envs.managers import GaitManagerConfig, GaitManager
from rlworld.rl.configs import (
    EnvConfig,
    SceneConfig,
    VisualizationConfig,
    ObservationConfig,
    ActionConfig,
    RewardConfig,
    CommandConfig,
    EventConfig
)

if TYPE_CHECKING:
    pass


class LocomotionEnv(GenesisEnv):
    """
    Specialized environment for legged locomotion tasks.

    Extends RLEnv with locomotion-specific components:
    - Gait pattern management for coordinated foot movements
    - Foot contact tracking
    - Additional locomotion-specific observations and rewards
    """

    gait_manager: GaitManager

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
        # Store locomotion-specific config before parent init
        self._gait_period = 0.8
        # Call parent constructor
        super().__init__(
            num_envs=num_envs,
            env_cfg=env_cfg,
            scene_cfg=scene_cfg,
            visualization_cfg=visualization_cfg,
            obs_cfg=obs_cfg,
            act_cfg=act_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            event_cfg=event_cfg,
        )

    def _setup_environment(self):
        """Override to add locomotion-specific managers"""
        super()._setup_environment()
        self._initialize_gait_manager()

    def _initialize_gait_manager(self):
        """Initialize gait pattern manager for locomotion"""
        config = GaitManagerConfig(
            num_envs=self.num_envs,
            gait_period=self._gait_period,
            foot_names=self.scene_cfg.robot_cfg.foot_names
        )
        self.gait_manager = GaitManager(
            env=self,
            config=config
        )

    def _pre_termination_hook(self):
        """Advance gait before termination check (so obs can use gait info)"""
        self.gait_manager.advance()

    def _reset_idx(self, env_ids):
        """Override to reset gait manager"""
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self.gait_manager.reset(env_ids)