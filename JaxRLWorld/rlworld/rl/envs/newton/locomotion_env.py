from __future__ import annotations
from typing import TYPE_CHECKING

from rlworld.rl.envs import NewtonEnv
from rlworld.rl.envs.managers import GaitManagerConfig, GaitManager
from rlworld.rl.configs.newton_config_classes import (
    NewtonEnvConfig,
    NewtonSceneConfig,
    NewtonObservationConfig,
    NewtonActionConfig,
    VisualizationConfig
)
from rlworld.rl.configs import RewardConfig, CommandConfig, EventConfig

if TYPE_CHECKING:
    pass


class NewtonLocomotionEnv(NewtonEnv):
    """Specialized Newton environment for legged locomotion tasks."""

    gait_manager: GaitManager

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
        self._gait_period = 0.8

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
        super()._setup_environment()
        self._initialize_gait_manager()

    def _initialize_gait_manager(self):
        config = GaitManagerConfig(
            num_envs=self.num_envs,
            gait_period=self._gait_period,
            foot_names=self.scene_cfg.robot_cfg.prefixed_foot_names
        )
        self.gait_manager = GaitManager(env=self, config=config)

    def _pre_termination_hook(self):
        self.gait_manager.advance()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self.gait_manager.reset(env_ids)