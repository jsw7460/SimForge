from __future__ import annotations

from typing import TYPE_CHECKING

from rlworld.rl.configs import (
    ActionConfig,
    CommandConfig,
    CurriculumManagerConfig,
    EnvConfig,
    EventConfig,
    GaitConfig,
    ObservationConfig,
    RewardConfig,
    SceneConfig,
    VisualizationConfig,
)
from rlworld.rl.envs.genesis.genesis_env import GenesisEnv
from rlworld.rl.envs.managers import GaitManager, gait_config_to_manager_config

if TYPE_CHECKING:
    pass


class GenesisLocomotionEnv(GenesisEnv):
    """Specialized environment for legged locomotion tasks.

    Extends GenesisEnv with gait pattern management.
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
        gait_cfg: GaitConfig,
        curriculum_cfg: CurriculumManagerConfig,
    ):
        self._gait_cfg = gait_cfg
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
            curriculum_cfg=curriculum_cfg,
        )

    def _post_setup(self):
        super()._post_setup()
        if self._gait_cfg is not None:
            manager_cfg = gait_config_to_manager_config(self._gait_cfg, self.num_envs)
            self.gait_manager = GaitManager(env=self, config=manager_cfg)

    def _pre_reward_hook(self):
        if hasattr(self, "gait_manager"):
            self.gait_manager.advance()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0 and hasattr(self, "gait_manager"):
            self.gait_manager.reset(env_ids)
