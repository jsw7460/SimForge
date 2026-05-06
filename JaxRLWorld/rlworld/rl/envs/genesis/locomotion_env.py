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
from rlworld.rl.envs import GenesisEnv
from rlworld.rl.envs.managers import GaitManager, GaitManagerConfig

if TYPE_CHECKING:
    pass


def _gait_config_to_manager_config(gait_cfg: GaitConfig, num_envs: int) -> GaitManagerConfig:
    """Convert high-level GaitConfig to internal GaitManagerConfig."""
    return GaitManagerConfig(
        num_envs=num_envs,
        foot_names=gait_cfg.foot_names,
        offset_mode=gait_cfg.offset_mode,
        gait_period=gait_cfg.gait_period,
        default_freq=gait_cfg.default_freq,
        default_duration=gait_cfg.default_duration,
        freq_command=gait_cfg.freq_command,
        duration_command=gait_cfg.duration_command,
        foot_offset_provider=gait_cfg.foot_offset_provider,
        contact_smoothing_sigma=gait_cfg.contact_smoothing_sigma,
    )


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
            manager_cfg = _gait_config_to_manager_config(self._gait_cfg, self.num_envs)
            self.gait_manager = GaitManager(env=self, config=manager_cfg)

    def _pre_reward_hook(self):
        if hasattr(self, "gait_manager"):
            self.gait_manager.advance()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0 and hasattr(self, "gait_manager"):
            self.gait_manager.reset(env_ids)
