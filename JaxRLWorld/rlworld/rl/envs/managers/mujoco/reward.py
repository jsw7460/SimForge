from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg

from rlworld.rl.configs.base_config import iter_terms
from rlworld.rl.configs.common_config_classes import RewardConfig
from rlworld.rl.configs.rewards import RewardTermConfig
from rlworld.rl.envs.managers.common.reward import RewardManager

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class MujocoRewardManager(RewardManager):
    """MuJoCo-specific reward manager with SceneEntityCfg resolution."""

    def __init__(self, env: "World", config: RewardConfig):
        # Resolve SceneEntityCfg params before parent init
        terms = iter_terms(config, RewardTermConfig)
        for term in terms.values():
            if term.params:
                for value in term.params.values():
                    if isinstance(value, SceneEntityCfg):
                        self._reset_ids(value)
                        value.resolve(env.scene_manager.scene)

        super().__init__(env, config)

    @staticmethod
    def _reset_ids(cfg: SceneEntityCfg) -> None:
        """Reset resolved ids back to slice(None) for clean re-resolution."""
        for attr in ("joint_ids", "body_ids", "geom_ids", "site_ids", "actuator_ids"):
            if isinstance(getattr(cfg, attr, None), list):
                setattr(cfg, attr, slice(None))
