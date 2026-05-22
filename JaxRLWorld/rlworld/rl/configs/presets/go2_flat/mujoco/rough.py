"""Go2 MuJoCo config: simple velocity-tracking walking on rough terrain.

Inherits the plain :class:`Go2FlatConfig` and only replaces the flat
ground plane with a generated rough-terrain heightfield via
``use_rough_terrain=True`` (the same canonical ``ROUGH_TERRAINS_CFG`` fed
to Newton). Blind simple-walking baseline on rough ground.
"""

from dataclasses import dataclass

from rlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun
from rlworld.rl.configs.presets.go2_flat.base import Go2FlatConfig


@dataclass
class Go2RoughMujocoConfig(Go2FlatConfig):
    sim_type: str = "mujoco"
    run_name: str = "Go2_Rough_Mujoco"
    use_rough_terrain: bool = True

    def build(self) -> MujocoConfigsForRun:
        return super().build()


def get_config() -> MujocoConfigsForRun:
    return Go2RoughMujocoConfig().build()
