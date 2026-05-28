"""Go2 Genesis config: simple velocity-tracking walking on rough terrain.

Inherits the plain :class:`Go2FlatConfig` and only replaces the flat
ground plane with a generated rough-terrain heightfield via
``use_rough_terrain=True`` (the same canonical ``ROUGH_TERRAINS_CFG`` fed
to Newton / MuJoCo). Blind simple-walking baseline on rough ground.
"""

from dataclasses import dataclass

from rlworld.rl.configs.genesis_config_classes import GenesisConfigsForRun
from rlworld.rl.configs.presets.go2_flat.base import Go2FlatConfig


@dataclass
class Go2RoughGenesisConfig(Go2FlatConfig):
    sim_type: str = "genesis"
    run_name: str = "Go2_Rough_Genesis"
    use_rough_terrain: bool = True

    def build(self) -> GenesisConfigsForRun:
        return super().build()


def get_config() -> GenesisConfigsForRun:
    return Go2RoughGenesisConfig().build()
