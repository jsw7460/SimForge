"""Go2 Newton config: simple velocity-tracking walking on rough terrain.

Inherits the plain :class:`Go2FlatConfig` (standard locomotion — velocity
command, proprioceptive obs, flat-locomotion rewards) and only replaces
the flat ground plane with a rough-terrain heightfield via
``use_rough_terrain=True``. This is the blind simple-walking baseline on
rough ground at a fixed terrain difficulty — NOT the gait-conditioned
(Walk-These-Ways) variant.
"""

from dataclasses import dataclass

from rlworld.rl.configs.newton_config_classes import NewtonConfigsForRun
from rlworld.rl.configs.presets.go2_flat.base import Go2FlatConfig


@dataclass
class Go2RoughNewtonConfig(Go2FlatConfig):
    sim_type: str = "newton"
    run_name: str = "Go2_Rough_Newton"
    use_rough_terrain: bool = True

    def build(self) -> NewtonConfigsForRun:
        return super().build()


def get_config() -> NewtonConfigsForRun:
    return Go2RoughNewtonConfig().build()
