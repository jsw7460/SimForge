"""G1 29-DOF humanoid on rough terrain (Genesis backend).

Sibling of ``presets/g1_29dof/newton/rough.py`` — identical structure,
only ``sim_type`` and ``run_name`` differ.  See the Newton variant
for the design rationale.
"""

from dataclasses import dataclass

from jaxrlworld.rl.configs import GenesisConfigsForRun
from jaxrlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig


@dataclass
class G1RoughGenesisConfig(G1FlatConfig):
    sim_type: str = "genesis"
    run_name: str = "G1_29dof_Rough_Genesis"
    use_rough_terrain: bool = True


def get_config() -> GenesisConfigsForRun:
    return G1RoughGenesisConfig().build()
