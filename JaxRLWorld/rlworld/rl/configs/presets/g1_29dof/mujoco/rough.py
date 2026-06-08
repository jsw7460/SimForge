"""G1 29-DOF humanoid on rough terrain (MuJoCo / mjlab backend).

Sibling of ``presets/g1_29dof/newton/rough.py`` — identical structure,
only ``sim_type`` and ``run_name`` differ.  See the Newton variant
for the design rationale.
"""

from dataclasses import dataclass

from rlworld.rl.configs import MujocoConfigsForRun
from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig


@dataclass
class G1RoughMujocoConfig(G1FlatConfig):
    sim_type: str = "mujoco"
    run_name: str = "G1_29dof_Rough_Mujoco"
    use_rough_terrain: bool = True


def get_config() -> MujocoConfigsForRun:
    return G1RoughMujocoConfig().build()
