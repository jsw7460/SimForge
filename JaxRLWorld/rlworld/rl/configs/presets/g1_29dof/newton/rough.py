"""G1 29-DOF humanoid on rough terrain (Newton backend).

Inherits :class:`G1FlatConfig` and only sets ``use_rough_terrain=True``;
the base preset's ``make_terrain_cfg()`` swaps the flat plane for the
shared ``ROUGH_TERRAINS_CFG`` heightfield, and the Newton builder
threads it through ``NewtonSceneConfig`` along with the rough-tuned
solver buffers (``nconmax`` / ``njmax`` / ``use_mujoco_contacts`` /
``collision_max_triangle_pairs``).  This is the blind humanoid
locomotion baseline on rough ground — no terrain-scan observation,
matching the Go2 rough preset's blind-walking variant.
"""

from dataclasses import dataclass

from rlworld.rl.configs import NewtonConfigsForRun
from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig


@dataclass
class G1RoughNewtonConfig(G1FlatConfig):
    sim_type: str = "newton"
    run_name: str = "G1_29dof_Rough_Newton"
    use_rough_terrain: bool = True


def get_config() -> NewtonConfigsForRun:
    return G1RoughNewtonConfig().build()
