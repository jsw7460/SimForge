from dataclasses import dataclass

from rlworld.rl.configs import NewtonConfigsForRun
from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig


@dataclass
class G1MLPConfig(G1FlatConfig):
    sim_type: str = "newton"
    actor_class_name: str = "ABAActor"
    run_name: str = "G1_29Dof_NT_ABDNet"


def get_config() -> NewtonConfigsForRun:
    return G1MLPConfig().build()
