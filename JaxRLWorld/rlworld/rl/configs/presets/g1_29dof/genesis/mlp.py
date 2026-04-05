from dataclasses import dataclass

from rlworld.rl.configs import GenesisConfigsForRun
from .base import G1FlatGenesisConfig


@dataclass
class G1MLPConfig(G1FlatGenesisConfig):
    actor_class_name: str = "MLPActor"
    run_name: str = "G1_29Dof_Genesis_MLP"


def get_config() -> GenesisConfigsForRun:
    return G1MLPConfig().build()
