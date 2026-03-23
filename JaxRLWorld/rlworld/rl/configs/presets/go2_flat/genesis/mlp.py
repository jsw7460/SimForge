from dataclasses import dataclass

from rlworld.rl.configs import GenesisConfigsForRun
from .base import Go2FlatGenesisConfig


@dataclass
class Go2MLPConfig(Go2FlatGenesisConfig):
    actor_class_name: str = "MLPActor"
    run_name: str = "Go2_MLP"


def get_config() -> GenesisConfigsForRun:
    return Go2MLPConfig().build()
