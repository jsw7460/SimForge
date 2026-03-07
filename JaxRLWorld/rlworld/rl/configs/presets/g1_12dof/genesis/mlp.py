from dataclasses import dataclass

from .base import G1FlatGenesisConfig


@dataclass
class G1MLPConfig(G1FlatGenesisConfig):
    actor_class_name: str = "MLPActor"
    run_name: str = "G1_MLP"


def get_config():
    return G1MLPConfig().build()
