from dataclasses import dataclass

from .base import G1FlatGenesisConfig


@dataclass
class G1MLPConfig(G1FlatGenesisConfig):
    actor_class_name: str = "ABAActor"
    run_name: str = "G1_ABDNet"


def get_config():
    return G1MLPConfig().build()
