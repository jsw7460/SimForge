from dataclasses import dataclass

from .base import Go1FlatGenesisConfig


@dataclass
class Go1MLPConfig(Go1FlatGenesisConfig):
    actor_class_name: str = "MLPActor"
    run_name: str = "Go1_MLP"


def get_config():
    return Go1MLPConfig().build()
