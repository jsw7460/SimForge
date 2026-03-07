from dataclasses import dataclass

from .base import G1FlatNewtonConfig


@dataclass
class Go1MLPConfig(G1FlatNewtonConfig):
    actor_class_name: str = "MLPActor"
    run_name: str = "G1_12Dof_MLP"


def get_config():
    return Go1MLPConfig().build()
