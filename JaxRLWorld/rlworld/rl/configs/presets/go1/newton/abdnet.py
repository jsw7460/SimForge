from dataclasses import dataclass

from .base import Go1FlatNewtonConfig


@dataclass
class Go1MLPConfig(Go1FlatNewtonConfig):
    actor_class_name: str = "ABAActor"
    run_name: str = "Go1_ABA"


def get_config():
    return Go1MLPConfig().build()
