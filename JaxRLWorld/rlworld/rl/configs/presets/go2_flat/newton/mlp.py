from rlworld.rl.configs import NewtonConfigsForRun
from .base import Go2FlatNewtonConfig
from dataclasses import dataclass

@dataclass
class Go2FlatNewtonMLPConfig(Go2FlatNewtonConfig):
    actor_class_name: str = "MLPActor"
    run_name: str = "Go2_Newton_MLP"


def get_config() -> NewtonConfigsForRun:
    """Complete configuration for Go2 flat terrain with MLP actor on Newton."""
    return Go2FlatNewtonMLPConfig().build()
