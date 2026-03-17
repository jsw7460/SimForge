from dataclasses import dataclass

from rlworld.rl.configs import NewtonConfigsForRun
from .base import G1FlatNewtonConfig


@dataclass
class G1MLPConfig(G1FlatNewtonConfig):
    actor_class_name: str = "ABAActor"
    run_name: str = "G1_29Dof_NT_ABDNet"


def get_config() -> NewtonConfigsForRun:
    return G1MLPConfig().build()
