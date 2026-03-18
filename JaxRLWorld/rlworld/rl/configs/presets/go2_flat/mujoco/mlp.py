from dataclasses import dataclass

from rlworld.rl.configs import MujocoConfigsForRun
from .base import Go2FlatMujocoConfig


@dataclass
class Go2MLPConfig(Go2FlatMujocoConfig):
    actor_class_name: str = "MLPActor"
    run_name: str = "Go2_Mujoco_MLP"


def get_config() -> MujocoConfigsForRun:
    return Go2MLPConfig().build()
