from dataclasses import dataclass

from rlworld.rl.configs import MujocoConfigsForRun
from .base import G1FlatMujocoConfig


@dataclass
class G1MLPConfig(G1FlatMujocoConfig):
    actor_class_name: str = "MLPActor"
    run_name: str = "G1_29Dof_Mujoco_MLP"


def get_config() -> MujocoConfigsForRun:
    return G1MLPConfig().build()
