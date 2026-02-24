from dataclasses import dataclass

from .base import G1FlatMujocoConfig


@dataclass
class G1MLPConfig(G1FlatMujocoConfig):
    actor_class_name: str = "MLPActor"
    run_name: str = "G1_Mujoco_MLP"


def get_config():
    return G1MLPConfig().to_dict()
