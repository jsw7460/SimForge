from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from ..base_config import BaseConfig

if TYPE_CHECKING:
    pass


@dataclass
class SACConfig(BaseConfig):
    algorithm_name: str = field(default="SAC")
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 1
    tau: float = 0.005
    ent_coef: Union[str, float] = "auto"
    target_entropy: Union[str, float] = "auto"
    obs_normalization: bool = False
    optimizer: str = "adam"
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    learning_starts: int = 100
    batch_size: int = 512
    buffer_size: int = 1_000_000
    policy_delay: int = 1
    num_gradient_steps: int = 1
    num_steps_per_env: int =  1
