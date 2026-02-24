from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from ..base_config import BaseConfig

if TYPE_CHECKING:
    pass


@dataclass
class TD3Config(BaseConfig):
    algorithm_name: str = field(default="TD3")
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 1
    tau: float = 0.005
    use_obs_norm: bool = False
    optimizer: str = "adam"
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    learning_starts: int = 100
    batch_size: int = 256
    buffer_size: int = 1_000_000

    # TD3-specific parameters
    policy_delay: int = 2
    exploration_noise: float = 0.0
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    num_steps_per_env: int = 1

    utd_ratio: float = 1.0
    max_grad_norm: float = 10.0

