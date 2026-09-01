from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    obs_normalization: bool = False
    optimizer: str = "adam"
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    learning_starts: int = 100
    batch_size: int = 256
    buffer_size: int = 1_000_000
    # Where the replay buffer's storage lives. "host" keeps it in NumPy,
    # which is right when transitions are born on the host (a CPU
    # Gymnasium env). "device" keeps it on the accelerator, which is
    # right when they are born there — with a GPU simulator the host
    # buffer makes every transition cross twice, down to be stored and
    # back up to be sampled. That costs device memory: 5M transitions
    # across 8192 environments is roughly 4 GB the physics no longer has,
    # and vision observations make it far worse. Default "host" because
    # it is what every existing preset was measured on.
    replay_buffer_device: str = "host"

    # TD3-specific parameters
    policy_delay: int = 2
    exploration_noise: float = 0.0
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    num_steps_per_env: int = 1

    num_gradient_steps: int = 1
    max_grad_norm: float = 10.0
