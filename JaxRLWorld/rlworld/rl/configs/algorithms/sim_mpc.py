from dataclasses import dataclass, field

from ..base_config import BaseConfig


@dataclass
class SimMPCConfig(BaseConfig):
    algorithm_name: str = field(default="SimMPC")

    # ---- Planning (MPPI) ----
    horizon: int = 5
    num_samples: int = 512
    num_pi_trajs: int = 64
    num_elites: int = 64
    num_iterations: int = 6
    temperature: float = 0.5
    min_std: float = 0.05
    max_std: float = 2.0

    # ---- Discount ----
    gamma: float = 0.99

    # ---- Learning rates ----
    lr: float = 3e-4
    pi_lr: float = 3e-4

    # ---- Target network ----
    tau: float = 0.005

    # ---- Networks ----
    hidden_dims: tuple = (512, 256)
    num_q: int = 5
    squash_policy: bool = False

    # ---- Training ----
    batch_size: int = 4096
    buffer_size: int = 1_000_000
    learning_starts: int = 5000
    num_gradient_steps: int = 1
    num_steps_per_env: int = 1

    # ---- MPPI/Policy mixing ----
    # Fraction of envs that use MPPI planning (rest use policy directly).
    # E.g., 0.1 with 100 envs → 10 envs MPPI, 90 envs policy.
    # Set to 1.0 to use MPPI for all envs (default, slow but highest quality).
    mppi_ratio: float = 1.0

    # ---- Episode ----
    episode_length: int = 1000
