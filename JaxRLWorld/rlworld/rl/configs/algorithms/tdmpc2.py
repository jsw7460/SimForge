from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..base_config import BaseConfig

if TYPE_CHECKING:
    pass


@dataclass
class TDMPC2Config(BaseConfig):
    algorithm_name: str = field(default="TDMPC2")

    # ---- Discount ----
    gamma: float = 0.99
    episode_length: int = 1000
    discount_min: float = 0.95
    discount_max: float = 0.995
    discount_denom: float = 5.0

    # ---- Learning rates ----
    lr: float = 3e-4
    pi_lr: float = 3e-4

    # ---- Target network ----
    tau: float = 0.01

    # ---- Planning (MPPI) ----
    mpc: bool = True
    horizon: int = 3
    num_samples: int = 512
    num_pi_trajs: int = 24
    num_elites: int = 64
    num_iterations: int = 6
    temperature: float = 0.5
    min_std: float = 0.05
    max_std: float = 2.0

    # ---- Loss coefficients ----
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 1e-4
    rho: float = 0.5

    # ---- Discrete regression ----
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    # ---- World model architecture ----
    latent_dim: int = 512
    mlp_dim: int = 512
    num_enc_layers: int = 2
    num_q: int = 5
    simnorm_dim: int = 8
    dropout: float = 0.01
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    squash_action: bool = True

    # ---- Training ----
    batch_size: int = 4096
    buffer_size: int = 1_000_000
    grad_clip_norm: float = 20.0
    max_grad_norm: float = 20.0
    learning_starts: int = 10000
    utd_ratio: int = 1
    num_steps_per_env: int = 1