from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..base_config import BaseConfig
from ..default_config import (
    DEFAULT_ALGORITHM_CONFIG,
)

if TYPE_CHECKING:
    pass


@dataclass
class PPOConfig(BaseConfig):
    algorithm_name: str = field(default="PPO")
    clip_param: float = field(default=DEFAULT_ALGORITHM_CONFIG["clip_param"])
    use_early_stop: bool = field(default=False)
    desired_kl: float = field(default=DEFAULT_ALGORITHM_CONFIG["desired_kl"])
    entropy_coef: float = field(default=DEFAULT_ALGORITHM_CONFIG["entropy_coef"])
    gamma: float = field(default=DEFAULT_ALGORITHM_CONFIG["gamma"])
    lam: float = field(default=DEFAULT_ALGORITHM_CONFIG["lam"])
    actor_lr: float = field(default=DEFAULT_ALGORITHM_CONFIG["actor_lr"])
    critic_lr: float = field(default=DEFAULT_ALGORITHM_CONFIG["critic_lr"])
    estimator_learning_rate: float = field(default=DEFAULT_ALGORITHM_CONFIG["estimator_learning_rate"])
    max_grad_norm: float = field(default=DEFAULT_ALGORITHM_CONFIG["max_grad_norm"])
    num_learning_epochs: int = field(default=DEFAULT_ALGORITHM_CONFIG["num_learning_epochs"])
    num_mini_batches: int = field(default=DEFAULT_ALGORITHM_CONFIG["num_mini_batches"])
    schedule: str = field(default=DEFAULT_ALGORITHM_CONFIG["schedule"])
    use_clipped_value_loss: bool = field(default=DEFAULT_ALGORITHM_CONFIG["use_clipped_value_loss"])
    use_reward_scaling: bool = field(default=True)
    value_loss_coef: float = field(default=DEFAULT_ALGORITHM_CONFIG["value_loss_coef"])
    use_truth_value_for_actor: bool = field(default=DEFAULT_ALGORITHM_CONFIG["use_truth_value_for_actor"])
    use_truth_value_for_critic: bool = field(default=DEFAULT_ALGORITHM_CONFIG["use_truth_value_for_critic"])
    use_barrier_style: bool = field(default=DEFAULT_ALGORITHM_CONFIG["use_barrier_style"])
    use_sde: bool = field(default=DEFAULT_ALGORITHM_CONFIG["use_sde"])
    sde_sample_freq: int = field(default=DEFAULT_ALGORITHM_CONFIG["sde_sample_freq"])
    learning_starts: int = field(default=DEFAULT_ALGORITHM_CONFIG["learning_starts"])
    num_steps_per_env: int = field(default=DEFAULT_ALGORITHM_CONFIG["num_steps_per_env"])
    obs_normalization: bool = field(default=False)