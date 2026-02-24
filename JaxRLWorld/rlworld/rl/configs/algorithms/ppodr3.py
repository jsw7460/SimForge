from dataclasses import dataclass, field

from rlworld.rl.configs.algorithms import PPOConfig


@dataclass
class PPODR3Config(PPOConfig):
    """Configuration for PPO with DR3 regularization."""

    algorithm_name: str = field(default="PPODR3")
    dr3_coef: float = field(default=0.001)

    @classmethod
    def from_dict(cls, d: dict) -> "PPODR3Config":
        config = super().from_dict(d)
        config.algorithm_name = "PPODR3"
        return config