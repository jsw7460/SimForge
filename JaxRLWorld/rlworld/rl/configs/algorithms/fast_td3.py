from dataclasses import dataclass, field

from .td3 import TD3Config


@dataclass
class FastTD3Config(TD3Config):
    algorithm_name: str = field(default="FastTD3")

    # Use squashed gaussian or not
    is_squashed: bool = field(default=True)

    # Normalization
    obs_normalization: bool = field(default=False)

    # Override defaults for FastTD3
    batch_size: int = 32768

    # Distributional RL (C51)
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0

    # Mixed exploration noise
    noise_min: float = 0.01
    noise_max: float = 0.05

    # Clipped Double Q-learning
    use_cdq: bool = True