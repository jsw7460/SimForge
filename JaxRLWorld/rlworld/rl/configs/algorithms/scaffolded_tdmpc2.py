from dataclasses import dataclass, field

from .tdmpc2 import TDMPC2Config


@dataclass
class ScaffoldedTDMPC2Config(TDMPC2Config):
    algorithm_name: str = field(default="ScaffoldedTDMPC2")

    warmup_std: float = 1.0

    # ---- ABD-Net architecture ----
    link_channels: int = 8
    spatial_dim: int = 6
    learnable_contribution_weight: bool = False
    use_positive_constraint: bool = True
    residual_scale_init: float = 0.1

    # ---- Scaffolding ----
    explore_ratio: float = 0.5
    ortho_coef: float = 0.01
