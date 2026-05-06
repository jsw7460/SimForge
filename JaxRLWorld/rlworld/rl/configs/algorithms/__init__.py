from .fast_td3 import FastTD3Config
from .ppo import PPOConfig
from .ppodr3 import PPODR3Config
from .sac import SACConfig
from .scaffolded_tdmpc2 import ScaffoldedTDMPC2Config
from .sim_mpc import SimMPCConfig
from .td3 import TD3Config
from .tdmpc2 import TDMPC2Config

ALGORITHM_CONFIGS = {
    "PPO": PPOConfig,
    "TD3": TD3Config,
    "SAC": SACConfig,
    "FastTD3": FastTD3Config,
    "PPODR3": PPODR3Config,
    "TDMPC2": TDMPC2Config,
    "ScaffoldedTDMPC2": ScaffoldedTDMPC2Config,
    "SimMPC": SimMPCConfig,
}

AlgorithmConfig = (
    PPOConfig
    | TD3Config
    | SACConfig
    | FastTD3Config
    | PPODR3Config
    | TDMPC2Config
    | ScaffoldedTDMPC2Config
    | SimMPCConfig
)


def get_algorithm_config_class(algorithm_name: str):
    if algorithm_name not in ALGORITHM_CONFIGS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    return ALGORITHM_CONFIGS[algorithm_name]
