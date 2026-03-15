from .ppo import PPO
from .td3 import TD3
from .sac import SAC
from .ppo_dr3 import PPODR3
from .fast_td3 import FastTD3
from .tdmpc2 import TDMPC2
from .scaffolded_tdmpc2 import ScaffoldedTDMPC2
from .sim_mpc import SimMPC


from enum import Enum


class PolicyType(Enum):
    ON_POLICY = "on_policy"
    OFF_POLICY = "off_policy"
    MODEL_BASED = "model_based"


# Add algorithm here
ALGORITHM_REGISTRY = {
    "PPO": {
        "class": "rlworld.rl.algorithms.ppo.PPO",
        "policy_type": PolicyType.ON_POLICY,
    },
    "PPODR3": {
        "class": "rlworld.rl.algorithms.ppo_dr3.PPODR3",
        "policy_type": PolicyType.ON_POLICY,
    },
    "SAC": {
        "class": "rlworld.rl.algorithms.sac.SAC",
        "policy_type": PolicyType.OFF_POLICY,
    },
    "TD3": {
        "class": "rlworld.rl.algorithms.td3.TD3",
        "policy_type": PolicyType.OFF_POLICY,
    },
    "FastTD3": {
        "class": "rlworld.rl.algorithms.fast_td3.FastTD3",
        "policy_type": PolicyType.OFF_POLICY,
    },
    "TDMPC2": {
        "class": "rlworld.rl.algorithms.tdmpc2.TDMPC2",
        "policy_type": PolicyType.MODEL_BASED,
    },
    "ScaffoldedTDMPC2": {
        "class": "rlworld.rl.algorithms.scaffolded_tdmpc2.ScaffoldedTDMPC2",
        "policy_type": PolicyType.MODEL_BASED,
    },
    "SimMPC": {
        "class": "rlworld.rl.algorithms.sim_mpc.SimMPC",
        "policy_type": PolicyType.MODEL_BASED,
    },
}

RUNNER_MAP = {
    PolicyType.ON_POLICY: "rlworld.rl.runners.on_policy_runner.OnPolicyRunner",
    PolicyType.OFF_POLICY: "rlworld.rl.runners.off_policy_runner.OffPolicyRunner",
    PolicyType.MODEL_BASED: "rlworld.rl.runners.model_based_runner.ModelBasedRunner",
}

# Algorithm-specific runner overrides (takes priority over RUNNER_MAP)
ALGORITHM_RUNNER_OVERRIDES = {
    "ScaffoldedTDMPC2": "rlworld.rl.runners.privileged_model_based_runner.PrivilegedModelBasedRunner",
    "SimMPC": "rlworld.rl.runners.sim_mpc_runner.SimMPCRunner",
}


def get_algorithm_info(algorithm_name: str) -> dict:
    if algorithm_name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[algorithm_name]


def get_policy_type(algorithm_name: str) -> PolicyType:
    return get_algorithm_info(algorithm_name)["policy_type"]


def get_runner_class(algorithm_name: str):
    # Check algorithm-specific override first
    if algorithm_name in ALGORITHM_RUNNER_OVERRIDES:
        runner_path = ALGORITHM_RUNNER_OVERRIDES[algorithm_name]
    else:
        policy_type = get_policy_type(algorithm_name)
        runner_path = RUNNER_MAP[policy_type]

    module_path, class_name = runner_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def get_algorithm_class(algorithm_name: str):
    info = get_algorithm_info(algorithm_name)
    module_path, class_name = info["class"].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)