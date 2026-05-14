"""RLWorld: Modular RL framework for robot locomotion across simulators."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Algorithms
    "PPO",
    "SAC",
    "TD3",
    "FastTD3",
    "TDMPC2",
    "PolicyType",
    "get_algorithm_class",
    "get_runner_class",
    # Runners
    "BaseRunner",
    "OnPolicyRunner",
    "OffPolicyRunner",
    "ModelBasedRunner",
    # Configs
    "configs_from_dict",
    "GenesisConfigsForRun",
    "NewtonConfigsForRun",
    "MujocoConfigsForRun",
    # Evaluation
    "PolicyEvaluator",
]


def __getattr__(name):
    if name in (
        "PPO",
        "SAC",
        "TD3",
        "FastTD3",
        "TDMPC2",
        "PolicyType",
        "get_algorithm_class",
        "get_runner_class",
    ):
        from rlworld.rl import algorithms

        return getattr(algorithms, name)

    if name in ("BaseRunner", "OnPolicyRunner", "OffPolicyRunner", "ModelBasedRunner"):
        from rlworld.rl import runners

        return getattr(runners, name)

    if name in (
        "configs_from_dict",
        "GenesisConfigsForRun",
        "NewtonConfigsForRun",
        "MujocoConfigsForRun",
    ):
        from rlworld.rl import configs

        return getattr(configs, name)

    if name == "PolicyEvaluator":
        from rlworld.rl.evals import PolicyEvaluator

        return PolicyEvaluator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
