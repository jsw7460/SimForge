from .common_config_classes import (
    RewardConfig,
    CommandConfig,
    GaitConfig,
    EventConfig,
    PolicyConfig,
    PPOPolicyConfig,
    SACPolicyConfig,
    TD3PolicyConfig,
    FastTD3PolicyConfig,
    NNConfig,
    RunnerConfig,
    VisualizationConfig,
)
from .genesis_config_classes import (
    EnvConfig,
    SceneConfig,
    ObservationConfig,
    ActionConfig,
    CurriculumConfig,
    GenesisConfigsForRun,
)
from .newton_config_classes import (
    NewtonEnvConfig,
    NewtonSceneConfig,
    NewtonObservationConfig,
    NewtonActionConfig,
    NewtonConfigsForRun,
)
from .mujoco_config_classes import (
    MujocoEnvConfig,
    MujocoSceneConfig,
    MujocoObservationConfig,
    MujocoActionConfig,
    MujocoConfigsForRun,
)

ConfigsForRun = GenesisConfigsForRun | NewtonConfigsForRun | MujocoConfigsForRun

# Canonical sim_type → ConfigsForRun class mapping
_CONFIGS_FOR_RUN_MAP: dict[str, type] = {
    "genesis": GenesisConfigsForRun,
    "newton": NewtonConfigsForRun,
    "mujoco": MujocoConfigsForRun,
}


def configs_from_dict(data: dict) -> ConfigsForRun:
    """Create the appropriate ConfigsForRun from a dict.

    Looks for ``sim_type`` first (new convention), then falls back to
    the legacy ``simulator`` key for backward compatibility.
    """
    sim_type = data.get("sim_type") or data.get("simulator")
    if sim_type is None:
        raise ValueError(
            "Cannot determine simulator: dict must contain 'sim_type' or 'simulator' key."
        )

    cls = _CONFIGS_FOR_RUN_MAP.get(sim_type)
    if cls is None:
        raise ValueError(
            f"Unknown sim_type={sim_type!r}. "
            f"Available: {list(_CONFIGS_FOR_RUN_MAP.keys())}"
        )
    return cls.from_dict(data)
