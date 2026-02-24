from .common_config_classes import (
    RewardConfig,
    CommandConfig,
    EventConfig,
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

def configs_from_dict(data: dict) -> ConfigsForRun:
    """Factory function to create appropriate config from dict."""
    if data.get("simulator") == "newton":
        return NewtonConfigsForRun.from_dict(data)

    elif data.get("simulator") == "genesis":
        return GenesisConfigsForRun.from_dict(data)

    elif data.get("simulator") == "mujoco":
        return MujocoConfigsForRun.from_dict(data)

    else:
        raise ValueError(f"Unknown simulator {data['simulator']}")