from .commands.command_term_config import CommandTermConfig
from .common_config_classes import (
    CommandConfig,
    EventConfig,
    FastTD3PolicyConfig,
    GaitConfig,
    NNConfig,
    PolicyConfig,
    PPOPolicyConfig,
    RewardConfig,
    RunnerConfig,
    SACPolicyConfig,
    TD3PolicyConfig,
    VisualizationConfig,
)
from .curriculums.curriculum_term_config import (
    CurriculumManagerConfig,
    CurriculumTermConfig,
)
from .genesis_config_classes import (
    ActionConfig,
    EnvConfig,
    GenesisConfigsForRun,
    ObservationConfig,
    SceneConfig,
)
from .mujoco_config_classes import (
    MujocoActionConfig,
    MujocoConfigsForRun,
    MujocoEnvConfig,
    MujocoObservationConfig,
    MujocoSceneConfig,
)
from .newton_config_classes import (
    NewtonActionConfig,
    NewtonConfigsForRun,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    SolverMuJoCoCfg,
)

# Term-level configs hoisted from the old rlworld.rl.envs.mdp.configs
# location so callers can do `from rlworld.rl.configs import ...` directly.
from .terminations.termination_term_config import TerminationResult, TerminationTermConfig

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
        raise ValueError("Cannot determine simulator: dict must contain 'sim_type' or 'simulator' key.")

    cls = _CONFIGS_FOR_RUN_MAP.get(sim_type)
    if cls is None:
        raise ValueError(f"Unknown sim_type={sim_type!r}. Available: {list(_CONFIGS_FOR_RUN_MAP.keys())}")
    return cls.from_dict(data)
