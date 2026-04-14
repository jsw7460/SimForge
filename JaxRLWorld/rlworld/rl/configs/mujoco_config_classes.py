from dataclasses import dataclass, field
from typing import Dict, Any, TYPE_CHECKING, Literal

from .algorithms import AlgorithmConfig, get_algorithm_config_class
from .base_config import BaseConfig
from .common_config_classes import (
    RewardConfig,
    CommandConfig,
    GaitConfig,
    EventConfig,
    NNConfig,
    RunnerConfig,
    VisualizationConfig,
)
if TYPE_CHECKING:
    from rlworld.rl.envs.mdp.configs import (
        CurriculumManagerConfig,
        TerminationTermConfig,
    )
    from rlworld.rl.configs.observations import ObservationTermConfig


def _default_curriculum_cfg() -> "CurriculumManagerConfig":
    """Lazy default to avoid importing CurriculumManagerConfig at module load."""
    from rlworld.rl.envs.mdp.configs import CurriculumManagerConfig
    return CurriculumManagerConfig()


@dataclass
class MujocoEnvConfig(BaseConfig):
    """MuJoCo/mjlab environment configuration."""
    num_envs: int = 4096
    env_name: str = "MujocoEnv"
    task_name: str = "Unknown"
    seed: int = 42
    episode_length_s: float = 20.0
    decimation: int = 10
    terminations: Any = None  # TerminationsConfig instance, set by preset


@dataclass
class MujocoSceneConfig(BaseConfig):
    """MuJoCo/mjlab scene configuration.

    Config-level fields only — no mjlab imports needed.
    The scene manager converts these to mjlab objects internally.
    """
    physics_dt: float = 0.002
    substeps: int = 1
    num_envs: int = 4096
    env_spacing: float = 2.0
    robot_entity_name: str = "robot"

    # Entities — unified EntityCfg dict (scene manager converts to mjlab)
    entities: Any = None  # dict[str, EntityCfg | GroundPlaneCfg]

    # Sensors — mjlab sensor config objects (passed through to SceneCfg)
    sensors: tuple = ()

    # Terrain
    terrain_type: str = "plane"

    # Solver settings
    solver_iterations: int = 10
    solver_ls_iterations: int = 20
    ccd_iterations: int = 50
    nconmax: int | None = 35
    njmax: int | None = 1500
    impratio: float = 1.0
    cone: Literal["pyramidal", "elliptic"] = "pyramidal"
    contact_sensor_maxmatch: int = 64

    # Preset info for auto-resolving non-serializable mjlab objects at eval time
    preset_class_name: str | None = None
    preset_module_path: str | None = None

    # Legacy — will be removed. Use entities/sensors/solver fields instead.
    mjlab_scene_cfg: Any = None
    mjlab_sim_cfg: Any = None
    unified_entities: Any = None

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result.pop('mjlab_scene_cfg', None)
        result.pop('mjlab_sim_cfg', None)
        result.pop('unified_entities', None)
        return result


@dataclass
class MujocoObservationConfig(BaseConfig):
    """MuJoCo/mjlab observation configuration. Groups are named ObservationGroupConfig attributes."""
    enable_noise: bool = True


@dataclass
class MujocoActionConfig(BaseConfig):
    """MuJoCo/mjlab action configuration."""
    entity_name: str = "robot"
    actuated_dof_names: list[str] = field(default_factory=list)
    action_scale: float | dict[str, float] = 0.25
    clip_actions: tuple[float, float] | dict[str, tuple[float, float]] | Literal["joint_limit"] | None = (-1.0, 1.0)
    offset: dict[str, float] = field(default_factory=dict)
    settle_steps: int = 0


@dataclass
class MujocoConfigsForRun(BaseConfig):
    """Complete configuration for MuJoCo/mjlab training runs."""
    sim_type: str = "mujoco"
    preset_module: str | None = None
    preset_class_name: str | None = None
    preset_kwargs: Dict[str, Any] | None = None
    env: MujocoEnvConfig = field(default_factory=MujocoEnvConfig)
    scene: MujocoSceneConfig = field(default_factory=MujocoSceneConfig)
    observation: MujocoObservationConfig = field(default_factory=MujocoObservationConfig)
    action: MujocoActionConfig = field(default_factory=MujocoActionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    event: EventConfig = field(default_factory=EventConfig)
    gait: "GaitConfig | None" = None
    curriculum: "CurriculumManagerConfig" = field(
        default_factory=lambda: _default_curriculum_cfg()
    )
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MujocoConfigsForRun":
        def _get_or_convert(key, config_cls, default_factory):
            val = config_dict.get(key, default_factory())
            if isinstance(val, dict):
                return config_cls.from_dict(val)
            return val

        env = _get_or_convert("env", MujocoEnvConfig, MujocoEnvConfig)
        scene = _get_or_convert("scene", MujocoSceneConfig, MujocoSceneConfig)
        observation = _get_or_convert("observation", MujocoObservationConfig, MujocoObservationConfig)
        action = _get_or_convert("action", MujocoActionConfig, MujocoActionConfig)
        visualization = _get_or_convert("visualization", VisualizationConfig, VisualizationConfig)
        reward = _get_or_convert("reward", RewardConfig, RewardConfig)
        command = _get_or_convert("command", CommandConfig, CommandConfig)
        event = _get_or_convert("event", EventConfig, EventConfig)
        gait_val = config_dict.get("gait", None)
        if isinstance(gait_val, dict):
            gait = GaitConfig.from_dict(gait_val)
        else:
            gait = gait_val
        nn = _get_or_convert("nn", NNConfig, NNConfig)
        runner = _get_or_convert("runner", RunnerConfig, RunnerConfig)

        algo_val = config_dict.get("algorithm", {})
        if isinstance(algo_val, dict):
            algo_name = algo_val.get("algorithm_name", "PPO")
            algo_config_cls = get_algorithm_config_class(algo_name)
            algorithm = algo_config_cls.from_dict(algo_val)
        else:
            algorithm = algo_val

        return cls(
            env=env,
            scene=scene,
            observation=observation,
            action=action,
            visualization=visualization,
            reward=reward,
            command=command,
            event=event,
            gait=gait,
            algorithm=algorithm,
            nn=nn,
            runner=runner,
        )
