from dataclasses import dataclass, field
from typing import Dict, Any, TYPE_CHECKING, Literal

from .algorithms import AlgorithmConfig, get_algorithm_config_class
from .base_config import BaseConfig
from .common_config_classes import (
    RewardConfig,
    CommandConfig,
    EventConfig,
    NNConfig,
    RunnerConfig,
    VisualizationConfig,
)
if TYPE_CHECKING:
    from rlworld.rl.envs.mdp.configs import TerminationTermConfig
    from rlworld.rl.configs.observations import ObservationTermConfig


@dataclass
class MujocoEnvConfig(BaseConfig):
    """MuJoCo/mjlab environment configuration."""
    num_envs: int = 4096
    env_name: str = "MujocoEnv"
    task_name: str = "Unknown"
    seed: int = 42
    episode_length_s: float = 20.0
    decimation: int = 10
    termination_criteria: list["TerminationTermConfig"] = field(default_factory=list)


@dataclass
class MujocoSceneConfig(BaseConfig):
    """MuJoCo/mjlab scene configuration.

    This wraps mjlab's SceneCfg for use with rlworld.
    """
    physics_dt: float = 0.002  # 2ms physics timestep (500Hz)
    num_envs: int = 4096
    env_spacing: float = 2.0

    # mjlab SceneCfg will be passed directly
    mjlab_scene_cfg: Any = None  # mjlab.SceneCfg
    mjlab_sim_cfg: Any = None  # mjlab.SimulationCfg

    # Entity configuration (alternative to mjlab_scene_cfg)
    robot_entity_name: str = "robot"

    # Preset info for auto-resolving non-serializable mjlab objects at eval time
    preset_class_name: str | None = None
    preset_module_path: str | None = None

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        # Exclude non-serializable mjlab objects (contain lambdas, etc.)
        result.pop('mjlab_scene_cfg', None)
        result.pop('mjlab_sim_cfg', None)
        return result


@dataclass
class MujocoObservationConfig(BaseConfig):
    """MuJoCo/mjlab observation configuration."""
    obs_group: dict[str, list["ObservationTermConfig"]] = field(default_factory=dict)
    short_history_len: int = 1
    max_history_len: int = 1


@dataclass
class MujocoActionConfig(BaseConfig):
    """MuJoCo/mjlab action configuration."""
    entity_name: str = "robot"
    actuated_dof_names: list[str] = field(default_factory=list)
    action_scale: float | dict[str, float] = 0.25
    clip_actions: tuple[float, float] | dict[str, tuple[float, float]] | Literal["joint_limit"] | None = (-1.0, 1.0)
    offset: dict[str, float] = field(default_factory=dict)


@dataclass
class MujocoConfigsForRun(BaseConfig):
    """Complete configuration for MuJoCo/mjlab training runs."""
    env: MujocoEnvConfig = field(default_factory=MujocoEnvConfig)
    scene: MujocoSceneConfig = field(default_factory=MujocoSceneConfig)
    observation: MujocoObservationConfig = field(default_factory=MujocoObservationConfig)
    action: MujocoActionConfig = field(default_factory=MujocoActionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    event: EventConfig = field(default_factory=EventConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MujocoConfigsForRun":
        """Create MujocoConfigsForRun from dictionary."""
        env = config_dict.get("env", MujocoEnvConfig())
        scene = config_dict.get("scene", MujocoSceneConfig())
        observation = config_dict.get("observation", MujocoObservationConfig())
        action = config_dict.get("action", MujocoActionConfig())
        visualization = config_dict.get("visualization", VisualizationConfig())
        reward = config_dict.get("reward", RewardConfig())
        command = config_dict.get("command", CommandConfig())
        event = config_dict.get("event", EventConfig())
        nn = config_dict.get("nn", NNConfig())
        runner = config_dict.get("runner", RunnerConfig())

        # Handle dataclass or dict
        if isinstance(env, dict):
            env = MujocoEnvConfig(**env)
        if isinstance(scene, dict):
            scene = MujocoSceneConfig(**scene)
        if isinstance(observation, dict):
            observation = MujocoObservationConfig(**observation)
        if isinstance(action, dict):
            action = MujocoActionConfig(**action)
        if isinstance(visualization, dict):
            visualization = VisualizationConfig(**visualization)
        if isinstance(reward, dict):
            reward = RewardConfig(**reward)
        if isinstance(command, dict):
            command = CommandConfig(**command)
        if isinstance(event, dict):
            event = EventConfig(**event)
        if isinstance(nn, dict):
            nn = NNConfig(**nn)
        if isinstance(runner, dict):
            runner = RunnerConfig(**runner)

        # Algorithm config dispatch
        algo_dict = config_dict["algorithm"]
        algo_name = algo_dict["algorithm_name"]
        algo_config_cls = get_algorithm_config_class(algo_name)
        algorithm = algo_config_cls.from_dict(algo_dict)

        return cls(
            env=env,
            scene=scene,
            observation=observation,
            action=action,
            visualization=visualization,
            reward=reward,
            command=command,
            event=event,
            algorithm=algorithm,
            nn=nn,
            runner=runner,
        )
