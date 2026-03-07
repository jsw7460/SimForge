from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Union, TYPE_CHECKING

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
    from rlworld.rl.configs.scene import NewtonEntityConfig
    from rlworld.rl.configs.sensors.newton_sensor_config import NewtonSensorConfig
    from rlworld.rl.envs.mdp.configs import TerminationTermConfig
    from rlworld.rl.configs.observations import ObservationTermConfig
    from rlworld.rl.configs.robots.base import RobotConfig


@dataclass
class NewtonEnvConfig(BaseConfig):
    """Newton environment configuration."""
    num_envs: int = 4096
    env_name: str = "NewtonEnv"
    task_name: str = "Unknown"
    seed: int = 42
    episode_length_s: float = 20.0
    decimation: int = 1
    termination_criteria: list["TerminationTermConfig"] = field(default_factory=list)


@dataclass
class NewtonSceneConfig(BaseConfig):
    """Newton scene configuration."""
    dt: float = 0.02
    substeps: int = 4
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    solver_type: Literal["mujoco"] = "mujoco"  # Currently, only support mujoco solver
    entities: list["NewtonEntityConfig"] = field(default_factory=list)
    sensors: list["NewtonSensorConfig"] | None = None
    add_ground: bool = True
    env_spacing: tuple[float, float, float] = (2.0, 2.0, 0.0)
    robot_cfg: Union["RobotConfig", None] = None


@dataclass
class NewtonObservationConfig(BaseConfig):
    """Newton observation configuration."""
    obs_group: dict[str, list["ObservationTermConfig"]] = field(default_factory=dict)
    enable_noise: bool = True


@dataclass
class NewtonActionConfig(BaseConfig):
    """Newton action configuration."""
    actuated_dof_names: list[str] = field(default_factory=list)
    action_scale: float | Dict[str, float] = 0.25
    clip_actions: tuple[float, float] | dict[str, tuple[float, float]] | Literal["joint_limit"] | None = (-1.0, 1.0)
    offset: dict[str, float] = field(default_factory=dict)


@dataclass
class NewtonConfigsForRun(BaseConfig):
    """Complete configuration for Newton training runs."""
    env: NewtonEnvConfig = field(default_factory=NewtonEnvConfig)
    scene: NewtonSceneConfig = field(default_factory=NewtonSceneConfig)
    observation: NewtonObservationConfig = field(default_factory=NewtonObservationConfig)
    action: NewtonActionConfig = field(default_factory=NewtonActionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    event: EventConfig = field(default_factory=EventConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NewtonConfigsForRun":
        """Create NewtonConfigsForRun from dictionary."""
        env = config_dict.get("env", NewtonEnvConfig())
        scene = config_dict.get("scene", NewtonSceneConfig())
        observation = config_dict.get("observation", NewtonObservationConfig())
        action = config_dict.get("action", NewtonActionConfig())
        visualization = config_dict.get("visualization", VisualizationConfig())
        reward = config_dict.get("reward", RewardConfig())
        command = config_dict.get("command", CommandConfig())
        event = config_dict.get("event", EventConfig())
        nn = config_dict.get("nn", NNConfig())
        runner = config_dict.get("runner", RunnerConfig())

        # Handle dataclass or dict
        if isinstance(env, dict):
            env = NewtonEnvConfig(**env)
        if isinstance(scene, dict):
            scene = NewtonSceneConfig(**scene)
        if isinstance(observation, dict):
            observation = NewtonObservationConfig(**observation)
        if isinstance(action, dict):
            action = NewtonActionConfig(**action)
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
