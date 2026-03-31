from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Union, TYPE_CHECKING

from rlworld.rl.configs.scene import GroundPlaneCfg
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
    terminations: Any = None  # TerminationsConfig instance, set by preset


@dataclass
class NewtonSceneConfig(BaseConfig):
    """Newton scene configuration."""
    _EXCLUDE_FROM_SERIALIZATION = ("robot_cfg",)

    dt: float = 0.02
    substeps: int = 4
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    solver_type: Literal["mujoco"] = "mujoco"  # Currently, only support mujoco solver
    entities: dict[str, Union["NewtonEntityConfig", "GroundPlaneCfg"]] = field(default_factory=list)
    sensors: list["NewtonSensorConfig"] | None = None
    add_ground: bool = True
    env_spacing: tuple[float, float, float] = (2.0, 2.0, 0.0)
    robot_cfg: Union["RobotConfig", None] = None


@dataclass
class NewtonObservationConfig(BaseConfig):
    """Newton observation configuration. Groups are named ObservationGroupConfig attributes."""
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
    sim_type: str = "newton"
    preset_module: str | None = None
    env: NewtonEnvConfig = field(default_factory=NewtonEnvConfig)
    scene: NewtonSceneConfig = field(default_factory=NewtonSceneConfig)
    observation: NewtonObservationConfig = field(default_factory=NewtonObservationConfig)
    action: NewtonActionConfig = field(default_factory=NewtonActionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    event: EventConfig = field(default_factory=EventConfig)
    gait: "GaitConfig | None" = None
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NewtonConfigsForRun":
        def _get_or_convert(key, config_cls, default_factory):
            val = config_dict.get(key, default_factory())
            if isinstance(val, dict):
                return config_cls.from_dict(val)
            return val

        env = _get_or_convert("env", NewtonEnvConfig, NewtonEnvConfig)
        scene = _get_or_convert("scene", NewtonSceneConfig, NewtonSceneConfig)
        observation = _get_or_convert("observation", NewtonObservationConfig, NewtonObservationConfig)
        action = _get_or_convert("action", NewtonActionConfig, NewtonActionConfig)
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
