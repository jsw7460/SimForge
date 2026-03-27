from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, Union, TYPE_CHECKING, Literal

import genesis as gs
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
from .observations import ObservationTermConfig
from .scene import EntityConfig
from .sensors import SensorConfig

if TYPE_CHECKING:
    from rlworld.rl.envs.mdp.configs import TerminationTermConfig
    from rlworld.rl.configs.robots.base import RobotConfig


@dataclass
class EnvConfig(BaseConfig):
    """Genesis environment configuration."""
    env_name: str = "World"
    task_name: str = "Unknown"
    gym_make_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_envs: int = 10000
    decimation: int = 1
    seed: int = 42
    termination_criteria: list["TerminationTermConfig"] = field(default_factory=list)
    episode_length_s: float = 20.0


@dataclass
class SceneConfig(BaseConfig):
    """Genesis scene configuration."""
    sim_options: gs.options.SimOptions = field(default_factory=gs.options.SimOptions)
    viewer_options: gs.options.ViewerOptions = field(default_factory=gs.options.ViewerOptions)
    vis_options: gs.options.VisOptions = field(default_factory=gs.options.VisOptions)
    rigid_options: gs.options.RigidOptions = field(default_factory=gs.options.RigidOptions)
    env_spacing: tuple[float, float] = (20.0, 20.0)
    entities: list[EntityConfig] = field(default_factory=list)
    sensors: list[SensorConfig] | None = field(default_factory=list)
    robot_cfg: Union["RobotConfig", None] = None


@dataclass
class ObservationConfig(BaseConfig):
    """Genesis observation configuration."""
    obs_group: dict[str, list[ObservationTermConfig]] = None
    enable_noise: bool = True


@dataclass
class ActionConfig(BaseConfig):
    """Genesis action configuration."""
    actuated_dof_names: list[str] = field(default_factory=list)
    num_joint_actions: int = 12
    action_scale: float | dict[str, float] = 0.4
    simulate_action_latency: bool = False
    clip_actions: tuple[float, float] | dict[str, tuple[float, float]] | Literal["joint_limit"] | None = (-100.0, 100.0)
    offset: dict[str, float] = field(default_factory=dict)
    control_mode: Literal["position", "force"] = "position"
    actuator_cfg: "ActuatorBaseCfg | None" = None


@dataclass
class CurriculumConfig(BaseConfig):
    """Curriculum configuration."""
    enable: bool = False
    initial_level: int = 0
    max_level: int = 3
    success_threshold: float = 0.8
    min_steps_per_level: int = 50000
    eval_window_size: int = 2
    curriculum_components: Dict[str, Dict[int, Any]] = field(default_factory=dict)
    criterion: Dict[str, float] = field(default_factory=dict)


@dataclass
class GenesisConfigsForRun(BaseConfig):
    """Complete configuration for Genesis training runs."""
    sim_type: str = "genesis"
    env: EnvConfig = field(default_factory=EnvConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    event: EventConfig = field(default_factory=EventConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    IMMUTABLE_SETTINGS = {
        'env': ['dof_names'],
        'command': ['num_commands'],
        'storage': ['action_shape', 'actor_obs_shape', 'estimator_obs_shape', 'robot_state_shape'],
    }

    @classmethod
    def from_dict(cls, config_dict: Dict):
        # Helper function
        def _get_or_convert(key, config_cls, default_factory):
            val = config_dict.get(key, default_factory())
            if isinstance(val, dict):
                return config_cls.from_dict(val)
            return val

        env = _get_or_convert("env", EnvConfig, EnvConfig)
        scene = _get_or_convert("scene", SceneConfig, SceneConfig)
        curriculum = _get_or_convert("curriculum", CurriculumConfig, CurriculumConfig)
        observation = _get_or_convert("observation", ObservationConfig, ObservationConfig)
        visualization = _get_or_convert("visualization", VisualizationConfig, VisualizationConfig)
        action = _get_or_convert("action", ActionConfig, ActionConfig)
        reward = _get_or_convert("reward", RewardConfig, RewardConfig)
        command = _get_or_convert("command", CommandConfig, CommandConfig)
        event = _get_or_convert("event", EventConfig, EventConfig)
        nn = _get_or_convert("nn", NNConfig, NNConfig)
        runner = _get_or_convert("runner", RunnerConfig, RunnerConfig)

        # Algorithm config dispatch
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
            curriculum=curriculum,
            visualization=visualization,
            observation=observation,
            action=action,
            reward=reward,
            event=event,
            command=command,
            algorithm=algorithm,
            nn=nn,
            runner=runner,
        )

    def merge_with_new_config(self, new_config: 'GenesisConfigsForRun') -> 'GenesisConfigsForRun':
        """
        Safely merge current config with new config, ensuring critical parameters remain unchanged.

        Args:
            new_config: New configuration to merge with

        Returns:
            Merged configuration

        Raises:
            ValueError: If any immutable setting would be changed by the merge
        """
        # First, validate immutable settings
        self._validate_immutable_settings(new_config)

        # Create merged config starting with current config
        merged_dict = self.recursive_to_dict()
        new_dict = new_config.recursive_to_dict()

        # Update with new config, skipping immutable settings
        for config_type, params in new_dict.items():
            if isinstance(params, dict):
                if config_type not in merged_dict:
                    merged_dict[config_type] = {}

                for param_name, value in params.items():
                    if not self._is_immutable(config_type, param_name):
                        merged_dict[config_type][param_name] = value

        return GenesisConfigsForRun.from_dict(merged_dict)

    def _is_immutable(self, config_type: str, param_name: str) -> bool:
        """Check if a parameter is in the immutable settings list."""
        if config_type not in self.IMMUTABLE_SETTINGS:
            return False

        return param_name in self.IMMUTABLE_SETTINGS[config_type]

    def _validate_immutable_settings(self, new_config: 'GenesisConfigsForRun'):
        """
        Validate that no immutable settings would be changed by the merge.

        Raises:
            ValueError: If any immutable setting would be changed
        """
        current_dict = self.recursive_to_dict()
        new_dict = new_config.recursive_to_dict()

        for config_type, immutable_params in self.IMMUTABLE_SETTINGS.items():
            if config_type not in new_dict:
                continue

            for param in immutable_params:
                if param not in new_dict[config_type]:
                    continue

                if new_dict[config_type][param] != current_dict[config_type][param]:
                    raise ValueError(
                        f"Cannot change immutable setting {config_type}.{param} "
                        f"from {current_dict[config_type][param]} to {new_dict[config_type][param]}"
                    )