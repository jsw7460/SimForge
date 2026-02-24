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
from .default_config import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_SCENE_CONFIG,
    DEFAULT_CURRICULUM_CONFIG,
    DEFAULT_OBS_CONFIG,
    DEFAULT_ACT_CONFIG,
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
    env_name: str = field(default=DEFAULT_ENV_CONFIG["env_name"])
    task_name: str = field(default=DEFAULT_ENV_CONFIG["task_name"])
    gym_make_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_envs: int = field(default=DEFAULT_ENV_CONFIG["num_envs"])
    decimation: int = field(default=1)
    seed: int = field(default=DEFAULT_ENV_CONFIG["seed"])
    termination_criteria: List["TerminationTermConfig"] = field(
        default_factory=lambda: DEFAULT_ENV_CONFIG["termination_criteria"])
    episode_length_s: float = field(default=DEFAULT_ENV_CONFIG["episode_length_s"])


@dataclass
class SceneConfig(BaseConfig):
    """Genesis scene configuration."""
    sim_options: gs.options.SimOptions = field(default_factory=gs.options.SimOptions)
    viewer_options: gs.options.ViewerOptions = field(default_factory=gs.options.ViewerOptions)
    vis_options: gs.options.VisOptions = field(default_factory=gs.options.VisOptions)
    rigid_options: gs.options.RigidOptions = field(default_factory=gs.options.RigidOptions)
    env_spacing: tuple[float, float] = field(default=DEFAULT_SCENE_CONFIG["env_spacing"])
    entities: list[EntityConfig] = field(default_factory=lambda: DEFAULT_SCENE_CONFIG["entities"])
    sensors: list[SensorConfig] | None = field(default_factory=lambda: DEFAULT_SCENE_CONFIG["sensors"])
    robot_cfg: Union["RobotConfig", None] = None


@dataclass
class ObservationConfig(BaseConfig):
    """Genesis observation configuration."""
    obs_group: dict[str, list[ObservationTermConfig]] = None
    robot_state_dim: int = field(default=DEFAULT_OBS_CONFIG.get("robot_state_dim", 0))
    short_history_len: int = field(default=DEFAULT_OBS_CONFIG["short_history_len"])
    max_history_len: int = field(default=DEFAULT_OBS_CONFIG["max_history_len"])
    use_vision: bool = field(default=DEFAULT_OBS_CONFIG["use_vision"])
    use_height_map: bool = field(default=DEFAULT_OBS_CONFIG["use_height_map"])
    horizontal_scale: float = field(default=DEFAULT_OBS_CONFIG["horizontal_scale"])
    map_size: int = field(default=DEFAULT_OBS_CONFIG["map_size"])
    map_resolution: float = field(default=DEFAULT_OBS_CONFIG["map_resolution"])
    base_lookat: List[float] = field(default_factory=lambda: DEFAULT_OBS_CONFIG["base_lookat"])
    cam_fov: int = field(default=DEFAULT_OBS_CONFIG["camera_fov"])
    cam_base_offset: List[float] = field(default_factory=lambda: DEFAULT_OBS_CONFIG["cam_base_offset"])
    cam_GUI: bool = field(default=DEFAULT_OBS_CONFIG["cam_GUI"])
    cam_resolution: Tuple[int, int] = field(default_factory=lambda: DEFAULT_OBS_CONFIG["cam_resolution"])


@dataclass
class ActionConfig(BaseConfig):
    """Genesis action configuration."""
    actuated_dof_names: list[str] = field(default_factory=lambda: DEFAULT_ACT_CONFIG["actuated_dof_names"])
    num_joint_actions: int = field(default=DEFAULT_ACT_CONFIG["num_joint_actions"])
    action_scale: float = field(default=DEFAULT_ACT_CONFIG["action_scale"])
    simulate_action_latency: bool = field(default=DEFAULT_ACT_CONFIG["simulate_action_latency"])
    clip_actions: tuple[float, float] | dict[str, tuple[float, float]] | Literal["joint_limit"] | None = field(
        default=DEFAULT_ACT_CONFIG["clip_actions"])
    offset: dict[str, float] = field(default_factory=lambda: DEFAULT_ACT_CONFIG["offset"])
    control_mode: Literal["position", "force"] = "position"


@dataclass
class CurriculumConfig(BaseConfig):
    """Curriculum configuration."""
    enable: bool = field(default=DEFAULT_CURRICULUM_CONFIG["enable"])
    initial_level: int = field(default=DEFAULT_CURRICULUM_CONFIG["initial_level"])
    max_level: int = field(default=DEFAULT_CURRICULUM_CONFIG["max_level"])
    success_threshold: float = field(default=DEFAULT_CURRICULUM_CONFIG["success_threshold"])
    min_steps_per_level: int = field(default=DEFAULT_CURRICULUM_CONFIG["min_steps_per_level"])
    eval_window_size: int = field(default=DEFAULT_CURRICULUM_CONFIG["eval_window_size"])
    curriculum_components: Dict[str, Dict[int, Any]] = field(
        default_factory=lambda: DEFAULT_CURRICULUM_CONFIG["curriculum_components"]
    )
    criterion: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_CURRICULUM_CONFIG["criterion"]
    )

    def __post_init__(self):
        levels = set(range(self.max_level + 1))
        for component_dict in self.curriculum_components.values():
            if set(component_dict.keys()) != levels:
                raise ValueError(f"Inconsistent level configuration across parameters")

        if "command_ranges" in self.curriculum_components:
            required_commands = {"lin_vel_x", "lin_vel_y", "ang_vel"}
            for level, ranges in self.curriculum_components["command_ranges"].items():
                if not all(cmd in ranges for cmd in required_commands):
                    raise ValueError(f"Missing required commands in level {level}: {required_commands}")


@dataclass
class GenesisConfigsForRun(BaseConfig):
    """Complete configuration for Genesis training runs."""
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
