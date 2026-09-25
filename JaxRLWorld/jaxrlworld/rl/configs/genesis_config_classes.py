from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Literal, Union

import genesis as gs

from .algorithms import AlgorithmConfig, PPOConfig, get_algorithm_config_class
from .base_config import BaseConfig
from .common_config_classes import (
    CommandConfig,
    EventConfig,
    GaitConfig,
    NNConfig,
    RewardConfig,
    RunnerConfig,
    VisualizationConfig,
)
from .scene.terrain_config import TerrainCfg
from .sensors import SensorConfig

if TYPE_CHECKING:
    from jaxrlworld.rl.configs import CurriculumManagerConfig
    from jaxrlworld.rl.configs.robots.base import RobotConfig
    from jaxrlworld.rl.configs.sensors import ContactSensorCfg


def _default_curriculum_cfg() -> "CurriculumManagerConfig":
    """Lazy default to avoid importing CurriculumManagerConfig at module load."""
    from jaxrlworld.rl.configs import CurriculumManagerConfig

    return CurriculumManagerConfig()


@dataclass
class EnvConfig(BaseConfig):
    """Genesis environment configuration."""

    env_name: str = "World"
    task_name: str = "Unknown"
    gym_make_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_envs: int = 10000
    decimation: int = 1
    seed: int = 42
    terminations: Any = None  # TerminationsConfig instance, set by preset
    episode_length_s: float = 20.0
    # Passed to ``gs.init(performance_mode=...)`` — compiles kernels
    # against the static FIELD array backend (shape-fixed), which steps
    # faster but forbids scene edits after build. Measured on go2_gait
    # @4096 envs: engine substep 3.01 -> 1.84 ms with reward parity
    # bit-consistent and zero-copy intact, so True is the default; a
    # tool that edits the scene after build sets False. Only honored by
    # the process's FIRST Genesis env (``gs.init`` runs once per process).
    performance_mode: bool = True
    # Run the per-substep contact capture + contact-timing update through
    # ``torch.compile`` (``GenesisContactBatch``). It is plain tensor math
    # on tensors already handed out of the engine — nothing engine-side is
    # traced — and eager it is ~75 tiny launches per substep; fused it is
    # a handful. ``found`` and the timing buffers are bit-identical to the
    # eager path; the link-frame force can differ in its last bits (fused
    # multiply-add, reduction order). False keeps the eager path.
    compile_contact_kernels: bool = True


@dataclass
class SceneConfig(BaseConfig):
    """Genesis scene configuration."""

    _EXCLUDE_FROM_SERIALIZATION = ("sim_options", "viewer_options", "vis_options", "rigid_options", "robot_cfg")

    sim_options: gs.options.SimOptions = field(default_factory=gs.options.SimOptions)
    viewer_options: gs.options.ViewerOptions = field(default_factory=gs.options.ViewerOptions)
    vis_options: gs.options.VisOptions = field(default_factory=gs.options.VisOptions)
    rigid_options: gs.options.RigidOptions = field(default_factory=gs.options.RigidOptions)
    env_spacing: tuple[float, float] = (20.0, 20.0)
    entities: dict = field(default_factory=dict)
    # Passive rigid objects (no actuated joints) — graspable objects, props,
    # static fixtures. Read via ``get_rigid_object_data(name)``. Empty by default.
    rigid_objects: dict = field(default_factory=dict)
    sensors: list[SensorConfig] | None = field(default_factory=list)
    # Simulator-agnostic contact sensor configs (``ContactSensorCfg``).
    contact_sensors: "list[ContactSensorCfg] | None" = None
    # Simulator-agnostic cameras (shared with mjlab and Newton).
    cameras: tuple = ()
    robot_cfg: Union["RobotConfig", None] = None
    # Terrain (flat plane by default; generator → heightfield) — owned by
    # the per-sim TerrainImporter the scene manager constructs.
    terrain_cfg: TerrainCfg = field(default_factory=lambda: TerrainCfg(terrain_type="plane"))


@dataclass
class ObservationConfig(BaseConfig):
    """Genesis observation configuration.

    Groups are named attributes of type ObservationGroupConfig.
    Per-group noise gating lives on each :class:`ObservationGroupConfig`'s
    ``enable_corruption`` field. Use :func:`disable_corruption` to silence
    every group at once for eval / test flows.

    Subclass and add groups::

        @dataclass
        class MyObsCfg(ObservationConfig):
            actor: ActorObsCfg = field(default_factory=ActorObsCfg)
            critic: CriticObsCfg = field(default_factory=CriticObsCfg)
    """

    pass


@dataclass
class ActionConfig(BaseConfig):
    """Genesis action configuration."""

    actuated_dof_names: list[str] = field(default_factory=list)
    action_scale: float | dict[str, float] | Literal["joint_limit"] = 0.4
    clip_actions: tuple[float, float] | dict[str, tuple[float, float]] | Literal["joint_limit"] | None = (-100.0, 100.0)
    offset: dict[str, float] | Literal["joint_limit_center"] = field(default_factory=dict)
    settle_steps: int = 0
    # Soft-limit factor for the "joint_limit" / "joint_limit_center"
    # auto modes (see ActionManagerBaseConfig.joint_limit_soft_factor).
    joint_limit_soft_factor: float = 0.9
    # Optional term-based action system (see jaxrlworld/rl/envs/mdp/actions/).
    action_terms: "dict[str, Any] | None" = None


@dataclass
class GenesisConfigsForRun(BaseConfig):
    """Complete configuration for Genesis training runs."""

    sim_type: str = "genesis"
    preset_module: str | None = None  # "jaxrlworld.rl.configs.presets.go2.mlp"
    preset_class_name: str | None = None
    preset_kwargs: Dict[str, Any] | None = None
    env: EnvConfig = field(default_factory=EnvConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    event: EventConfig = field(default_factory=EventConfig)
    gait: "GaitConfig | None" = None
    curriculum: "CurriculumManagerConfig" = field(default_factory=lambda: _default_curriculum_cfg())
    algorithm: AlgorithmConfig = field(default_factory=PPOConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    IMMUTABLE_SETTINGS = {
        "env": ["dof_names"],
        "command": ["num_commands"],
        "storage": ["action_shape", "actor_obs_shape", "estimator_obs_shape", "robot_state_shape"],
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
        observation = _get_or_convert("observation", ObservationConfig, ObservationConfig)
        visualization = _get_or_convert("visualization", VisualizationConfig, VisualizationConfig)
        action = _get_or_convert("action", ActionConfig, ActionConfig)
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
            visualization=visualization,
            observation=observation,
            action=action,
            reward=reward,
            event=event,
            gait=gait,
            command=command,
            algorithm=algorithm,
            nn=nn,
            runner=runner,
        )
