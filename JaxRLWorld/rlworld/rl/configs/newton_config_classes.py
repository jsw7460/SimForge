from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Literal, Union

from rlworld.rl.configs.scene import GroundPlaneCfg

from .algorithms import AlgorithmConfig, get_algorithm_config_class
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

if TYPE_CHECKING:
    from rlworld.rl.configs import (
        CurriculumManagerConfig,
    )
    from rlworld.rl.configs.robots.base import RobotConfig
    from rlworld.rl.configs.scene import EntityCfg, NewtonEntityConfig, TerrainCfg
    from rlworld.rl.configs.sensors.contact_sensor_config import ContactSensorCfg
    from rlworld.rl.configs.sensors.newton_sensor_config import NewtonSensorConfig


def _default_curriculum_cfg() -> "CurriculumManagerConfig":
    """Lazy default to avoid importing CurriculumManagerConfig at module load."""
    from rlworld.rl.configs import CurriculumManagerConfig

    return CurriculumManagerConfig()


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
class SolverMuJoCoCfg(BaseConfig):
    """Configuration for ``newton.solvers.SolverMuJoCo``.

    Fields mirror ``SolverMuJoCo.__init__`` 1:1. Any field set to ``None``
    defers to Newton's documented 3-tier priority: ctor arg > the model's
    ``mujoco.<option>`` custom attribute > MuJoCo's built-in default.

    Defaults below follow Newton's canonical humanoid-locomotion recipe
    from ``newton/examples/robot/example_robot_g1.py`` (elliptic friction
    cone, implicitfast integrator, 100 / 50 iterations, impratio=100),
    which is what the repository's locomotion-style presets actually want.
    """

    # Core algorithm choices
    solver: Literal["newton", "cg"] | None = "newton"
    integrator: Literal["implicitfast", "euler", "rk4"] | None = "implicitfast"
    cone: Literal["pyramidal", "elliptic"] | None = "elliptic"

    # Iteration budgets
    iterations: int | None = 100
    ls_iterations: int | None = 50

    # Contact / constraint buffer sizes
    njmax: int | None = 1500
    nconmax: int | None = 150

    # Friction cone tuning
    impratio: float | None = 100.0

    # Solver tolerances (None → MuJoCo defaults: 1e-8 / 0.01 / 1e-6)
    tolerance: float | None = None
    ls_tolerance: float | None = None
    ccd_tolerance: float | None = None

    # Advanced iteration caps (None → MuJoCo defaults: 35 / 10)
    ccd_iterations: int | None = None
    sdf_iterations: int | None = None

    # Mode flags
    ls_parallel: bool = True
    use_mujoco_contacts: bool = True
    use_mujoco_cpu: bool = False
    enable_multiccd: bool = False
    disable_contacts: bool = False


@dataclass
class NewtonSceneConfig(BaseConfig):
    """Newton scene configuration."""

    _EXCLUDE_FROM_SERIALIZATION = ("robot_cfg",)

    dt: float = 0.02
    substeps: int = 4
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    solver_type: Literal["mujoco"] = "mujoco"  # Currently, only support mujoco solver
    solver_cfg: SolverMuJoCoCfg = field(default_factory=SolverMuJoCoCfg)
    entities: dict[str, Union["EntityCfg", "NewtonEntityConfig", "GroundPlaneCfg", "TerrainCfg"]] = field(
        default_factory=list
    )
    sensors: list["NewtonSensorConfig"] | None = None
    # Simulator-agnostic contact sensors (shared with Genesis / mjlab).
    contact_sensors: "list[ContactSensorCfg] | None" = None
    add_ground: bool = True
    env_spacing: tuple[float, float, float] = (2.0, 2.0, 0.0)
    robot_cfg: Union["RobotConfig", None] = None
    # Collision broad-phase triangle-pair budget. ``None`` keeps Newton's
    # default (1e6); rough-terrain scenes set ``num_envs * per_env`` so
    # heightfield contacts aren't silently dropped (see the Newton scene
    # manager and the go2 rough preset).
    collision_max_triangle_pairs: int | None = None


@dataclass
class NewtonObservationConfig(BaseConfig):
    """Newton observation configuration. Groups are named ObservationGroupConfig attributes.

    Per-group noise gating lives on each :class:`ObservationGroupConfig`'s
    ``enable_corruption`` field. Use :func:`disable_corruption` to silence
    every group at once for eval / test flows.
    """

    pass


@dataclass
class NewtonActionConfig(BaseConfig):
    """Newton action configuration."""

    actuated_dof_names: list[str] = field(default_factory=list)
    action_scale: float | Dict[str, float] = 0.25
    clip_actions: tuple[float, float] | dict[str, tuple[float, float]] | Literal["joint_limit"] | None = (-1.0, 1.0)
    offset: dict[str, float] = field(default_factory=dict)
    settle_steps: int = 0
    # Optional term-based action system (see rlworld/rl/envs/mdp/actions/).
    # When provided, ``scale``/``clip``/``offset`` above are ignored and
    # each term is instantiated by the action manager. Used by tasks
    # that need relative / settle-relative / composite actions.
    action_terms: "dict[str, Any] | None" = None


@dataclass
class NewtonConfigsForRun(BaseConfig):
    """Complete configuration for Newton training runs."""

    sim_type: str = "newton"
    preset_module: str | None = None
    preset_class_name: str | None = None
    preset_kwargs: Dict[str, Any] | None = None
    env: NewtonEnvConfig = field(default_factory=NewtonEnvConfig)
    scene: NewtonSceneConfig = field(default_factory=NewtonSceneConfig)
    observation: NewtonObservationConfig = field(default_factory=NewtonObservationConfig)
    action: NewtonActionConfig = field(default_factory=NewtonActionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    event: EventConfig = field(default_factory=EventConfig)
    gait: "GaitConfig | None" = None
    curriculum: "CurriculumManagerConfig" = field(default_factory=lambda: _default_curriculum_cfg())
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
