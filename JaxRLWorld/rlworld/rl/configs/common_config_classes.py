from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, TYPE_CHECKING, Literal

from .base_config import BaseConfig
from .rewards import RewardTermConfig

if TYPE_CHECKING:
    from rlworld.rl.envs.mdp.configs import CommandTermConfig


@dataclass
class RewardConfig(BaseConfig):
    """Reward configuration (shared)."""
    reward_terms: dict[str, RewardTermConfig] = None


@dataclass
class CommandConfig(BaseConfig):
    """Command configuration (shared).

    Holds a dict of named CommandTermCfg objects. Each term independently
    manages its own sampling ranges, resampling timer, and post-processing.
    """
    terms: dict[str, Any] = field(default_factory=dict)  # str -> CommandTermCfg


@dataclass
class GaitConfig(BaseConfig):
    """Gait configuration (shared across simulators).

    Two modes:
        ``"fixed"``
            Evenly-spaced phase offsets with constant period.
            No commands needed. Suitable for basic locomotion training.

        ``"command"``
            Frequency and stance duration read from CommandManager each step.
            Per-foot phase offsets produced by ``foot_offset_provider``.
            Use this for gait-conditioned training (e.g. Walk-These-Ways style).

    The ``foot_offset_provider`` is a callable
    ``(CommandManager) -> [num_envs, num_feet]`` that reads whatever commands
    it needs and returns per-foot phase offsets. Pre-built providers:
        - ``QuadrupedOffsets()``:  reads (phase, offset, bound) -> 4-foot offsets
        - ``DirectOffsets(names)``: reads N named commands -> N-foot offsets
    """
    foot_names: tuple[str, ...] | list[str] = field(default_factory=tuple)

    # "fixed" or "command"
    offset_mode: str = "fixed"

    # ── Fixed-mode settings ──
    gait_period: float = 0.8
    default_freq: float = 2.5
    default_duration: float = 0.5

    # ── Command-mode settings ──
    freq_command: str = "gait_freq"
    duration_command: str = "gait_duration"

    # Callable: (CommandManager) -> [num_envs, num_feet].
    # See QuadrupedOffsets and DirectOffsets in managers/common/gait.py.
    foot_offset_provider: Any = None

    # Von Mises smoothing for desired contact states
    contact_smoothing_sigma: float = 0.07


@dataclass
class EventConfig(BaseConfig):
    """Event configuration (shared)."""
    event_terms: list = field(default_factory=list)


@dataclass
class PolicyConfig(BaseConfig):
    """Base policy network configuration — common to all algorithms."""
    actor_class_name: str = "MLPActor"
    actor_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "activation": "elu",
        "ortho_init": True,
        "hidden_dims": [256, 128, 64],
    })
    critic_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "activation": "elu",
        "ortho_init": True,
        "hidden_dims": [256, 128, 64],
    })

    def to(self, target_cls: type) -> "PolicyConfig":
        """Convert to another PolicyConfig subclass, copying common fields."""
        import dataclasses as dc
        base_field_names = {f.name for f in dc.fields(PolicyConfig)}
        kwargs = {k: getattr(self, k) for k in base_field_names}
        return target_cls(**kwargs)

    def recursive_to_dict(self) -> Dict:
        result = super().recursive_to_dict()
        result["_type"] = self.__class__.__name__
        return result


@dataclass
class PPOPolicyConfig(PolicyConfig):
    """PPO policy settings."""
    init_noise_std: float = 1.0
    distribution_type: str = "gaussian"
    std_type: str = "state_independent"


@dataclass
class SACPolicyConfig(PolicyConfig):
    """SAC policy settings."""
    init_noise_std: float = 0.05
    distribution_type: str = "squashed_gaussian"
    log_std_min: float = -20.0
    log_std_max: float = 2.0


@dataclass
class TD3PolicyConfig(PolicyConfig):
    """TD3 policy settings (deterministic — no extra fields)."""
    pass


@dataclass
class FastTD3PolicyConfig(PolicyConfig):
    """FastTD3 policy settings (deterministic — no extra fields)."""
    pass


# Registry for deserialization
_POLICY_CONFIG_CLASSES = {
    "PolicyConfig": PolicyConfig,
    "PPOPolicyConfig": PPOPolicyConfig,
    "SACPolicyConfig": SACPolicyConfig,
    "TD3PolicyConfig": TD3PolicyConfig,
    "FastTD3PolicyConfig": FastTD3PolicyConfig,
}


@dataclass
class NNConfig(BaseConfig):
    """Neural network configuration (shared)."""
    policy: PolicyConfig = field(default_factory=PPOPolicyConfig)
    state_estimator: Dict[str, Any] = field(default_factory=lambda: {
        "activation": "relu",
        "hidden_dims": [256, 128, 64],
    })

    def __post_init__(self):
        if isinstance(self.policy, dict):
            cls_name = self.policy.pop("_type", "PPOPolicyConfig")
            cls = _POLICY_CONFIG_CLASSES.get(cls_name, PPOPolicyConfig)
            self.policy = cls.from_dict(self.policy)


@dataclass
class RunnerConfig(BaseConfig):
    """Runner configuration (shared)."""
    checkpoint: int = -1
    log_interval: int = 1
    max_iterations: int = 99999
    init_at_random_ep_len: bool = False
    resume: bool = False
    resume_path: Optional[str] = None
    run_name: str = ""
    logger: str = "wandb"
    wandb_project: str = "SimForge"
    save_interval: int = 1000
    output_dir: str = "auto"
    upload_checkpoint: bool = False
    delete_local_after_upload: bool = False

    # In-training evaluation
    eval_interval: int = 50  # 0 = disabled
    eval_num_envs: int = 32
    eval_num_episodes: int = 10
    eval_deterministic: bool = True
    eval_disable_noise: bool = True
    eval_disable_interval_events: bool = True


from rlworld.rl.vis.overlays.hud_items.items import HUDItem

@dataclass
class VisualizationConfig(BaseConfig):
    """Visualization configuration (shared)."""
    show_viewer: bool = False
    record_video: bool = False
    video_dir: str = ""
    video_fps: int | None = None
    record_env_ids: list[int] = field(default_factory=lambda: [0])
    grid_layout: bool = True

    # 3D Overlay for Genesis
    enable_command_arrow: bool = True
    command_arrow_radius: float = 0.02
    command_arrow_length_scale: float = 0.5
    max_arrow_length: float = 1.0

    # 2D HUD for Genesis
    enable_text_hud: bool = True
    hud_position: str = "top_left"
    show_base_height: bool = True
    show_command_vel: bool = True
    show_feet_height: bool = True
    show_episode_info: bool = True
    feet_names: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    extra_hud_items: list[HUDItem] = field(default_factory=lambda: [])

    # Viewer type
    viewer_type: Literal["gl", "viser", "rerun", "usd", "file"] = "gl"
    viser_port: int = 8080
    viser_share: bool = True
    rerun_web_port: int = 9191

    # Unified Viser viewer (SimForge custom)
    viser_enable_reward_plots: bool = True
    viser_enable_debug_viz: bool = False