from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, TYPE_CHECKING, Literal

from .base_config import BaseConfig
from .rewards import RewardTermConfig

if TYPE_CHECKING:
    from rlworld.rl.envs.mdp.configs import CommandTermConfig


@dataclass
class RewardConfig(BaseConfig):
    """Reward configuration (shared)."""
    reward_terms: list[RewardTermConfig] = None


@dataclass
class CommandConfig(BaseConfig):
    """Command configuration (shared)."""
    sampler: list["CommandTermConfig"] = field(default_factory=tuple)
    resampling_time_s: tuple[float, float] = (8.0, 12.0)

    # Standing environment fraction: this fraction of envs will have
    # all commands zeroed out on each resample.
    rel_standing_envs: float = 0.0

    # Heading command: when enabled, a heading target is sampled and
    # ang_vel_z is overwritten with P-control toward the heading target.
    heading_command: bool = False
    heading_control_stiffness: float = 0.5
    heading_range: tuple[float, float] = (-3.14, 3.14)

    # Fraction of envs that use heading control (rest use raw ang_vel_z).
    # Only effective when heading_command=True.
    rel_heading_envs: float = 1.0


@dataclass
class EventConfig(BaseConfig):
    """Event configuration (shared)."""
    event_terms: list = field(default_factory=list)


@dataclass
class NNConfig(BaseConfig):
    """Neural network configuration (shared)."""
    policy: Dict[str, Any] = field(default_factory=lambda: {
        "actor_class": None,
        "activation": "tanh",
        "actor_hidden_dims": [128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "init_noise_std": 1.0,
        "distribution_type": "gaussian",
        "std_type": "fixed",
    })
    state_estimator: Dict[str, Any] = field(default_factory=lambda: {
        "activation": "relu",
        "hidden_dims": [256, 128, 64],
    })


@dataclass
class RunnerConfig(BaseConfig):
    """Runner configuration (shared)."""
    checkpoint: int = -1
    experiment_name: str = "GoAnywhere"
    load_run: str = None
    log_interval: int = 1
    max_iterations: int = 99999
    init_at_random_ep_len: bool = False
    policy_class_name: str = "PPOActorCritic"
    state_estimator_class_name: str = "StateEstimator"
    low_level_path: str = None
    high_level_update_freq: int = 1
    record_interval: int = -1
    resume: bool = False
    resume_path: Optional[str] = None
    run_name: str = ""
    logger: str = "wandb"
    wandb_project: str = "RLArchitecture"
    runner_class_name: str = "runner_class_name"
    save_interval: int = 1000
    output_dir: str = "auto"
    upload_checkpoint: bool = True
    delete_local_after_upload: bool = False


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

    # Viewer type for Newton
    viewer_type: Literal["gl", "viser", "rerun", "usd", "file"] = "gl"
    viser_port: int = 8080
    viser_share: bool = True
    rerun_web_port: int = 9191