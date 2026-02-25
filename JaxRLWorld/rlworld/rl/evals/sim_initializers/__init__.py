"""SimInitializer framework — strategy pattern for simulator-specific eval setup."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class SimInitializer(ABC):
    """ABC for simulator-specific initialization during evaluation."""

    @abstractmethod
    def init_device(self) -> torch.device:
        """Return the torch device for this simulator."""
        ...

    @abstractmethod
    def prepare_configs(
        self,
        policy_path: str,
        eval_env_cfgs: dict | None,
        extra_overrides: dict | None,
        metadata: dict,
        show_viewer: bool,
        record_video: bool,
        video_dir: str | None,
    ) -> Any:
        """Load saved configs, apply eval overrides, and return ConfigsForRun."""
        ...

    @abstractmethod
    def init_environment(self, eval_cfgs: Any, **kwargs) -> Any:
        """Create and return the evaluation environment."""
        ...

    def start_recording(self, env: Any) -> None:
        """Start video recording (no-op by default)."""
        pass

    def stop_recording(self, env: Any) -> None:
        """Stop video recording (no-op by default)."""
        pass

    def cleanup(self, env: Any) -> None:
        """Cleanup resources (no-op by default)."""
        pass

    @property
    def supports_success_tracking(self) -> bool:
        return False

    @property
    def video_extension(self) -> str:
        return ".mp4"


def detect_sim_type(metadata: dict) -> str:
    """Detect simulator type from checkpoint metadata."""
    env_name = metadata.get('config', {}).get('env', {}).get('env_name', '')
    if "Genesis" in env_name:
        return "Genesis"
    elif "Newton" in env_name:
        return "Newton"
    elif "MjlabEnv" in env_name:
        return "MjlabEnv"
    elif env_name == 'Maniskill':
        return "ManiSkill"
    elif env_name == 'Gymnasium':
        return "Gymnasium"
    else:
        return "Genesis"


def get_initializer(sim_type: str) -> SimInitializer:
    """Lazy-import factory: return the appropriate SimInitializer subclass."""
    if sim_type == "Genesis":
        from .genesis import GenesisInitializer
        return GenesisInitializer()
    elif sim_type == "Newton":
        from .newton import NewtonInitializer
        return NewtonInitializer()
    elif sim_type == "MjlabEnv":
        from .mjlab import MjlabInitializer
        return MjlabInitializer()
    elif sim_type == "ManiSkill":
        from .maniskill import ManiSkillInitializer
        return ManiSkillInitializer()
    elif sim_type == "Gymnasium":
        from .gymnasium import GymnasiumInitializer
        return GymnasiumInitializer()
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")
