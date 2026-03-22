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
        extra_overrides: dict | None,
        metadata: dict,
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


# Registry: (robot_key, sim_type) -> module path for get_config()
# robot_key is derived from checkpoint task_name (lowercased, keywords matched)
_PRESET_REGISTRY: dict[tuple[str, str], str] = {
    # Go2
    ("go2", "genesis"):  "rlworld.rl.configs.presets.go2_flat.genesis.mlp",
    ("go2", "newton"):   "rlworld.rl.configs.presets.go2_flat.newton.mlp",
    ("go2", "mujoco"):   "rlworld.rl.configs.presets.go2_flat.mujoco.mlp",
    # G1 29-DOF
    ("g1_29dof", "genesis"): "rlworld.rl.configs.presets.g1_29dof.genesis.mlp",
    ("g1_29dof", "newton"):  "rlworld.rl.configs.presets.g1_29dof.newton.mlp",
    ("g1_29dof", "mujoco"):  "rlworld.rl.configs.presets.g1_29dof.mujoco.mlp",
    # G1 12-DOF
    ("g1_12dof", "genesis"): "rlworld.rl.configs.presets.g1_12dof.genesis.mlp",
    ("g1_12dof", "newton"):  "rlworld.rl.configs.presets.g1_12dof.newton.mlp",
    # Go1
    ("go1", "genesis"): "rlworld.rl.configs.presets.go1.genesis.mlp",
    ("go1", "newton"):  "rlworld.rl.configs.presets.go1.newton.mlp",
    ("go1", "mujoco"):  "rlworld.rl.configs.presets.go1.mujoco.mlp",
}


def _detect_robot_key(metadata: dict) -> str:
    """Detect robot key from checkpoint metadata for preset lookup."""
    config = metadata.get("config", {})
    task_name = config.get("env", {}).get("task_name", "").lower()
    # Also check action dim as a heuristic
    action_cfg = config.get("action", {})
    num_actions = action_cfg.get("num_joint_actions", 0)

    if "go2" in task_name:
        return "go2"
    elif "go1" in task_name:
        return "go1"
    elif "g1" in task_name:
        num_dofs = len(action_cfg.get("actuated_dof_names", []))
        if num_actions >= 29 or num_dofs >= 29 or "29" in task_name:
            return "g1_29dof"
        else:
            return "g1_12dof"

    # Fallback: check dof_names length
    dof_names = action_cfg.get("actuated_dof_names", [])
    if len(dof_names) == 12:
        return "go2"  # Go2 has 12 DOFs
    elif len(dof_names) == 29:
        return "g1_29dof"

    raise ValueError(
        f"Cannot detect robot from checkpoint (task_name={task_name!r}, "
        f"num_actions={num_actions}). Please provide eval_cfgs manually."
    )


def resolve_cross_sim_config(metadata: dict, target_sim: str):
    """Auto-resolve a ConfigsForRun for the target simulator.

    Uses the checkpoint metadata to detect the robot, then loads the
    appropriate preset config for that robot on the target simulator.

    Args:
        metadata: Checkpoint metadata dict.
        target_sim: Target simulator name ("genesis", "newton", "mujoco").

    Returns:
        A ConfigsForRun object for the target simulator and robot.
    """
    import importlib

    robot_key = _detect_robot_key(metadata)
    sim_key = target_sim.lower()
    # Normalize sim names
    if sim_key in ("mjlabenv", "mjlab"):
        sim_key = "mujoco"

    key = (robot_key, sim_key)
    if key not in _PRESET_REGISTRY:
        available = [
            f"{r}/{s}" for (r, s) in _PRESET_REGISTRY if r == robot_key
        ]
        raise ValueError(
            f"No preset found for robot={robot_key!r} on sim={sim_key!r}. "
            f"Available for {robot_key}: {available}. "
            f"Please provide eval_cfgs manually."
        )

    module_path = _PRESET_REGISTRY[key]
    mod = importlib.import_module(module_path)
    return mod.get_config()


def detect_sim_type(metadata: dict) -> str:
    """Detect simulator type from checkpoint metadata."""
    # Multisim checkpoint (new format)
    train_sims = metadata.get("train_sim_names")
    if train_sims and len(train_sims) > 1:
        return "MultiSim(" + "+".join(train_sims) + ")"

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

    # Fallback: check sim_type field directly
    sim_type = metadata.get("sim_type", "")
    if sim_type == "genesis":
        return "Genesis"
    elif sim_type == "newton":
        return "Newton"
    elif sim_type in ("mujoco", "mjlab"):
        return "MjlabEnv"

    return "Unknown"


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
