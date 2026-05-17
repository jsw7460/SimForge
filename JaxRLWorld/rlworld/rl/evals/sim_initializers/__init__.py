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

    def create_play_scene(self, env: Any):
        """Build the viser ``PlayScene`` for this simulator's interactive viewer.

        Subclasses that support the interactive viewer override this to
        return a ``PlayScene`` instance (typically ``BridgePlayScene``
        wrapping a sim-specific ``SimulatorBridge``, or a backend-native
        scene like ``MujocoPlayScene``). The default raises so callers
        get a clear error for unsupported simulators (ManiSkill,
        Gymnasium).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support the interactive "
            f"viser viewer. Use evaluator.evaluate() instead of "
            f"evaluator.play()."
        )

    def try_stop_mid_episode_recording(self, env: Any, target_steps: int) -> bool:
        """Optionally stop recording before episode ends, return whether stopped.

        Hook for backends whose recording API records as the env steps
        (e.g. Genesis writes a video file frame-by-frame). Such backends
        need to call ``stop_recording`` after a fixed number of env
        steps rather than at episode end. Default: never stops mid-episode.
        Returns ``True`` if recording was stopped on this call so the
        caller can clear its ``record_steps`` counter.
        """
        return False

    @property
    def supports_success_tracking(self) -> bool:
        return False

    @property
    def video_extension(self) -> str:
        return ".mp4"


def _detect_robot_key(metadata: dict) -> str:
    """Best-effort robot identifier from checkpoint metadata.

    Purely for labelling — the cross-sim config resolver does NOT use
    this; it resolves via the ``preset_module`` / ``preset_class_name``
    fields instead.
    """
    config = metadata.get("config", {})
    task_name = config.get("env", {}).get("task_name", "").lower()
    action_cfg = config.get("action", {})

    if "go2" in task_name:
        return "go2"
    if "g1" in task_name:
        return "g1_29dof"
    if "t1" in task_name:
        return "t1"

    dof_names = action_cfg.get("actuated_dof_names", [])
    if len(dof_names) == 12:
        return "go2"
    if len(dof_names) == 29:
        return "g1_29dof"
    if len(dof_names) in (23, 24):
        return "t1"

    num_actions = action_cfg.get("num_joint_actions", 0)
    raise ValueError(f"Cannot detect robot from checkpoint (task_name={task_name!r}, num_actions={num_actions}).")


def resolve_cross_sim_config(metadata: dict, target_sim: str):
    """Auto-resolve a ConfigsForRun for the target simulator.

    Both resolution paths rely on metadata injected by
    :meth:`PresetConfig.build`:

    - ``preset_module``: fully-qualified module holding the preset class
    - ``preset_class_name``: the class within that module
    - ``preset_kwargs``: non-default constructor kwargs captured at build

    Strategy:
        1. **Path substitution** — for sim-specific subclasses whose
           module path contains a simulator segment (e.g.
           ``go2_flat.newton.gait_conditioned``). Swap the segment for
           *target_sim* and call ``get_config()`` on the new module.
        2. **Class reinstantiation** — for unified configs (single
           class, ``sim_type`` field selects the backend). Import the
           stored class, re-build with ``sim_type=target_sim`` and the
           preserved ``preset_kwargs``.

    Args:
        metadata: Checkpoint metadata dict.
        target_sim: Target simulator name ("genesis", "newton", "mujoco").

    Returns:
        A ConfigsForRun object for the target simulator and robot.
    """
    import importlib
    from dataclasses import fields, is_dataclass

    sim_key = target_sim.lower()
    if sim_key in ("mjlabenv", "mjlab"):
        sim_key = "mujoco"

    _SIM_NAMES = {"genesis", "newton", "mujoco"}

    config_dict = metadata.get("config", {})
    preset_module_path = config_dict.get("preset_module")
    preset_class_name = config_dict.get("preset_class_name")

    # --- Strategy 1: path substitution (sim-specific subclass) ---
    if preset_module_path is not None:
        parts = preset_module_path.split(".")
        sim_idx = next((i for i, p in enumerate(parts) if p in _SIM_NAMES), None)
        if sim_idx is not None:
            parts[sim_idx] = sim_key
            target_module_path = ".".join(parts)
            try:
                mod = importlib.import_module(target_module_path)
                return mod.get_config()
            except ModuleNotFoundError:
                pass  # fall through

    # --- Strategy 2: unified-config reinstantiation ---
    if preset_module_path and preset_class_name:
        try:
            mod = importlib.import_module(preset_module_path)
            cls = getattr(mod, preset_class_name, None)
        except ModuleNotFoundError:
            cls = None
        if cls is not None and is_dataclass(cls):
            field_names = {f.name for f in fields(cls)}
            if "sim_type" in field_names:
                kwargs = {k: v for k, v in config_dict.get("preset_kwargs", {}).items() if k in field_names}
                kwargs["sim_type"] = sim_key
                return cls(**kwargs).build()

    raise ValueError(
        f"Could not resolve cross-sim config for target_sim={sim_key!r}. "
        f"Checkpoint preset_module={preset_module_path!r}, "
        f"preset_class_name={preset_class_name!r}. "
        "Checkpoint may predate the unified-config metadata fields "
        "(preset_module / preset_class_name / preset_kwargs). Retrain or "
        "pass eval_cfgs manually."
    )


def detect_sim_type(metadata: dict) -> str:
    """Detect simulator type from checkpoint metadata."""
    # Multisim checkpoint (new format)
    train_sims = metadata.get("train_sim_names")
    if train_sims and len(train_sims) > 1:
        return "MultiSim(" + "+".join(train_sims) + ")"

    env_name = metadata.get("config", {}).get("env", {}).get("env_name", "")
    if "Genesis" in env_name:
        return "Genesis"
    elif "Newton" in env_name:
        return "Newton"
    elif "MujocoEnv" in env_name:
        return "MujocoEnv"
    elif env_name == "Maniskill":
        return "ManiSkill"
    elif env_name == "Gymnasium":
        return "Gymnasium"

    # Fallback: check sim_type field directly
    sim_type = metadata.get("sim_type", "")
    if sim_type == "genesis":
        return "Genesis"
    elif sim_type == "newton":
        return "Newton"
    elif sim_type in ("mujoco", "mjlab"):
        return "MujocoEnv"

    return "Unknown"


def get_initializer(sim_type: str) -> SimInitializer:
    """Lazy-import factory: return the appropriate SimInitializer subclass."""
    if sim_type == "Genesis":
        from .genesis import GenesisInitializer

        return GenesisInitializer()
    elif sim_type == "Newton":
        from .newton import NewtonInitializer

        return NewtonInitializer()
    elif sim_type == "MujocoEnv":
        from .mjlab import MujocoInitializer

        return MujocoInitializer()
    elif sim_type == "ManiSkill":
        from .maniskill import ManiSkillInitializer

        return ManiSkillInitializer()
    elif sim_type == "Gymnasium":
        from .gymnasium import GymnasiumInitializer

        return GymnasiumInitializer()
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")
