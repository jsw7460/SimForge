import os
from typing import TYPE_CHECKING, Optional

from rlworld.rl.utils.yaml_io import load_yaml

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from rlworld.rl.configs import ConfigsForRun
    from rlworld.rl.runners import BaseRunner


def load_checkpoint_metadata(checkpoint_path: str) -> dict:
    """Load config and train state from a checkpoint directory.

    Returns a merged dict: ``{**train_state, "config": config_dict}``.
    """
    config_path = os.path.join(checkpoint_path, "config.yaml")
    train_state_path = os.path.join(checkpoint_path, "train_state.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(train_state_path):
        raise FileNotFoundError(f"Train state file not found: {train_state_path}")

    config = load_yaml(config_path)
    train_state = load_yaml(train_state_path)

    return {**train_state, "config": config}


def load_config_from_checkpoint(metadata: dict) -> "ConfigsForRun":
    """Reconstruct config from checkpoint by re-running the preset.

    Uses ``preset_module`` to re-instantiate the full config (with all nested
    objects properly created).  The config YAML is **not** merged back — it
    exists only for logging.  This follows the IsaacLab pattern: the preset
    is the single source of truth for config structure.
    """
    import importlib

    config_dict = metadata.get("config", {})
    preset_module_path = config_dict.get("preset_module")

    if preset_module_path is None:
        raise ValueError(
            "Checkpoint missing 'preset_module' in config.yaml. "
            "This checkpoint was saved before preset_module tracking was added."
        )

    preset_mod = importlib.import_module(preset_module_path)
    return preset_mod.get_config()


def load_runner(
    env: "World",
    checkpoint_path: str,
    cfgs: Optional["ConfigsForRun"] = None,
    use_wandb: bool = False,
) -> "BaseRunner":
    """Load a runner from checkpoint, automatically detecting the runner class."""
    metadata = load_checkpoint_metadata(checkpoint_path)

    runner_class_name = metadata.get("runner_class")
    if runner_class_name is None:
        raise ValueError(
            "Checkpoint missing 'runner_class' in train_state.yaml."
        )

    from rlworld.rl import runners

    if not hasattr(runners, runner_class_name):
        raise ValueError(
            f"Unknown runner class: {runner_class_name}. "
            f"Available: {[name for name in dir(runners) if 'Runner' in name]}"
        )

    runner_class = getattr(runners, runner_class_name)

    if cfgs is None:
        cfgs = load_config_from_checkpoint(metadata)

    runner = runner_class.load_checkpoint(
        env=env,
        checkpoint_path=checkpoint_path,
        cfgs=cfgs,
        use_wandb=use_wandb,
    )

    return runner


def get_checkpoint_info(checkpoint_path: str) -> dict:
    """Get summary information about a checkpoint without loading it."""
    metadata = load_checkpoint_metadata(checkpoint_path)

    files = os.listdir(checkpoint_path)
    file_sizes = {
        f: os.path.getsize(os.path.join(checkpoint_path, f))
        for f in files
    }

    return {
        "runner_class": metadata.get("runner_class", "unknown"),
        "alg_class": metadata.get("alg_class", "unknown"),
        "iteration": metadata.get("iteration", 0),
        "total_timesteps": metadata.get("total_timesteps", 0),
        "total_time": metadata.get("total_time", 0),
        "files": file_sizes,
    }


def print_checkpoint_info(checkpoint_path: str) -> None:
    """Print formatted information about a checkpoint."""
    info = get_checkpoint_info(checkpoint_path)

    print(f"\n{'=' * 50}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'=' * 50}")
    print(f"  Runner class: {info['runner_class']}")
    print(f"  Algorithm class: {info['alg_class']}")
    print(f"  Iteration: {info['iteration']}")
    print(f"  Timesteps: {info['total_timesteps']:,}")
    print(f"  Training time: {info['total_time']:.1f}s")
    print(f"\n  Files:")
    for fname, size in info['files'].items():
        print(f"    {fname}: {size / 1024:.1f} KB")
    print(f"{'=' * 50}\n")
