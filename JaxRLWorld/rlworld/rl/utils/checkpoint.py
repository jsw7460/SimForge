import os
from typing import TYPE_CHECKING, Optional

from rlworld.rl.utils.yaml_io import load_yaml

if TYPE_CHECKING:
    from rlworld.rl.configs import ConfigsForRun
    from rlworld.rl.envs import World
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

    Resolution order (most specific → least specific):

    1. **Class + kwargs** (preferred, set automatically by
       ``Go2FlatConfig.build()`` / ``G1FlatConfig.build()``): if the
       checkpoint has both ``preset_class_name`` and ``preset_kwargs``,
       reconstruct via ``getattr(module, class_name)(**kwargs).build()``.

    2. **Module-level get_config()** (legacy entry-point convention):
       fall through to ``preset_mod.get_config()`` for older checkpoints
       whose ``preset_module`` already pointed at a wrapper module like
       ``presets.go2_flat.mlp``.

    3. **Convention-based fallback**: locate a buildable ``*Config``
       class inside ``preset_module`` (e.g. ``Go2FlatConfig`` inside
       ``presets.go2_flat.base``) and instantiate it with
       ``sim_type=<config.sim_type>`` from the checkpoint metadata. This
       covers the gap created in Phase A, when the unified
       ``Go2FlatConfig`` started writing ``preset_module = base`` (the
       dataclass module) but no ``get_config()`` lived there. Existing
       Phase-A-era checkpoints reload via this path.

    Raises ``AttributeError`` only when none of the three paths can
    rebuild the config.
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

    # Path 1: explicit class + kwargs
    preset_class_name = config_dict.get("preset_class_name")
    if preset_class_name is not None:
        cls = getattr(preset_mod, preset_class_name, None)
        if cls is None:
            raise AttributeError(
                f"Checkpoint references preset class {preset_class_name!r} "
                f"in module {preset_module_path!r}, but the class is no "
                f"longer defined there."
            )
        preset_kwargs = config_dict.get("preset_kwargs") or {}
        return cls(**preset_kwargs).build()

    # Path 2: legacy module-level get_config()
    if hasattr(preset_mod, "get_config"):
        return preset_mod.get_config()

    # Path 3: convention-based fallback for Phase-A-era checkpoints.
    # Locate a buildable *Config class inside the module.
    sim_type = config_dict.get("sim_type")
    candidate_cls = None
    for attr_name in dir(preset_mod):
        if attr_name.startswith("_") or not attr_name.endswith("Config"):
            continue
        obj = getattr(preset_mod, attr_name)
        if isinstance(obj, type) and hasattr(obj, "build"):
            candidate_cls = obj
            break

    if candidate_cls is None:
        raise AttributeError(
            f"Cannot reconstruct config from checkpoint: module "
            f"{preset_module_path!r} has neither preset_class_name in "
            f"the checkpoint nor a get_config() function nor a buildable "
            f"*Config class."
        )

    try:
        if sim_type is not None:
            return candidate_cls(sim_type=sim_type).build()
        return candidate_cls().build()
    except TypeError as e:
        raise AttributeError(
            f"Found candidate class {candidate_cls.__name__} in "
            f"{preset_module_path!r} but could not instantiate it with "
            f"sim_type={sim_type!r}: {e}"
        ) from e


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
        raise ValueError("Checkpoint missing 'runner_class' in train_state.yaml.")

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
    file_sizes = {f: os.path.getsize(os.path.join(checkpoint_path, f)) for f in files}

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
    print("\n  Files:")
    for fname, size in info["files"].items():
        print(f"    {fname}: {size / 1024:.1f} KB")
    print(f"{'=' * 50}\n")
