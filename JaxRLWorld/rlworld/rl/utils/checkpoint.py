import os
import pickle
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from rlworld.rl.configs import ConfigsForRun
    from rlworld.rl.runners import BaseRunner


def load_checkpoint_metadata(checkpoint_path: str) -> dict:
    """
    Load metadata from a checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Metadata dictionary
    """
    metadata_path = os.path.join(checkpoint_path, "metadata.pkl")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return metadata


def load_runner(
    env: "World",
    checkpoint_path: str,
    cfgs: Optional["ConfigsForRun"] = None,
    use_wandb: bool = False,
) -> "BaseRunner":
    """
    Load a runner from checkpoint, automatically detecting the runner class.

    Args:
        env: Environment instance
        checkpoint_path: Path to checkpoint directory
        cfgs: Optional config override. If None, uses config from checkpoint.
        use_wandb: Whether to use WandB logging

    Returns:
        Loaded runner instance

    Example:
        >>> from rlworld.rl.utils.checkpoint import load_runner
        >>> runner = load_runner(env, "outputs/models/checkpoint_100")
    """
    # Load metadata to get runner class
    metadata = load_checkpoint_metadata(checkpoint_path)

    runner_class_name = metadata.get("runner_class")
    if runner_class_name is None:
        raise ValueError(
            f"Checkpoint missing 'runner_class' in metadata. "
            f"This checkpoint may be from an older version."
        )

    # Dynamically import runner class
    from rlworld.rl import runners

    if not hasattr(runners, runner_class_name):
        raise ValueError(
            f"Unknown runner class: {runner_class_name}. "
            f"Available: {[name for name in dir(runners) if 'Runner' in name]}"
        )

    runner_class = getattr(runners, runner_class_name)

    # Use config from checkpoint if not provided
    if cfgs is None:
        from rlworld.rl.configs.config_classes import ConfigsForRun
        cfgs = ConfigsForRun.from_dict(metadata["config"])

    # Load using the appropriate runner's load_checkpoint method
    runner = runner_class.load_checkpoint(
        env=env,
        checkpoint_path=checkpoint_path,
        cfgs=cfgs,
        use_wandb=use_wandb,
    )

    return runner


def get_checkpoint_info(checkpoint_path: str) -> dict:
    """
    Get summary information about a checkpoint without loading it.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Dictionary with checkpoint info
    """
    metadata = load_checkpoint_metadata(checkpoint_path)

    # List files in checkpoint
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
    """
    Print formatted information about a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
    """
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