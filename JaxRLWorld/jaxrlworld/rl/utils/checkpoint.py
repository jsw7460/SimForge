import importlib
import os
from typing import TYPE_CHECKING, Optional

from jaxrlworld.rl.configs.base_config import diff_config_dicts, flatten_leaf_paths, missing_from_saved
from jaxrlworld.rl.utils.yaml_io import load_yaml

if TYPE_CHECKING:
    from jaxrlworld.rl.configs import ConfigsForRun
    from jaxrlworld.rl.envs import World
    from jaxrlworld.rl.runners import BaseRunner


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
    """Reconstruct the config a checkpoint was trained with.

    The preset named in ``config.yaml`` is rebuilt (see
    :func:`_rebuild_preset`), then every value the saved config differs
    in from that rebuild, whether a command-line or programmatic override
    applied after ``build()`` or a preset default that has since changed,
    is written back, so the result is the saved config, with named terms
    intact. A saved field the current preset no longer has, or a value
    it holds immutable, raises rather than being dropped.
    """
    config_dict = metadata["config"]
    cfgs = _rebuild_preset(config_dict)
    restore_saved_config(cfgs, config_dict)
    return cfgs


def restore_saved_config(cfgs: "ConfigsForRun", saved: dict) -> None:
    """Write the leaves of ``saved`` that differ from ``cfgs`` back onto it.

    ``saved`` is a ``recursive_to_dict()`` output (after its YAML round
    trip). The differences are applied through ``apply_overrides`` and
    the result is checked to serialize back to ``saved``. Config the
    rebuilt preset has and the saved run did not (a term or field added
    since training) raises: the saved run never ran with it, and a restore
    that let it through would evaluate a different MDP without a word.
    """
    diff = _saved_diff(saved, cfgs.recursive_to_dict())
    if not diff:
        return
    cfgs.apply_overrides(**diff)
    residual = diff_config_dicts(saved, cfgs.recursive_to_dict())
    if residual:
        raise ValueError(
            "Saved config could not be restored onto the rebuilt preset; still differing after the "
            f"override pass: {flatten_leaf_paths(residual)}"
        )
    paths = flatten_leaf_paths(diff)
    shown = ", ".join(paths[:20]) + (f", ... ({len(paths) - 20} more)" if len(paths) > 20 else "")
    print(f"[checkpoint] restored {len(paths)} config value(s) that differ from the rebuilt preset: {shown}")


def _saved_diff(saved: dict, current: dict) -> dict:
    """The nested overrides that turn ``current`` into ``saved``.

    Both are ``recursive_to_dict()`` outputs; ``current`` is a fresh build
    of the saved run's preset. Config ``current`` has and ``saved`` does
    not (added since training) raises, and so does a difference in the
    top-level scalars that identify the preset itself.
    """
    added = missing_from_saved(saved, current)
    if added:
        raise ValueError(
            f"The rebuilt preset has {len(added)} config value(s) the saved run did not: {added[:20]}"
            f"{' ...' if len(added) > 20 else ''}. They were added after this checkpoint was trained; disable "
            "them for the evaluation or re-train."
        )
    diff = diff_config_dicts(saved, current)
    scalar_drift = {k: v for k, v in diff.items() if not isinstance(v, dict)}
    if scalar_drift:
        raise ValueError(
            f"Saved config differs from the rebuilt preset in top-level fields {sorted(scalar_drift)}; "
            "these identify the preset itself and cannot be overridden"
        )
    return diff


# Sections of a saved config that cross-sim evaluation does not replay.
# The scene, observation and action sections are spelled in the training
# simulator's terms, so the target preset's own are kept (the evaluator
# checks the observation layout and the action mapping against the saved
# ones) and a saved difference there is reported as not carried over.
# Algorithm and nn are copied whole from the checkpoint by the evaluator
# and the visualization is the evaluator's own, so differences there are
# neither replayed nor reported.
SIM_BOUND_SECTIONS: frozenset[str] = frozenset({"scene", "observation", "action"})
EVALUATOR_OWNED_SECTIONS: frozenset[str] = frozenset({"algorithm", "nn", "visualization"})


def training_overrides(saved: dict) -> dict:
    """The values a saved run differs in from a fresh build of its preset.

    The preset named in ``saved`` is rebuilt for the simulator it was
    trained on, and the result is the nested override dict (the shape
    ``apply_overrides`` takes) of every leaf the saved config changed:
    command-line and programmatic overrides applied after ``build()``, and
    preset defaults that have moved since.
    """
    return _saved_diff(saved, _rebuild_preset(saved).recursive_to_dict())


def replay_training_overrides(cfgs: "ConfigsForRun", saved: dict) -> tuple[list[str], list[str]]:
    """Apply a saved run's simulator-independent overrides to ``cfgs``.

    ``cfgs`` is a config built for a simulator other than the one the run
    trained on, so the saved config cannot be restored onto it whole:
    the sections in :data:`SIM_BOUND_SECTIONS` and
    :data:`EVALUATOR_OWNED_SECTIONS` are left as ``cfgs`` has them and
    every other difference (reward weights, command ranges, event,
    termination, curriculum and env settings, runner options) is written
    on through ``apply_overrides``, which raises on a path ``cfgs`` does
    not have. Returns the dotted paths applied and the dotted paths of
    the simulator-bound differences left alone, for the caller to report.
    """
    diff = training_overrides(saved)
    left_alone = SIM_BOUND_SECTIONS | EVALUATOR_OWNED_SECTIONS
    replay = {section: values for section, values in diff.items() if section not in left_alone}
    skipped = {section: values for section, values in diff.items() if section in SIM_BOUND_SECTIONS}
    if replay:
        cfgs.apply_overrides(**replay)
    return flatten_leaf_paths(replay), flatten_leaf_paths(skipped)


def _rebuild_preset(config_dict: dict) -> "ConfigsForRun":
    """Re-run the preset named in a saved config.

    ``build()`` stamps ``preset_module``, ``preset_class_name`` and
    ``preset_kwargs`` on the config; the preset is rebuilt as
    ``getattr(module, class_name)(**kwargs).build()``.
    """
    preset_module_path = config_dict["preset_module"]
    preset_class_name = config_dict["preset_class_name"]
    if preset_module_path is None or preset_class_name is None:
        raise ValueError(
            f"Saved config has preset_module={preset_module_path!r}, preset_class_name={preset_class_name!r}; "
            "only a config built by a preset class's build() can be rebuilt."
        )

    preset_mod = importlib.import_module(preset_module_path)
    cls = getattr(preset_mod, preset_class_name, None)  # dynamic-attr: class named by checkpoint
    if cls is None:
        raise AttributeError(
            f"Checkpoint references preset class {preset_class_name!r} "
            f"in module {preset_module_path!r}, but the class is no "
            f"longer defined there."
        )
    return cls(**config_dict["preset_kwargs"]).build()


def runner_class_from_metadata(metadata: dict) -> type:
    """The runner class a checkpoint was trained with."""
    module = importlib.import_module(metadata["runner_module"])
    return getattr(module, metadata["runner_class"])  # dynamic-attr: runner named by checkpoint


def load_runner(
    env: "World",
    checkpoint_path: str,
    cfgs: Optional["ConfigsForRun"] = None,
    use_wandb: bool = False,
) -> "BaseRunner":
    """Load a runner from checkpoint, automatically detecting the runner class."""
    metadata = load_checkpoint_metadata(checkpoint_path)
    runner_class = runner_class_from_metadata(metadata)

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
        "runner_class": metadata["runner_class"],
        "alg_class": metadata["alg_class"],
        "iteration": metadata["iteration"],
        "total_timesteps": metadata["total_timesteps"],
        "total_time": metadata["total_time"],
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
