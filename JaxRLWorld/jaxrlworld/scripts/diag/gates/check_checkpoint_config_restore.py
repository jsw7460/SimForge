"""Gate: a checkpoint loads the config it was trained with.

Part A needs no simulator. A ``ConfigsForRun`` with class-level reward
terms stands in for a built preset; overrides are applied the way a
training run applies them (command-line style through
``apply_overrides``, programmatic attribute writes, an ``_type`` swap of
a nested NN config), the config is written through the checkpoint's YAML
path and read back, a fresh "preset" is rebuilt, and
``restore_saved_config`` must turn it into the saved config:

1. the restored config serializes back to the saved dict exactly;
2. an overridden reward term is still a term object, with the overridden
   weight and params (the override used to replace it with a dict);
3. a saved field the preset no longer has raises instead of vanishing;
4. a term-level command-line override keeps the term.

Part B runs the real path on a preset and needs its simulator package::

    python -m jaxrlworld.scripts.diag.gates.check_checkpoint_config_restore \\
        --preset jaxrlworld.rl.configs.presets.go2.base:Go2FlatConfig --sim-type mujoco

It builds the preset, applies overrides, saves ``config.yaml`` as the
runner does and loads it with ``load_config_from_checkpoint``.

Run (part A only)::

    python -m jaxrlworld.scripts.diag.gates.check_checkpoint_config_restore
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
import sys
import tempfile
from dataclasses import dataclass

from jaxrlworld.rl.configs.algorithms import get_algorithm_config_class
from jaxrlworld.rl.configs.base_config import iter_terms, parse_override_args
from jaxrlworld.rl.configs.common_config_classes import RewardConfig
from jaxrlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun
from jaxrlworld.rl.configs.rewards.reward_term_config import RewardTermConfig
from jaxrlworld.rl.utils.checkpoint import load_config_from_checkpoint, restore_saved_config
from jaxrlworld.rl.utils.resolve import resolve_callable
from jaxrlworld.rl.utils.yaml_io import dump_yaml, load_yaml

failures: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


@dataclass
class _Rewards(RewardConfig):
    track = RewardTermConfig(func=resolve_callable, weight=2.0, params={"std": 0.5, "penalize_z": True})
    upright = RewardTermConfig(func=resolve_callable, weight=-1.0)


def synthetic_build() -> MujocoConfigsForRun:
    """What a preset's ``build()`` would return: fresh objects every call."""
    cfgs = MujocoConfigsForRun(algorithm=get_algorithm_config_class("PPO")())
    cfgs.reward = _Rewards()
    return cfgs


def yaml_round_trip(cfgs) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.yaml")
        dump_yaml(path, cfgs.recursive_to_dict())
        return load_yaml(path)


def part_a() -> None:
    print("=== A1. overrides survive save -> rebuild -> restore ===")
    trained = synthetic_build()
    # Command-line style (what with_cli_overrides applies), including a term leaf.
    trained.apply_overrides(
        algorithm={"gamma": 0.93, "num_learning_epochs": 7},
        env={"num_envs": 12},
        reward={"track": {"weight": 3.5, "params": {"std": 0.9}}, "upright": None},
        nn={"policy": {"actor": {"_type": "SpaceTimeTransformerActorCfg"}}},
    )
    # Programmatic writes after build().
    trained.runner.save_interval = 123
    trained.reward.reward_mode = "exponential"
    saved = yaml_round_trip(trained)

    rebuilt = synthetic_build()
    chk("rebuilt preset differs before restore", rebuilt.recursive_to_dict() != saved)
    restore_saved_config(rebuilt, saved)
    chk("restored config serializes to the saved dict", rebuilt.recursive_to_dict() == saved)
    chk("algorithm.gamma", rebuilt.algorithm.gamma == 0.93, f"{rebuilt.algorithm.gamma}")
    chk("env.num_envs", rebuilt.env.num_envs == 12, f"{rebuilt.env.num_envs}")
    chk("runner.save_interval (programmatic)", rebuilt.runner.save_interval == 123)
    chk("reward.reward_mode (programmatic)", rebuilt.reward.reward_mode == "exponential")
    chk(
        "nn actor _type swap restored",
        type(rebuilt.nn.policy.actor).__name__ == "SpaceTimeTransformerActorCfg",
        type(rebuilt.nn.policy.actor).__name__,
    )
    terms = iter_terms(rebuilt.reward, RewardTermConfig)
    chk(
        "overridden term is still a RewardTermConfig",
        isinstance(rebuilt.reward.track, RewardTermConfig),
        type(rebuilt.reward.track).__name__,
    )
    chk(
        "term weight / params restored",
        terms["track"].weight == 3.5 and terms["track"].params == {"std": 0.9, "penalize_z": True},
        f"{terms['track']}",
    )
    chk("term disabled at train time stays disabled", "upright" not in terms, f"terms {sorted(terms)}")
    chk("term func kept callable-resolvable", callable(terms["track"].resolved_func))

    print("\n=== A2. restore is a no-op on an unchanged preset ===")
    plain = synthetic_build()
    restore_saved_config(plain, yaml_round_trip(synthetic_build()))
    chk("no drift -> unchanged", plain.recursive_to_dict() == synthetic_build().recursive_to_dict())

    print("\n=== A3. a saved field the preset no longer has raises ===")
    stale = copy.deepcopy(saved)
    stale["algorithm"]["removed_since_training"] = 1
    try:
        restore_saved_config(synthetic_build(), stale)
        chk("unknown saved field raises", False, "no exception")
    except ValueError as e:
        chk("unknown saved field raises", "removed_since_training" in str(e), str(e)[:90])

    print("\n=== A4. term-level command-line override keeps the term ===")
    cfgs = synthetic_build()
    argv_backup = sys.argv
    sys.argv = ["prog", "reward.track.weight=4.25", "reward.track.params.std=0.1", "algorithm.gamma=0.5"]
    try:
        overrides = parse_override_args()
    finally:
        sys.argv = argv_backup
    cfgs.apply_overrides(**overrides)
    terms = iter_terms(cfgs.reward, RewardTermConfig)
    chk(
        "reward.track.weight= keeps the term",
        "track" in terms and isinstance(cfgs.reward.track, RewardTermConfig),
        type(cfgs.reward.track).__name__,
    )
    chk(
        "weight and nested params applied",
        terms.get("track") is not None and terms["track"].weight == 4.25 and terms["track"].params["std"] == 0.1,
    )
    chk("sibling term untouched", terms["upright"].weight == -1.0)
    # Terms are class attributes, so the write above lands on the object every
    # instance of _Rewards shares; presets avoid this by defining their term
    # classes inside build(), which this synthetic class does not.
    print(f"  (note) class-level term after the override: _Rewards.track.weight = {_Rewards.track.weight}")


def part_b(preset: str, sim_type: str) -> None:
    print(f"\n=== B. real preset round trip: {preset} ({sim_type}) ===")
    module_name, class_name = preset.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    trained = cls(sim_type=sim_type).build()
    trained.apply_overrides(algorithm={"gamma": 0.931}, env={"num_envs": 24})
    trained.runner.save_interval = 777
    first_term = next(iter(iter_terms(trained.reward, RewardTermConfig)))
    getattr(trained.reward, first_term).weight = 12.5
    saved = yaml_round_trip(trained)
    restored = load_config_from_checkpoint({"config": saved})
    chk("load_config_from_checkpoint returns the saved config", restored.recursive_to_dict() == saved)
    chk("algorithm.gamma", restored.algorithm.gamma == 0.931)
    chk("env.num_envs", restored.env.num_envs == 24)
    chk("runner.save_interval", restored.runner.save_interval == 777)
    chk(f"reward.{first_term}.weight", getattr(restored.reward, first_term).weight == 12.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None, help="module:Class of a preset for part B")
    ap.add_argument("--sim-type", default="mujoco")
    args = ap.parse_args()
    part_a()
    if args.preset is not None:
        part_b(args.preset, args.sim_type)
    else:
        print("\n(part B not run: pass --preset module:Class on a machine with the simulator installed)")
    print(f"\n=== RESULT: {'ALL OK' if not failures else f'{len(failures)} FAILED: {failures}'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
