"""Gate: managers own their term objects; the config's terms stay declarative.

Named terms are class attributes of their config, shared by every instance
of the class and by every ``deepcopy`` of an instance (the in-training eval
config). A manager resolves selectors into ``term.params`` and lets a
curriculum move weights, so it must work on its own copies.

Part A needs no simulator: a stub env stands in for selector resolution.

1. two managers built from one config hold distinct term objects, and
   neither is the config's;
2. the config's term still carries the declarative ``SceneEntitySelector``
   after both managers resolved theirs;
3. a curriculum-style write on one manager's term (weight, params) reaches
   neither the other manager nor the config;
4. the manager's resolved params are what its stateful/function terms see.

Part B (``--preset module:Class --sim-type ...``, needs the simulator)
builds the preset's env, then the deep-copied eval config's env the way
the runner does, and checks the same properties on the real managers.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_term_ownership
"""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
from dataclasses import dataclass

from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.common_config_classes import RewardConfig
from jaxrlworld.rl.configs.rewards.reward_term_config import RewardTermConfig
from jaxrlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from jaxrlworld.rl.envs.managers.base import BaseManager

failures: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


class _Resolved:
    """Stands in for ResolvedEntity: what a stub env hands back per resolve call."""

    count = 0

    def __init__(self, selector: SceneEntitySelector):
        _Resolved.count += 1
        self.selector = selector
        self.serial = _Resolved.count


class _StubEnv:
    device = "cpu"

    def resolve_selector(self, selector: SceneEntitySelector) -> _Resolved:
        return _Resolved(selector)


def _term_fn(env, asset_cfg=SceneEntitySelector(name="robot", joint_names=(".*_knee",))):
    return None


@dataclass
class _Rewards(RewardConfig):
    feet = RewardTermConfig(
        func=_term_fn, weight=1.5, params={"asset_cfg": SceneEntitySelector(body_names=(".*_foot",))}
    )
    knees = RewardTermConfig(func=_term_fn, weight=-0.2)  # selector comes from the function default


class _StubManager(BaseManager):
    """The term-ownership part of the real managers' __init__."""

    def __init__(self, env, config):
        super().__init__(env=env)
        self.terms = self._own_terms(iter_terms(config, RewardTermConfig))
        for term in self.terms.values():
            self._resolve_term_selectors(term.resolved_func, term.params)


def part_a() -> None:
    print("=== A1. distinct term objects per manager ===")
    cfg = _Rewards()
    m1 = _StubManager(_StubEnv(), cfg)
    m2 = _StubManager(_StubEnv(), copy.deepcopy(cfg))
    chk("manager term is not the config's", m1.terms["feet"] is not cfg.feet and m1.terms["feet"] is not _Rewards.feet)
    chk("two managers hold different term objects", m1.terms["feet"] is not m2.terms["feet"])
    chk(
        "params dicts are distinct",
        m1.terms["feet"].params is not cfg.feet.params and m1.terms["feet"].params is not m2.terms["feet"].params,
    )

    print("\n=== A2. the config keeps its declarative selector ===")
    chk("config params still hold SceneEntitySelector", isinstance(cfg.feet.params["asset_cfg"], SceneEntitySelector))
    chk(
        "config term without asset_cfg stays without it",
        "asset_cfg" not in cfg.knees.params,
        f"{sorted(cfg.knees.params)}",
    )
    chk("manager params hold the resolved entity", isinstance(m1.terms["feet"].params["asset_cfg"], _Resolved))
    chk(
        "function-default selector injected into the manager's copy only",
        isinstance(m1.terms["knees"].params["asset_cfg"], _Resolved),
    )
    chk(
        "each manager resolved on its own",
        m1.terms["feet"].params["asset_cfg"].serial != m2.terms["feet"].params["asset_cfg"].serial,
    )

    print("\n=== A3. curriculum-style writes stay local ===")
    m1.terms["feet"].weight = 9.0
    m1.terms["feet"].params["std"] = 0.01
    chk("other manager weight unchanged", m2.terms["feet"].weight == 1.5, f"{m2.terms['feet'].weight}")
    chk("config weight unchanged", cfg.feet.weight == 1.5 and _Rewards.feet.weight == 1.5)
    chk("other manager params unchanged", "std" not in m2.terms["feet"].params and "std" not in cfg.feet.params)

    print("\n=== A4. fresh manager from the same class resolves afresh ===")
    m3 = _StubManager(_StubEnv(), _Rewards())
    chk(
        "new manager sees the declarative selector, resolves it",
        isinstance(m3.terms["feet"].params["asset_cfg"], _Resolved) and m3.terms["feet"].weight == 1.5,
    )


def part_b(preset: str, sim_type: str) -> None:
    print(f"\n=== B. real env: {preset} ({sim_type}) ===")
    from jaxrlworld.rl.runners.base_runner import BaseRunner

    module_name, class_name = preset.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    cfgs = cls(sim_type=sim_type).build()
    cfgs.env.num_envs = 4
    cfgs.visualization.show_viewer = False
    env = BaseRunner._create_env_from_config(cfgs)
    name, cfg_term = next(iter(iter_terms(cfgs.reward, RewardTermConfig).items()))
    live = env.reward_manager.get_term_cfg(name)
    chk("reward manager term is not the config's", live is not cfg_term)
    selectors_left = all(
        not type(v).__name__ == "ResolvedEntity"
        for t in iter_terms(cfgs.reward, RewardTermConfig).values()
        for v in t.params.values()
    )
    chk("config reward params carry no ResolvedEntity", selectors_left)
    obs_ok = all(
        not type(v).__name__ == "ResolvedEntity"
        for group in vars(type(cfgs.observation)).values()
        if hasattr(group, "__dict__")
        for t in iter_terms(group, object).values()
        if isinstance(t, RewardTermConfig) or hasattr(t, "params")
        for v in getattr(t, "params", {}).values()
    )
    chk("config observation params carry no ResolvedEntity", obs_ok)
    eval_cfgs = copy.deepcopy(cfgs)
    eval_env = BaseRunner._create_env_from_config(eval_cfgs)
    chk("eval env term is its own object", eval_env.reward_manager.get_term_cfg(name) is not live)
    live.weight = live.weight + 1.0
    chk(
        "weight write on train env does not reach eval env",
        eval_env.reward_manager.get_term_cfg(name).weight != live.weight,
    )


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
