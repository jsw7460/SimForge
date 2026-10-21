"""Gate: evaluation mode means one thing everywhere it is applied.

Three places turned a training config into an evaluation one, each with
its own list of event modes to drop: the offline evaluator removed
``interval`` / ``interval_dr`` / ``reset_dr``, the in-training evaluation
kept ``reset_dr`` (so its ``Eval/*`` metrics carried the per-reset
randomization), and the eval-path determinism gate kept ``interval_dr``.
They now all call ``eval_mode.apply_eval_mode``.

1. on a config with every event mode and noisy observation groups, the
   default call removes exactly the three randomizing / disturbing modes,
   keeps ``startup`` and ``reset``, and turns corruption off on every
   group; applying it twice changes nothing more;
2. the two switches hold back their half (noise kept, events kept);
3. the three call sites go through the shared function and no site keeps
   a mode list of its own.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_eval_mode
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import jaxrlworld
from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.common_config_classes import EventConfig, ObservationGroupConfig
from jaxrlworld.rl.configs.eval_mode import EVAL_DISABLED_EVENT_MODES, apply_eval_mode
from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig
from jaxrlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun
from jaxrlworld.rl.utils.resolve import resolve_callable

ROOT = Path(jaxrlworld.__file__).parent
CALL_SITES = (
    "rl/evals/evaluator.py",
    "rl/runners/base_runner.py",
    "scripts/diag/gates/check_eval_determinism.py",
)
ALL_MODES = ("startup", "reset", "reset_dr", "interval", "interval_dr")


@dataclass
class _Events(EventConfig):
    place_robot = EventTermConfig(func=resolve_callable, mode="startup")
    reset_root = EventTermConfig(func=resolve_callable, mode="reset")
    mass_dr = EventTermConfig(func=resolve_callable, mode="reset_dr")
    push_robot = EventTermConfig(func=resolve_callable, mode="interval", interval_range_s=(2.0, 20.0))
    friction_dr = EventTermConfig(func=resolve_callable, mode="interval_dr", interval_dr_period_s=5.0)


def build() -> MujocoConfigsForRun:
    cfgs = MujocoConfigsForRun()
    cfgs.event = _Events()
    cfgs.observation.actor = ObservationGroupConfig(enable_corruption=True)
    cfgs.observation.critic = ObservationGroupConfig(enable_corruption=False)
    cfgs.observation.privileged = ObservationGroupConfig(enable_corruption=True)
    return cfgs


def modes(cfgs) -> dict[str, str]:
    return {name: term.mode for name, term in iter_terms(cfgs.event, EventTermConfig).items()}


def corruption(cfgs) -> dict[str, bool]:
    return {
        name: group.enable_corruption
        for name, group in vars(cfgs.observation).items()
        if isinstance(group, ObservationGroupConfig)
    }


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("EVAL MODE")
    print("=" * 78)
    chk(
        "the disabled modes are the three randomizing / disturbing ones",
        set(EVAL_DISABLED_EVENT_MODES) == {"interval", "interval_dr", "reset_dr"},
        str(EVAL_DISABLED_EVENT_MODES),
    )
    fresh = build()
    chk("fixture carries every event mode", set(modes(fresh).values()) == set(ALL_MODES), str(modes(fresh)))

    print("\n-- 1. default call --")
    cfgs = build()
    apply_eval_mode(cfgs)
    chk(
        "only startup and reset terms remain",
        modes(cfgs) == {"place_robot": "startup", "reset_root": "reset"},
        str(modes(cfgs)),
    )
    chk(
        "the removed terms are None on the instance (serialize as disabled)",
        all(getattr(cfgs.event, name) is None for name in ("mass_dr", "push_robot", "friction_dr")),
    )
    chk("corruption off on every group", set(corruption(cfgs).values()) == {False}, str(corruption(cfgs)))
    before = cfgs.recursive_to_dict()
    apply_eval_mode(cfgs)
    chk("a second application changes nothing", cfgs.recursive_to_dict() == before)

    print("\n-- 2. switches --")
    cfgs = build()
    apply_eval_mode(cfgs, disable_events=False)
    chk("disable_events=False keeps every event term", modes(cfgs) == modes(fresh), str(modes(cfgs)))
    chk("... and still turns noise off", set(corruption(cfgs).values()) == {False})
    cfgs = build()
    apply_eval_mode(cfgs, disable_noise=False)
    chk("disable_noise=False leaves corruption as configured", corruption(cfgs) == corruption(fresh))
    chk("... and still removes the three modes", set(modes(cfgs).values()) == {"startup", "reset"})
    cfgs = build()
    untouched = deepcopy(cfgs).recursive_to_dict()
    apply_eval_mode(cfgs, disable_noise=False, disable_events=False)
    chk("both off is a no-op", cfgs.recursive_to_dict() == untouched)

    print("\n-- 3. call sites --")
    mode_list = re.compile(r"term\.mode in \(")
    for rel in CALL_SITES:
        src = (ROOT / rel).read_text()
        chk(f"{rel} calls apply_eval_mode", "apply_eval_mode(" in src)
        chk(f"{rel} keeps no mode list of its own", mode_list.search(src) is None)

    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
