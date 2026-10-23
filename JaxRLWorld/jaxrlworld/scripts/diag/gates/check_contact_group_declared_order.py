"""Gate: every contact group's columns are in the order its config declares.

A contact group's column ``i`` is ``tracked_names[i]``. The backends
resolve the same ``ContactSensorCfg.primary.pattern`` in different
native orders — mjlab and Newton list the matches in model order, Genesis
in pattern order — so until the manager put the columns in DECLARED order
a tuple ``("FR_foot", "FL_foot", "RR_foot", "RL_foot")`` gave
``FL, FR, RL, RR`` columns on two backends and ``FR, FL, RR, RL`` on the
third. Anything that reads those columns per element (per-foot
observations, a reward multiplying them by another per-foot tensor)
then meant a different foot per simulator.

This gate builds every preset on one backend (one process per cell, like
``check_all_presets``) and records, per contact group, the native order
the backend produced, the declared patterns and the order the manager
exposes. Per cell it checks that

1. the exposed order is the declared order of the native names
   (``BaseContactManager.declared_order``), i.e. no backend path bypasses
   the permutation;
2. a group declared as a tuple of literal names exposes exactly that
   tuple — column ``i`` IS declared name ``i``;
3. ``is_contact`` has one column per tracked name.

``--compare`` then takes the per-backend dumps and checks, per group,
the cross-backend ORDER:

- the same literal tuple on every backend → identical columns;
- a literal tuple spelled per backend (geom names on mjlab, body names
  elsewhere, as go2's feet) → column ``i`` names the same part on every
  backend, by the ``_``-token rule ``_feet_contact_order`` uses;
- a regex on some backend → the names every backend exposes appear in
  the same relative order. The backends legitimately expose different
  sets (mjlab's single subtree column for ``self_collision``, frame
  bodies Newton's importer drops, mjlab's mocap wrapper body), so the
  extras are printed as ``[INFO]``, not failed.

Run (one backend per process; the parent spawns one child per cell)::

    jaxpy -m jaxrlworld.scripts.diag.gates.check_contact_group_declared_order --sims mujoco --out parity_out/contact_order
    jaxpy -m jaxrlworld.scripts.diag.gates.check_contact_group_declared_order --sims newton --out parity_out/contact_order
    jaxpy -m jaxrlworld.scripts.diag.gates.check_contact_group_declared_order --sims genesis --out parity_out/contact_order
    jaxpy -m jaxrlworld.scripts.diag.gates.check_contact_group_declared_order --compare parity_out/contact_order
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager
from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.scripts.diag.gates.check_all_presets import _PUBLIC, _SIMS, _load

_REGEX_CHARS = re.compile(r"[.*+?\[\]{}()\\|^$]")


def is_literal(pattern: str) -> bool:
    return _REGEX_CHARS.search(pattern) is None


def run_cell(label: str, loader: str, sim: str, num_envs: int) -> dict:
    cfgs = _load(loader, sim, num_envs)
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    env = runner.env
    env.reset()
    manager = env.contact_manager
    groups = {}
    failures = []
    for name in manager.group_names():
        group = manager._get_group(name)
        if group.declared_patterns is None:
            failures.append(f"{name}: registered without declared patterns")
            continue
        declared = BaseContactManager.declared_order(group.declared_patterns, group.native_names)
        if group.tracked_names != declared:
            failures.append(f"{name}: exposed {group.tracked_names} != declared order {declared}")
        literal = all(is_literal(p) for p in group.declared_patterns)
        if literal and group.tracked_names != list(group.declared_patterns):
            failures.append(
                f"{name}: literal declaration {list(group.declared_patterns)} exposed as {group.tracked_names}"
            )
        contact = manager.is_contact(name)
        if tuple(contact.shape) != (env.num_envs, len(group.tracked_names)):
            failures.append(f"{name}: is_contact shape {tuple(contact.shape)} for {len(group.tracked_names)} names")
        groups[name] = {
            "native": list(group.native_names),
            "declared_patterns": list(group.declared_patterns),
            "tracked": list(group.tracked_names),
            "literal": literal,
            "permuted": group.native_perm is not None,
        }
    return {"label": label, "sim": sim, "groups": groups, "failures": failures}


def _print_cell(result: dict) -> None:
    for name, g in result["groups"].items():
        tag = "literal" if g["literal"] else "regex"
        moved = " (permuted from native)" if g["permuted"] else ""
        print(f"    {name} [{tag}]{moved}")
        print(f"      declared : {g['declared_patterns']}")
        print(f"      native   : {g['native']}")
        print(f"      exposed  : {g['tracked']}")
    for f in result["failures"]:
        print(f"    FAIL {f}")


def _tokens(name: str) -> set[str]:
    return set(name.split("_"))


def _token_compatible(a: str, b: str) -> bool:
    """Same rule as ``_feet_contact_order``: one name's ``_`` tokens contain the other's."""
    ta, tb = _tokens(a), _tokens(b)
    return ta <= tb or tb <= ta


def _common_order(cells: list[dict], name: str) -> tuple[list[str], dict[str, list[str]]]:
    """Names every backend exposes, in the first backend's column order, plus the per-backend extras."""
    tracked = {c["sim"]: c["groups"][name]["tracked"] for c in cells}
    common = set.intersection(*(set(t) for t in tracked.values()))
    ordered = [n for n in tracked[cells[0]["sim"]] if n in common]
    extras = {sim: [n for n in t if n not in common] for sim, t in tracked.items()}
    return ordered, extras


def compare(out: Path) -> int:
    results = [json.loads(p.read_text()) for p in sorted(out.glob("*__*.json"))]
    by_label: dict[str, list[dict]] = {}
    for r in results:
        by_label.setdefault(r["label"], []).append(r)
    failures = 0
    print("=" * 78)
    print("CONTACT GROUP DECLARED ORDER — cross-backend")
    print("=" * 78)
    for label, cells in sorted(by_label.items()):
        sims = [c["sim"] for c in cells]
        print(f"\n{label}  ({', '.join(sims)})")
        for c in cells:
            failures += len(c["failures"])
            for f in c["failures"]:
                print(f"  [FAIL] {c['sim']}: {f}")
        group_names = sorted({n for c in cells for n in c["groups"]})
        for name in group_names:
            present = [c for c in cells if name in c["groups"]]
            if len(present) < 2:
                continue
            groups = {c["sim"]: c["groups"][name] for c in present}
            declarations = {tuple(g["declared_patterns"]) for g in groups.values()}
            all_literal = all(g["literal"] for g in groups.values())
            if all_literal and len(declarations) == 1:
                # Same literal tuple everywhere: column i must be the same name on every backend.
                first = present[0]["groups"][name]["tracked"]
                same = all(g["tracked"] == first for g in groups.values())
                failures += 0 if same else 1
                print(f"  [{'OK' if same else 'FAIL'}] {name}: literal columns identical on {list(groups)}")
                if not same:
                    for sim, g in groups.items():
                        print(f"         {sim:8s} {g['tracked']}")
            elif all_literal:
                # Literal on every backend but spelled per backend (geom names on one, body
                # names on another): column i must name the same part on every backend.
                lengths = {len(g["tracked"]) for g in groups.values()}
                same = len(lengths) == 1 and all(
                    _token_compatible(a, b)
                    for g in groups.values()
                    for a, b in zip(present[0]["groups"][name]["tracked"], g["tracked"])
                )
                failures += 0 if same else 1
                print(
                    f"  [{'OK' if same else 'FAIL'}] {name}: literal columns token-compatible per position on {list(groups)}"
                )
                for sim, g in groups.items():
                    print(f"         {sim:8s} {g['tracked']}")
            else:
                # A regex somewhere: the backends may expose different bodies (a subtree sensor,
                # frame bodies one importer drops, a mocap wrapper). The ORDER of the names they
                # share is what this gate asserts; the extras are printed for reading.
                ordered, extras = _common_order(present, name)
                same = all([n for n in g["tracked"] if n in ordered] == ordered for g in groups.values())
                failures += 0 if same else 1
                counts = {sim: len(g["tracked"]) for sim, g in groups.items()}
                print(
                    f"  [{'OK' if same else 'FAIL'}] {name}: {len(ordered)} shared columns in the same order on {list(groups)}; columns {counts}"
                )
                for sim, g in groups.items():
                    print(f"         {sim:8s} {g['tracked']}")
                for sim, extra in extras.items():
                    if extra:
                        print(f"         [INFO] only on {sim}: {extra}")
    print("\n" + "=" * 78)
    print("=== RESULT: ALL OK ===" if failures == 0 else f"=== RESULT: {failures} FAILED ===")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", default=",".join(_SIMS))
    ap.add_argument("--only", default=None, help="substring filter on the preset label")
    ap.add_argument("--out", default="parity_out/contact_order", help="directory for the per-cell json files")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--compare", default=None, help="an --out directory: compare its dumps across backends")
    ap.add_argument("--cell", default=None, help="internal: 'label|loader|sim'")
    ap.add_argument("--result-json", default=None, help="internal")
    args = ap.parse_args()

    if args.compare is not None:
        return compare(Path(args.compare))

    if args.cell is not None:
        label, loader, sim = args.cell.split("|")
        result = run_cell(label, loader, sim, args.num_envs)
        Path(args.result_json).write_text(json.dumps(result))
        return 0

    sims = tuple(args.sims.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for label, loader, spec_sims in _PUBLIC:
        if args.only and args.only not in label:
            continue
        for sim in spec_sims:
            if sim not in sims:
                continue
            result_path = out / f"{label}__{sim}.json"
            log_path = out / f"{label}__{sim}.log"
            if result_path.exists():
                result_path.unlink()
            print(f"{label}:{sim}")
            cmd = [
                sys.executable,
                "-m",
                "jaxrlworld.scripts.diag.gates.check_contact_group_declared_order",
                "--cell",
                f"{label}|{loader}|{sim}",
                "--result-json",
                str(result_path),
                "--num-envs",
                str(args.num_envs),
            ]
            with log_path.open("w") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ)).returncode
            if rc != 0 or not result_path.exists():
                failures += 1
                print(f"    [FAIL] build/run failed (rc={rc}), see {log_path}")
                continue
            result = json.loads(result_path.read_text())
            failures += len(result["failures"])
            _print_cell(result)
    print("=" * 78)
    print("=== RESULT: ALL OK ===" if failures == 0 else f"=== RESULT: {failures} FAILED ===")
    print(f"dumps in {out}; run --compare {out} once every backend has been dumped there")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
