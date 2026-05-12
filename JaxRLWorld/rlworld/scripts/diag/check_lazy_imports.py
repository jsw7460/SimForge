"""Verify per-simulator lazy imports.

Runs the import path a real training script takes for one simulator
(import the preset config + the runner, then ``cfgs.build()``), then
inspects ``sys.modules`` to confirm the *other* simulators' heavyweight
packages were not dragged in.

Usage::

    # default: build a Genesis preset, assert neither newton/warp nor mjlab loaded
    python -m rlworld.scripts.diag.check_lazy_imports

    # other sims:
    python -m rlworld.scripts.diag.check_lazy_imports --sim newton
    python -m rlworld.scripts.diag.check_lazy_imports --sim mujoco

    # pick the preset (default: t1_getup):
    python -m rlworld.scripts.diag.check_lazy_imports --sim genesis --preset go2_flat

    # on a leak, print the stack trace of the first import of each sim pkg:
    python -m rlworld.scripts.diag.check_lazy_imports --sim genesis --trace

Exit code is 0 when clean, 1 when an other-sim package leaked.
"""

from __future__ import annotations

import argparse
import builtins
import sys
import traceback

# Top-level package names for each backend.  ``warp`` is listed for both
# Newton (it builds on Warp directly) and MuJoCo (``mjlab`` depends on
# ``mujoco_warp`` → ``warp``), so loading it is expected for either backend.
_SIM_PKGS: dict[str, tuple[str, ...]] = {
    "genesis": ("genesis",),
    "newton": ("warp", "newton"),
    "mujoco": ("mjlab", "warp"),
}

# preset key -> "rlworld.rl.configs.presets.<...>.base" module + config class
_PRESETS: dict[str, tuple[str, str]] = {
    "t1_getup": ("rlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
    "t1_tracking": ("rlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
    "go2_flat": ("rlworld.rl.configs.presets.go2_flat.base", "Go2FlatConfig"),
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("rlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
}


def _loaded_sim_pkgs() -> set[str]:
    """Top-level sim package names currently present in ``sys.modules``."""
    wanted = {p for pkgs in _SIM_PKGS.values() for p in pkgs}
    return {m.split(".")[0] for m in sys.modules if m.split(".")[0] in wanted}


def _install_import_tracer(watch: set[str]) -> None:
    """Print a stack trace the first time each name in ``watch`` is imported."""
    seen: set[str] = set()
    orig_import = builtins.__import__

    def traced(name, _globals=None, _locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if top in watch and top not in seen and top not in sys.modules:
            seen.add(top)
            print(f"\n>>> first import of {top!r} (via {name!r}) — stack:")
            traceback.print_stack()
            print()
        return orig_import(name, _globals, _locals, fromlist, level)

    builtins.__import__ = traced


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", choices=sorted(_SIM_PKGS), default="genesis")
    ap.add_argument("--preset", choices=sorted(_PRESETS), default="t1_getup")
    ap.add_argument(
        "--trace",
        action="store_true",
        help="print the stack trace of the first import of every sim package",
    )
    args = ap.parse_args()

    if args.trace:
        own = set(_SIM_PKGS[args.sim])
        watch = {p for pkgs in _SIM_PKGS.values() for p in pkgs} - own
        _install_import_tracer(watch)

    # 1) what a training script imports first
    import importlib

    from rlworld.rl.runners import BaseRunner  # noqa: F401  (this is the import under test)

    after_runner_import = _loaded_sim_pkgs()

    # 2) the preset config module + .build() (this is what picks the sim builder)
    mod_path, cls_name = _PRESETS[args.preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfg_cls(sim_type=args.sim).build()

    after_build = _loaded_sim_pkgs()

    own = set(_SIM_PKGS[args.sim])
    other = after_build - own

    print(f"=== lazy-import check: sim={args.sim!r} preset={args.preset!r} ===")
    print(f"  after `import rlworld.rl.runners`:        {sorted(after_runner_import) or 'NONE'}")
    print(f"  after `{cls_name}(sim_type={args.sim!r}).build()`: {sorted(after_build) or 'NONE'}")
    print(
        f"  expected own-sim packages ({args.sim}):   {sorted(own)}  -> "
        f"{'loaded ✓' if own <= after_build else 'MISSING ✗'}"
    )
    if other:
        print(f"  ✗ LEAKED other-sim packages: {sorted(other)}")
        return 1
    print("  ✓ no other-sim packages were imported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
