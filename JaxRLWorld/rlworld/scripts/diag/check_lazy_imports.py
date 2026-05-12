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

Exit code is 0 when clean, 1 when an other-sim package leaked.
"""

from __future__ import annotations

import argparse
import sys

# Top-level package names for each backend.
_SIM_PKGS: dict[str, tuple[str, ...]] = {
    "genesis": ("genesis",),
    "newton": ("warp", "newton"),
    "mujoco": ("mjlab",),
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", choices=sorted(_SIM_PKGS), default="genesis")
    ap.add_argument("--preset", choices=sorted(_PRESETS), default="t1_getup")
    args = ap.parse_args()

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
