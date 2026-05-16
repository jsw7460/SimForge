"""Cross-sim foot-collision friction inspection.

Builds each requested sim with the same preset (defaults to g1_29dof),
then reads back the actual sliding/torsional/rolling friction baked into
the live simulation model for every collision geom. Prints per-geom and
per-sim friction so we can see whether mjlab's ``FULL_COLLISION``
override (``friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)}``)
landed where intended and what Genesis / Newton picked up from the same
MJCF (whose ``<default class="foot_capsule">`` does *not* author a
friction attribute, so each sim falls back to its own default).

The expected result if the MJCF is the single source of truth:

  * mjlab: foot geoms ≈ 0.6 (forced by ``FULL_COLLISION``), non-foot
    collisions defaulted to whatever mjlab/MuJoCo chose.
  * Genesis: foot geoms = MuJoCo compile-time default (typically 1.0)
    unless an authored value is present in the XML.
  * Newton: foot geoms = ``ShapeConfig.mu`` default (1.0) unless the
    MJCF parser inherits the value from a ``<default>`` block.

A divergence between the three is a sim-parity issue that the user
should resolve (either by adding ``friction="0.6"`` to the MJCF's
``foot_capsule`` default, or by removing the mjlab override).

Usage:
    python -m rlworld.scripts.diag.check_foot_friction
    python -m rlworld.scripts.diag.check_foot_friction --sim genesis
    python -m rlworld.scripts.diag.check_foot_friction --preset g1_29dof --num-envs 1
"""

from __future__ import annotations

import argparse
import os
import re

# Multi-sim run in one process — bypass the single-backend guard.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np

_PRESETS: dict[str, tuple[str, str]] = {
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
}
_SIMS = ("genesis", "newton", "mujoco")

# Geoms matching this regex are highlighted as the "feet" group. Adjust
# to whichever pattern the preset's collision geoms use. Covers:
#   * mjlab / Newton: ``{left,right}_foot[1-7]_collision`` (MJCF geom names)
#   * Genesis: ``{left,right}_ankle_roll_link/g<i>`` (synthesised — Genesis
#     loses geom names during MJCF import; we label by parent link + index)
_FOOT_REGEX = r"^(left|right)_(foot[1-7]_collision|ankle_roll_link/g\d+)$"


def _build_env(preset: str, sim: str, num_envs: int):
    import importlib

    from rlworld.rl.runners import BaseRunner

    mod_path, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _read_genesis_friction(env) -> dict[str, tuple[float, float, float]]:
    """Return {synthetic_name: (slide, torsion, roll)} for every collision geom.

    Genesis's MJCF parser discards the geom name (parse_geom builds an
    info dict without a ``name`` key), and ``RigidGeom`` doesn't have a
    ``.name`` property — so cross-sim name matching is impossible at the
    geom granularity. We synthesise the name as ``<link_name>/g<i>``
    where ``i`` is the geom's position within the link; this still lets
    the user eyeball "is foot-link friction 0.6 or 1.0?" against the
    mjlab/Newton rows.

    Genesis models only sliding friction as a scalar (``_friction``);
    torsion / roll are not separately stored, so they read as 0.0.
    """
    out: dict[str, tuple[float, float, float]] = {}
    robot = env.scene_manager["robot"]
    geoms = robot.geoms  # collision geoms only (visuals are in ``vgeoms``)
    # Group geoms by parent link so we can label them within their link.
    by_link: dict[str, list] = {}
    for geom in geoms:
        link_name = geom.link.name if geom.link is not None else "unknown"
        by_link.setdefault(link_name, []).append(geom)
    for link_name, link_geoms in by_link.items():
        for i, geom in enumerate(link_geoms):
            slide = float(geom.friction)
            label = f"{link_name}/g{i}"
            out[label] = (slide, 0.0, 0.0)
    return out


def _read_newton_friction(env) -> dict[str, tuple[float, float, float]]:
    """Return {geom_name: (slide, torsion, roll)} for every collision shape."""

    out: dict[str, tuple[float, float, float]] = {}
    model = env.scene_manager.solver.model
    n_shapes = model.shape_count
    # shape_material_mu / mu_torsional / mu_rolling are wp.array — pull as torch then numpy.
    mu = (
        model.shape_material_mu.numpy()
        if hasattr(model.shape_material_mu, "numpy")
        else np.asarray(model.shape_material_mu)
    )
    mu_tor = (
        model.shape_material_mu_torsional.numpy()
        if hasattr(model.shape_material_mu_torsional, "numpy")
        else np.asarray(model.shape_material_mu_torsional)
    )
    mu_rol = (
        model.shape_material_mu_rolling.numpy()
        if hasattr(model.shape_material_mu_rolling, "numpy")
        else np.asarray(model.shape_material_mu_rolling)
    )
    labels = list(getattr(model, "shape_label", []))
    for i in range(n_shapes):
        name = labels[i] if i < len(labels) else f"shape_{i}"
        # Drop the world/entity prefix Newton's MJCF importer adds (e.g. ``robot/g1_29dof/left_foot1_collision``).
        bare = name.rsplit("/", 1)[-1] if "/" in name else name
        out[bare] = (float(mu[i]), float(mu_tor[i]), float(mu_rol[i]))
    return out


def _read_mujoco_friction(env) -> dict[str, tuple[float, float, float]]:
    """Return {geom_name: (slide, torsion, roll)} from the live MjModel."""
    import mujoco

    out: dict[str, tuple[float, float, float]] = {}
    mj_model = env.scene_manager.mj_model
    for i in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name is None:
            continue
        # Drop any world/entity prefix mjlab might prepend.
        bare = name.rsplit("/", 1)[-1] if "/" in name else name
        f = mj_model.geom_friction[i]
        out[bare] = (float(f[0]), float(f[1]), float(f[2]))
    return out


_READERS = {
    "genesis": _read_genesis_friction,
    "newton": _read_newton_friction,
    "mujoco": _read_mujoco_friction,
}


def _fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0]:6.3f} / {t[1]:7.4f} / {t[2]:8.5f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(_PRESETS), default="g1_29dof")
    ap.add_argument("--sim", choices=[*_SIMS, "all"], default="all")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--foot-regex", default=_FOOT_REGEX)
    args = ap.parse_args()

    sims = list(_SIMS) if args.sim == "all" else [args.sim]
    results: dict[str, dict[str, tuple[float, float, float]]] = {}

    for sim in sims:
        print(f"\n{'=' * 78}\nBuilding [{sim}] {args.preset!r} (num_envs={args.num_envs}) ...")
        env = _build_env(args.preset, sim, args.num_envs)
        try:
            results[sim] = _READERS[sim](env)
            print(f"  read {len(results[sim])} collision geoms")
        except Exception as e:
            print(f"  ERROR reading friction: {type(e).__name__}: {e}")
            results[sim] = {}
        del env
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Union of geom names across sims (alphabetical for stable order).
    all_names = sorted({n for d in results.values() for n in d})
    foot_re = re.compile(args.foot_regex)
    foot_names = [n for n in all_names if foot_re.match(n)]
    other_names = [n for n in all_names if not foot_re.match(n)]

    # ── Per-section tables ───────────────────────────────────────────
    sim_w = max(len(s) for s in sims)
    cell_w = 28  # width for "X.XXX / X.XXXX / X.XXXXX"
    name_w = max(28, max((len(n) for n in all_names), default=28))

    def _print_section(title: str, names: list[str]) -> None:
        print(f"\n{'=' * 78}\n{title}  (slide / torsion / roll)")
        header = f"{'geom':<{name_w}}  " + "  ".join(f"{s:^{cell_w}}" for s in sims) + f"  {'slide Δ%':>9}"
        print(header)
        print("─" * len(header))
        for n in names:
            cells = []
            slides = []
            for sim in sims:
                t = results[sim].get(n)
                if t is None:
                    cells.append("--")
                else:
                    cells.append(_fmt(t))
                    slides.append(t[0])
            # slide-only cross-sim Δ%
            if len(slides) >= 2 and max(slides) > 1e-6:
                delta_pct = (max(slides) - min(slides)) / max(slides) * 100.0
                delta_cell = f"{delta_pct:7.2f}%"
            else:
                delta_cell = "--"
            body = "  ".join(f"{c:^{cell_w}}" for c in cells)
            print(f"{n:<{name_w}}  {body}  {delta_cell:>9}")

    if foot_names:
        _print_section(f"FOOT geoms (regex {args.foot_regex!r})", foot_names)
    else:
        print(f"\n(no foot geoms matched regex {args.foot_regex!r})")

    if other_names:
        _print_section("OTHER collision geoms (for contrast)", other_names)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 78}\nVERDICT")
    if len(sims) < 2:
        print("  (single-sim mode — no cross-sim comparison)")
        return 0

    worst_pct = 0.0
    worst_geom = ""
    for n in foot_names:
        slides = [results[sim].get(n, (np.nan,))[0] for sim in sims]
        slides = [s for s in slides if not np.isnan(s)]
        if len(slides) < 2 or max(slides) <= 1e-6:
            continue
        pct = (max(slides) - min(slides)) / max(slides) * 100.0
        if pct > worst_pct:
            worst_pct = pct
            worst_geom = n

    if worst_pct < 1.0:
        print(f"  foot-friction worst Δ across sims: {worst_pct:.2f}%  [PASS]")
    else:
        print(f"  foot-friction worst Δ across sims: {worst_pct:.2f}%  ({worst_geom})  [FAIL]")
        print(
            "  → mjlab and Genesis/Newton are reading different sliding friction for the feet. "
            'Options: (a) add `friction="0.6" priority="1" condim="3"` to the MJCF\'s '
            '`<default class="foot_capsule">` so all three sims pick it up at import time; '
            "(b) remove the mjlab `FULL_COLLISION` foot-friction override so all three default to 1.0."
        )
    return 0 if worst_pct < 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
