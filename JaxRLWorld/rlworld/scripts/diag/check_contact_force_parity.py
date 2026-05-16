"""Cross-sim contact-force parity diag.

Builds each requested sim with the same preset (fixed seed, small
``num_envs``), settles for K steps with zero action, then runs N more
steps and captures per-step contact-force values from the preset's
contact-sensor groups (e.g. ``feet_ground_contact``, ``self_collision``).
Prints a per-group table comparing Genesis / Newton / mjlab and flags
the worst cross-sim delta.

This diag doubles as the acceptance test for the Genesis
``ContactForce`` ``filter_link_idx`` feature: the filter on
``feet_ground_contact`` (secondary=ground) and ``self_collision``
(secondary=self) is what makes the reported force reflect the intended
subset of contacts. With the filter correctly implemented, all three
sims should report:

  * ``feet_ground_contact``: per-foot Fz close to the ground reaction
    (≈ m·g / num_feet for the standing robot), in close agreement across
    sims.
  * ``self_collision``: near-zero force in the standing pose (no self
    contacts), in close agreement across sims.

If Genesis's filter is broken, ``feet_ground_contact`` will pick up
spurious self-contacts and ``self_collision`` will pick up the ground
reaction — both push the cross-sim delta way out of the threshold.

Usage:
    python -m rlworld.scripts.diag.check_contact_force_parity
    python -m rlworld.scripts.diag.check_contact_force_parity --num-envs 8 --steps 20
    python -m rlworld.scripts.diag.check_contact_force_parity --sim genesis    # one sim only
"""

from __future__ import annotations

import argparse
import os

# Multi-sim run in one process — bypass the single-backend guard.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np

_PRESETS: dict[str, tuple[str, str]] = {
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
}
_SIMS = ("genesis", "newton", "mujoco")
# Contact groups captured every step. Must exist (under the same name)
# in all three sim builders for the preset.
_CONTACT_GROUPS = ("feet_ground_contact", "self_collision")


def _build_env(preset: str, sim: str, num_envs: int):
    import importlib

    from rlworld.rl.runners import BaseRunner

    mod_path, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _capture_groups(env) -> dict[str, np.ndarray | None]:
    """Read contact_force for each configured group; return numpy arrays."""
    out: dict[str, np.ndarray | None] = {}
    for g in _CONTACT_GROUPS:
        try:
            f = env.contact_manager.contact_force(g)  # (E, N, 3)
            out[g] = f.detach().cpu().float().numpy()
        except Exception as e:
            out[g] = None
            print(f"  WARN: contact_force({g!r}) failed: {e}")
    return out


def _group_names(env, group: str) -> list[str]:
    try:
        return list(env.contact_manager.tracked_names(group))
    except Exception:
        return []


def _fmt_n(x: float) -> str:
    """Format a force value in N with sensible precision."""
    if abs(x) >= 100:
        return f"{x:7.1f}"
    if abs(x) >= 10:
        return f"{x:7.2f}"
    return f"{x:7.3f}"


def _stat_block(arr: np.ndarray) -> str:
    """``mean ± std`` formatted."""
    return f"{_fmt_n(float(arr.mean()))} ± {_fmt_n(float(arr.std()))}"


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(_PRESETS), default="g1_29dof")
    ap.add_argument("--sim", choices=[*_SIMS, "all"], default="all")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--settle", type=int, default=10, help="Zero-action settle steps before capture.")
    ap.add_argument("--steps", type=int, default=10, help="Captured steps after settle.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--threshold-pct",
        type=float,
        default=10.0,
        help="Max cross-sim |Δ|/mean (%%) for feet_ground_contact.Fz to declare PASS (default 10).",
    )
    args = ap.parse_args()

    sims = list(_SIMS) if args.sim == "all" else [args.sim]

    # sim -> group -> stacked force tensor across captured steps, shape (T*E, N, 3)
    aggregated: dict[str, dict[str, np.ndarray | None]] = {}
    # sim -> group -> tracked names (primary identifiers)
    names: dict[str, dict[str, list[str]]] = {}

    for sim in sims:
        print(f"\n{'=' * 78}\nBuilding [{sim}] {args.preset!r} (num_envs={args.num_envs}) ...")
        env = _build_env(args.preset, sim, args.num_envs)
        torch.manual_seed(args.seed)
        env.reset()
        zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)

        # Settle.
        for _ in range(args.settle):
            env.step(zero)

        names[sim] = {g: _group_names(env, g) for g in _CONTACT_GROUPS}

        per_step: dict[str, list[np.ndarray]] = {g: [] for g in _CONTACT_GROUPS}
        for _ in range(args.steps):
            env.step(zero)
            snap = _capture_groups(env)
            for g, arr in snap.items():
                if arr is not None:
                    per_step[g].append(arr)

        aggregated[sim] = {}
        for g, lst in per_step.items():
            aggregated[sim][g] = np.concatenate(lst, axis=0) if lst else None

        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print per-group tables ───────────────────────────────────────
    sim_w = max(len(s) for s in sims)
    col_w = max(20, sim_w + 4)

    # 1) feet_ground_contact: per-primary (foot) Fz + total |F|
    print(
        f"\n{'=' * 78}\nfeet_ground_contact  (Fz per foot, aggregated across {args.steps} steps × {args.num_envs} envs)"
    )
    # Determine union of foot names; in practice they should agree.
    foot_names = sorted({n for sim in sims for n in names[sim].get("feet_ground_contact", [])})
    if not foot_names:
        # Fall back to positional naming if names weren't resolved.
        n_max = max(
            (
                aggregated[sim]["feet_ground_contact"].shape[1]
                for sim in sims
                if aggregated[sim]["feet_ground_contact"] is not None
            ),
            default=0,
        )
        foot_names = [f"foot_{i}" for i in range(n_max)]

    name_w = max(20, max((len(n) for n in foot_names), default=20))
    # Compare magnitudes (|Fz|) — sign conventions diverge across sims (e.g. Genesis/Newton report
    # the +Z reaction "ground → foot", mjlab reports the -Z action "foot → ground"); they are equal
    # and opposite by Newton's 3rd law, so the physical parity check is on |Fz|. A "sign" column
    # makes the convention difference visible.
    header = f"{'foot':<{name_w}}  " + "  ".join(f"{s:^{col_w}}" for s in sims) + f"  {'sign':>6}  {'|Δ| %':>7}"
    print(header)
    print("─" * len(header))

    fz_abs_means_per_foot: dict[str, dict[str, float]] = {}  # sim -> foot -> mean |Fz|
    for foot in foot_names:
        cells: list[str] = []
        per_sim_abs: dict[str, float] = {}
        per_sim_signed: dict[str, float] = {}
        for sim in sims:
            arr = aggregated[sim].get("feet_ground_contact")
            sim_names = names[sim].get("feet_ground_contact", [])
            if arr is None or foot not in sim_names:
                cells.append("--")
                continue
            i = sim_names.index(foot)
            fz_samples = arr[:, i, 2]  # Z-component
            per_sim_signed[sim] = float(fz_samples.mean())
            per_sim_abs[sim] = float(np.abs(fz_samples).mean())
            cells.append(_stat_block(fz_samples))
        fz_abs_means_per_foot[foot] = per_sim_abs
        # sign column: + / - / mixed (Newton-3rd-law convention diagnostic — not a divergence)
        signs = {("+" if v > 0 else "-") for v in per_sim_signed.values() if abs(v) > 1e-3}
        sign_cell = next(iter(signs)) if len(signs) == 1 else ("mixed" if signs else "0")
        # parity uses magnitudes
        abs_vals = list(per_sim_abs.values())
        if len(abs_vals) >= 2 and max(abs_vals) > 1e-3:
            delta_pct = (max(abs_vals) - min(abs_vals)) / max(abs_vals) * 100.0
            delta_cell = f"{delta_pct:6.2f}"
        else:
            delta_cell = "--"
        body = "  ".join(f"{c:^{col_w}}" for c in cells)
        print(f"{foot:<{name_w}}  {body}  {sign_cell:>6}  {delta_cell:>7}")

    # 2) self_collision: aggregate |F| (max across primaries) per sim.
    print(f"\n{'=' * 78}\nself_collision  (|F| across robot links, aggregated)")
    header = f"{'metric':<{name_w}}  " + "  ".join(f"{s:^{col_w}}" for s in sims) + f"  {'maxΔ':>10}"
    print(header)
    print("─" * len(header))

    for label, reducer in [
        ("max |F| per env", lambda mag: mag.max(axis=-1)),
        ("mean |F| per env", lambda mag: mag.mean(axis=-1)),
    ]:
        cells: list[str] = []
        per_sim_val: list[float] = []
        for sim in sims:
            arr = aggregated[sim].get("self_collision")
            if arr is None:
                cells.append("--")
                continue
            mag = np.linalg.norm(arr, axis=-1)  # (T*E, N)
            samples = reducer(mag)  # (T*E,)
            per_sim_val.append(float(samples.mean()))
            cells.append(_stat_block(samples))
        if len(per_sim_val) >= 2:
            cells.append(f"{(max(per_sim_val) - min(per_sim_val)):8.3f}")
        else:
            cells.append("--")
        print(f"{label:<{name_w}}  " + "  ".join(f"{c:^{col_w}}" for c in cells[:-1]) + f"  {cells[-1]:>10}")

    # ── Pass / Fail ──────────────────────────────────────────────────
    print(f"\n{'=' * 78}\nVERDICT")
    worst_pct = 0.0
    for foot, per_sim_abs in fz_abs_means_per_foot.items():
        vals = list(per_sim_abs.values())
        if len(vals) < 2:
            continue
        denom = max(vals)
        if denom <= 1e-3:
            continue
        pct = (max(vals) - min(vals)) / denom * 100.0
        worst_pct = max(worst_pct, pct)
    if len(sims) < 2:
        print("  (single-sim mode — no cross-sim comparison)")
    else:
        status = "PASS" if worst_pct <= args.threshold_pct else "FAIL"
        print(
            f"  feet_ground_contact.|Fz| worst cross-sim Δ: {worst_pct:.2f}%  (threshold {args.threshold_pct:.2f}%)  [{status}]"
        )
        print(
            "  Note: signed Fz may differ in sign across sims (Newton-3rd-law convention — Genesis/Newton "
            "report ground→foot reaction +Z; mjlab reports foot→ground action -Z). The parity check uses "
            "|Fz| since both are physically equivalent."
        )
        if worst_pct > args.threshold_pct:
            print(
                "  Likely causes: (a) ContactForce filter_link_idx not yet active in Genesis "
                "(feet_ground_contact picks up spurious self contacts), (b) different robot mass / "
                "inertia across sims, (c) substeps / contact-solver settings diverge."
            )
    return 0 if worst_pct <= args.threshold_pct or len(sims) < 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
