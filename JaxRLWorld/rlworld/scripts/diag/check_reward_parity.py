"""Cross-sim per-reward initial-value comparison.

Builds each requested sim with the same preset (small ``num_envs``, fixed
seed, identical initial state by construction), steps with zero action for
N steps, and captures **per-reward raw values** (before ``weight * dt``)
at every step. Prints a table per step:

  term                | genesis (mean ± std) | newton (mean ± std) | mujoco (mean ± std)

Use this to localize sim-to-sim reward divergences without re-running full
training. The capture is implemented by monkey-patching the reward
manager's ``_compute_weighted_reward`` while ``env.step()`` runs, so
stateful reward terms (e.g. ``feet_swing_height``) are invoked exactly
once per step — same as production.

Sets ``JAXRLWORLD_ALLOW_MULTI_SIM=1`` for itself so the single-backend
guard doesn't reject multi-sim runs.

Usage:
    python -m rlworld.scripts.diag.check_reward_parity                    # g1_29dof, all 3 sims
    python -m rlworld.scripts.diag.check_reward_parity --preset go2_flat
    python -m rlworld.scripts.diag.check_reward_parity --sim newton       # one sim
    python -m rlworld.scripts.diag.check_reward_parity --steps 50 --terms feet_slip,angular_momentum_penalty
"""

from __future__ import annotations

import argparse
import os

# This diag builds multiple sim backends sequentially in one process —
# bypass the single-backend guard in BaseRunner.create_with_env.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np

_PRESETS: dict[str, tuple[str, str]] = {
    "go2_flat": ("rlworld.rl.configs.presets.go2_flat.base", "Go2FlatConfig"),
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("rlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
    "t1_tracking": ("rlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
}
_SIMS = ("genesis", "newton", "mujoco")


def _build_env(preset: str, sim: str, num_envs: int):
    import importlib

    from rlworld.rl.runners import BaseRunner

    mod_path, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _capture_step(env, action) -> dict[str, np.ndarray]:
    """Step env once and capture per-reward raw values (before weight*dt).

    Monkey-patches the reward manager's ``_compute_weighted_reward`` for
    the duration of one ``env.step()`` so each reward function is invoked
    exactly once (same as a normal training step) and we record its raw
    output. The patched method's weighted return is bit-identical to the
    original so the env still trains correctly.
    """
    from rlworld.rl.envs.managers.common.reward import get_weight_value

    mgr = env.reward_manager
    captured: dict[str, np.ndarray] = {}
    orig = mgr._compute_weighted_reward

    def wrapper(name, term):
        if name in mgr._instances:
            raw = mgr._instances[name](mgr.env)
        else:
            raw = mgr._resolved_fns[name](mgr.env, **term.params)
        captured[name] = raw.detach().cpu().float().numpy()
        w = get_weight_value(term.weight, mgr.env_step_calls)
        return raw * w * mgr.env.control_dt

    mgr._compute_weighted_reward = wrapper
    try:
        env.step(action)
    finally:
        mgr._compute_weighted_reward = orig
    return captured


def _stat(arr: np.ndarray) -> str:
    if arr is None:
        return "—"
    return f"{arr.mean():+.5f} ± {arr.std():.5f}"


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(_PRESETS), default="g1_29dof")
    ap.add_argument("--sim", choices=[*_SIMS, "all"], default="all")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--terms",
        type=str,
        default=None,
        help="Comma-separated subset of reward term names to display "
        "(default: all). Useful for focusing: --terms feet_slip,feet_clearance",
    )
    args = ap.parse_args()

    term_filter = set(t.strip() for t in args.terms.split(",")) if args.terms else None

    sims = list(_SIMS) if args.sim == "all" else [args.sim]
    # sim -> step -> {term -> per-env values}
    results: dict[str, dict[int, dict[str, np.ndarray]]] = {}

    for sim in sims:
        print(f"\n{'=' * 70}\nBuilding [{sim}] {args.preset!r} (num_envs={args.num_envs}) ...")
        env = _build_env(args.preset, sim, args.num_envs)
        torch.manual_seed(args.seed)
        env.reset()
        zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        results[sim] = {}
        for step in range(args.steps):
            results[sim][step] = _capture_step(env, zero)
        # Free the env so we don't blow GPU memory before the next sim.
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Union of all term names across sims.
    all_terms = sorted({n for r in results.values() for s in r.values() for n in s})
    if term_filter is not None:
        unknown = term_filter - set(all_terms)
        if unknown:
            print(f"WARN: unknown --terms entries (ignored): {sorted(unknown)}")
        all_terms = [t for t in all_terms if t in term_filter]

    sims_run = list(results.keys())
    name_w = max(len(n) for n in all_terms) + 1 if all_terms else 8
    col_w = 22  # width per sim column

    for step in range(args.steps):
        print(f"\n{'─' * 70}\n[step {step}]")
        header = f"{'term':<{name_w}}  " + "  ".join(f"{s:^{col_w}}" for s in sims_run)
        print(header)
        print("─" * len(header))
        for name in all_terms:
            cells = []
            for sim in sims_run:
                v = results[sim][step].get(name)
                cells.append(_stat(v))
            print(f"{name:<{name_w}}  " + "  ".join(f"{c:^{col_w}}" for c in cells))

    if len(sims_run) > 1:
        # Final-step summary: max abs ratio between sims per term, to flag
        # the biggest divergences at a glance.
        last = args.steps - 1
        print(f"\n{'=' * 70}\nDIVERGENCE SUMMARY (step {last}, mean across envs)\n")
        ref_sim = sims_run[0]
        print(f"{'term':<{name_w}}  " + "  ".join(f"{s:^{col_w}}" for s in sims_run))
        print("─" * (name_w + 2 + col_w * len(sims_run) + 2 * (len(sims_run) - 1)))
        for name in all_terms:
            ref = results[ref_sim][last].get(name)
            cells = []
            for sim in sims_run:
                v = results[sim][last].get(name)
                if v is None:
                    cells.append("—")
                    continue
                m = float(v.mean())
                if sim == ref_sim or ref is None:
                    cells.append(f"{m:+.5f}")
                else:
                    rm = float(ref.mean())
                    if abs(rm) > 1e-9:
                        ratio = m / rm
                        cells.append(f"{m:+.5f} ({ratio:+.2f}x)")
                    else:
                        cells.append(f"{m:+.5f}")
            print(f"{name:<{name_w}}  " + "  ".join(f"{c:^{col_w}}" for c in cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
