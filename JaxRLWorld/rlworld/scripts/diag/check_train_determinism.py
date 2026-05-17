"""Verify run-to-run determinism on the **training** code path across 3 sims.

The eval-path counterpart (``check_eval_determinism``) exercises the
sim_initializer + eval-mode defaults (no obs noise, reset_dr / interval
events stripped). This script instead exercises the *training* code
path used by every actual training command: env is built via
``BaseRunner.create_with_env(cfgs)`` from the preset, and the
training-time stochastic surfaces are **left active**:

  * observation noise per group is preserved
  * ``interval`` and ``reset_dr`` event terms are preserved — these
    fire on every reset and re-sample the per-env DR

That mix exercises every RNG surface a real training run hits each
reset / step. Compared to ``check_eval_determinism``, the training
path has many more torch / numpy draws per reset, so it's a stricter
test of "does ``set_seed`` + ``wp.rand_init`` + ``gs.init(seed=...)``
actually pin everything that gets drawn during the rollout schedule?"

How it works (mirrors the eval-determinism diag):

  1. The script invokes itself twice as separate subprocesses (via
     ``--inner``) so each run starts from a fresh Python process.
  2. Each inner run builds the env via the training path with a fixed
     seed, resets, then alternates ``env.reset()`` and ``env.step()``
     with a deterministic scripted action sequence over ``--rollouts``
     resets × ``--steps`` steps each. Per-step state (joint_pos /
     joint_vel / root_pos / root_lin_vel / contact_force_sq_sum) is
     dumped as a numeric table to the run-specific file.
  3. The driver process parses both run files, computes drift per sim
     and per field-section, and writes the verdict to
     ``diag/train_determinism.txt``.

Usage:

    python -m rlworld.scripts.diag.check_train_determinism
    python -m rlworld.scripts.diag.check_train_determinism --steps 30 --rollouts 3
    python -m rlworld.scripts.diag.check_train_determinism --seed 7 --sim newton
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

# Multi-sim run in one process — bypass the single-backend guard.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

_PRESETS: dict[str, tuple[str, str]] = {
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "go2_flat": ("rlworld.rl.configs.presets.go2_flat.base", "Go2FlatConfig"),
    "t1_tracking": ("rlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
}
_SIMS = ("genesis", "newton", "mujoco")
_OUT_DEFAULT = "diag/train_determinism.txt"


# ---------------------------------------------------------------------------
# Inner: build training env, rollout, dump
# ---------------------------------------------------------------------------


def _build_train_env(sim: str, preset: str, num_envs: int, seed: int):
    """Build env via the training code path.

    Mirrors the training builder used by every preset: construct the
    preset config, pin ``cfgs.env.seed``, then go through
    ``BaseRunner.create_with_env`` exactly the way the training entry
    points do. Observation noise and ``reset_dr`` / ``interval`` event
    terms are left active — that's the whole point of this diag,
    versus :mod:`check_eval_determinism` which strips them.
    """
    from rlworld.rl.runners import BaseRunner

    mod_path, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    cfgs.env.seed = seed
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _capture_state(env) -> list[float]:
    """Return a flat list of floats describing the env's env-0 state.

    Field order is part of the on-disk format the driver parses; do
    NOT reorder without updating ``_section_slices``.
    """
    import torch

    rd = env.get_robot_data()
    out: list[float] = []
    out.extend(rd.joint_pos[0].detach().cpu().numpy().tolist())
    out.extend(rd.joint_vel[0].detach().cpu().numpy().tolist())
    try:
        ids = torch.tensor([0], device=env.device, dtype=torch.long)
        out.extend(rd.body_pos_w_by_ids(ids)[0, 0].detach().cpu().numpy().tolist())
        out.extend(rd.body_lin_vel_w_by_ids(ids)[0, 0].detach().cpu().numpy().tolist())
    except Exception:
        out.extend([0.0] * 6)
    try:
        groups = list(env.contact_manager._groups.keys())
        if groups:
            f = env.contact_manager.contact_force(groups[0])[0].detach().cpu().numpy()
            out.append(float((f * f).sum()))
        else:
            out.append(0.0)
    except Exception:
        out.append(0.0)
    return out


def _build_action_seq(env, steps: int, seed: int):
    """Deterministic action sequence — same seed → same actions."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed + 99999)
    actions = []
    for _ in range(steps):
        a = (torch.rand(env.num_envs, env.num_actions, generator=g, device="cpu") * 2.0 - 1.0) * 0.2
        actions.append(a.to(env.device))
    return actions


def _run_inner(sim: str, preset: str, num_envs: int, seed: int, rollouts: int, steps: int, out_path: Path) -> int:
    """Build env, run the rollout schedule, dump a numeric trace to disk."""
    import torch

    lines: list[str] = []
    lines.append(f"# sim={sim} preset={preset} num_envs={num_envs} seed={seed} rollouts={rollouts} steps={steps}")
    try:
        env = _build_train_env(sim, preset, num_envs, seed)
    except Exception as e:
        lines.append(f"# BUILD_ERROR: {type(e).__name__}: {e}")
        lines.append(traceback.format_exc())
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return 1

    n_joints = int(env.act_manager._total_action_dim)
    lines.append(f"# n_joints={n_joints}")

    action_seq = _build_action_seq(env, steps, seed)

    global_step = 0
    for rollout_idx in range(rollouts):
        try:
            env.reset()
            state = _capture_state(env)
            lines.append(f"{rollout_idx} {global_step} reset " + " ".join(f"{v:.10f}" for v in state))
            for step_idx in range(steps):
                action = action_seq[step_idx]
                env.step(action)
                state = _capture_state(env)
                lines.append(f"{rollout_idx} {global_step} step{step_idx} " + " ".join(f"{v:.10f}" for v in state))
                global_step += 1
        except Exception as e:
            lines.append(f"# STEP_ERROR rollout={rollout_idx}: {type(e).__name__}: {e}")
            lines.append(traceback.format_exc())
            break

    out_path.write_text("\n".join(lines), encoding="utf-8")

    del env
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return 0


# ---------------------------------------------------------------------------
# Driver: run inner twice as subprocess, compare
# ---------------------------------------------------------------------------


def _spawn_inner(sim: str, run_label: str, args, out_path: Path) -> int:
    """Run the inner trace as a fresh subprocess for the given sim/run."""
    cmd = [
        sys.executable,
        "-m",
        "rlworld.scripts.diag.check_train_determinism",
        "--inner",
        "--sim",
        sim,
        "--preset",
        args.preset,
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--rollouts",
        str(args.rollouts),
        "--steps",
        str(args.steps),
        "--out",
        str(out_path),
    ]
    print(f"[{sim}/{run_label}] $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


_DATA_LINE = re.compile(r"^(\d+)\s+(\d+)\s+(\S+)\s+(.+)$")
_N_JOINTS_LINE = re.compile(r"^#\s*n_joints=(\d+)\s*$")


def _parse_trace(path: Path) -> tuple[int, list[tuple[str, list[float]]]]:
    n_joints = 0
    rows: list[tuple[str, list[float]]] = []
    for line in path.read_text().splitlines():
        m_nj = _N_JOINTS_LINE.match(line)
        if m_nj:
            n_joints = int(m_nj.group(1))
            continue
        m = _DATA_LINE.match(line)
        if not m:
            continue
        rollout, _gstep, step_label, payload = m.groups()
        try:
            values = [float(x) for x in payload.split()]
        except ValueError:
            continue
        rows.append((f"R{rollout}_{step_label}", values))
    return n_joints, rows


def _section_slices(n_joints: int) -> list[tuple[str, int, int]]:
    a = n_joints
    return [
        ("joint_pos", 0, a),
        ("joint_vel", a, 2 * a),
        ("root_pos", 2 * a, 2 * a + 3),
        ("root_lin_vel", 2 * a + 3, 2 * a + 6),
        ("contact_force_sq", 2 * a + 6, 2 * a + 7),
    ]


def _compare(rows_a, rows_b, n_joints: int) -> dict:
    sections = _section_slices(n_joints) if n_joints > 0 else []
    n = min(len(rows_a), len(rows_b))
    out: dict = {
        "n_rows_a": len(rows_a),
        "n_rows_b": len(rows_b),
        "n_aligned": n,
        "first_diff_idx": -1,
        "first_diff_tag": "",
        "max_abs": 0.0,
        "max_abs_tag": "",
        "max_abs_col": -1,
        "n_diff_rows": 0,
        "sections": {name: {"max_abs": 0.0, "sum_abs": 0.0, "n_diff": 0, "last_abs": 0.0} for name, _, _ in sections},
    }
    for i in range(n):
        tag_a, va = rows_a[i]
        tag_b, vb = rows_b[i]
        if tag_a != tag_b:
            out["first_diff_idx"] = i
            out["first_diff_tag"] = f"{tag_a}!={tag_b}"
            break
        if len(va) != len(vb):
            out["first_diff_idx"] = i
            out["first_diff_tag"] = f"{tag_a} (len mismatch {len(va)}/{len(vb)})"
            break
        row_diff = 0.0
        row_max_col = -1
        for j in range(len(va)):
            d = abs(va[j] - vb[j])
            if d > row_diff:
                row_diff = d
                row_max_col = j
        if row_diff > 0:
            out["n_diff_rows"] += 1
            if out["first_diff_idx"] < 0:
                out["first_diff_idx"] = i
                out["first_diff_tag"] = tag_a
            if row_diff > out["max_abs"]:
                out["max_abs"] = row_diff
                out["max_abs_tag"] = tag_a
                out["max_abs_col"] = row_max_col

        for name, lo, hi in sections:
            seg_max = 0.0
            for j in range(lo, hi):
                d = abs(va[j] - vb[j])
                if d > seg_max:
                    seg_max = d
            s = out["sections"][name]
            if seg_max > 0:
                s["n_diff"] += 1
                s["sum_abs"] += seg_max
                if seg_max > s["max_abs"]:
                    s["max_abs"] = seg_max
            s["last_abs"] = seg_max
    return out


_SECTION_NAMES = ("joint_pos", "joint_vel", "root_pos", "root_lin_vel", "contact_force_sq")


def _format_summary(per_sim: dict, args) -> str:
    lines: list[str] = []
    lines.append("=" * 110)
    lines.append(
        "Training-path determinism verdict — "
        f"preset={args.preset}  seed={args.seed}  "
        f"rollouts={args.rollouts}  steps/rollout={args.steps}  num_envs={args.num_envs}"
    )
    lines.append("  (obs noise + reset_dr / interval events kept active — the actual training-time RNG surface)")
    lines.append("=" * 110)
    for sim in _SIMS:
        if sim not in per_sim:
            lines.append(f"  [{sim:>7s}] SKIPPED")
            continue
        info = per_sim[sim]
        if info.get("error"):
            lines.append(f"  [{sim:>7s}] BUILD/RUN ERROR — see trace file")
            continue
        n_rows = info["n_aligned"]
        if info["n_diff_rows"] == 0 and n_rows > 0:
            lines.append(f"  [{sim:>7s}] DETERMINISTIC ({n_rows} aligned rows, all values match)")
        else:
            lines.append(
                f"  [{sim:>7s}] DRIFT "
                f"({info['n_diff_rows']}/{n_rows} rows differ; "
                f"first divergence at row {info['first_diff_idx']} [{info['first_diff_tag']}]; "
                f"max |Δ|={info['max_abs']:.6g} at column {info['max_abs_col']})"
            )
    lines.append("=" * 110)

    lines.append("")
    lines.append("Per-field-section drift breakdown")
    lines.append("=" * 110)
    header = (
        f"  {'sim':<8s} | {'section':<18s} | {'n_diff':>6s} | "
        f"{'max |Δ|':>12s} | {'mean |Δ|':>12s} | {'last-row |Δ|':>14s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for sim in _SIMS:
        if sim not in per_sim or per_sim[sim].get("error"):
            continue
        info = per_sim[sim]
        sections = info.get("sections", {})
        for name in _SECTION_NAMES:
            s = sections.get(name)
            if s is None:
                continue
            mean_abs = (s["sum_abs"] / s["n_diff"]) if s["n_diff"] > 0 else 0.0
            lines.append(
                f"  {sim:<8s} | {name:<18s} | {s['n_diff']:>6d} | "
                f"{s['max_abs']:>12.6g} | {mean_abs:>12.6g} | {s['last_abs']:>14.6g}"
            )
        lines.append("  " + "-" * (len(header) - 2))
    lines.append("")
    lines.append("Field layout per row (in order):")
    lines.append("  joint_pos[0..N-1], joint_vel[0..N-1], root_pos[xyz], root_lin_vel[xyz], contact_force_sq_sum")
    lines.append("")
    lines.append(
        "Interpretation:\n"
        "  * ``max |Δ|`` is the worst per-row peak across the whole rollout for that section.\n"
        "  * ``mean |Δ|`` averages the per-row peak over rows that actually differ — small\n"
        "    here + large ``max`` = a few spike rows, the rest near-identical.\n"
        "  * ``last-row |Δ|`` is the per-row peak on the final captured row — the\n"
        "    cumulative drift at the end of the rollout.\n"
        "  * Compared to ``check_eval_determinism``: ``reset_dr`` and obs noise are\n"
        "    *active* here, so any additional drift over the eval-path numbers must\n"
        "    come from those terms' RNG draws (not from the physics step itself)."
    )
    return "\n".join(lines)


def _driver(args) -> int:
    sims = list(_SIMS) if args.sim == "all" else [args.sim]
    workdir = Path("diag")
    workdir.mkdir(parents=True, exist_ok=True)

    per_sim: dict[str, dict] = {}
    files_to_clean: list[Path] = []
    for sim in sims:
        out_a = workdir / f"_train_trace_{sim}_run_a.txt"
        out_b = workdir / f"_train_trace_{sim}_run_b.txt"
        files_to_clean.extend([out_a, out_b])
        rc_a = _spawn_inner(sim, "run_a", args, out_a)
        rc_b = _spawn_inner(sim, "run_b", args, out_b)
        if rc_a != 0 or rc_b != 0:
            per_sim[sim] = {"error": True}
            continue
        nj_a, rows_a = _parse_trace(out_a)
        nj_b, rows_b = _parse_trace(out_b)
        n_joints = nj_a if nj_a > 0 else nj_b
        per_sim[sim] = _compare(rows_a, rows_b, n_joints)

    summary = _format_summary(per_sim, args)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nWrote: {out_path}")

    if not args.keep:
        for f in files_to_clean:
            f.unlink(missing_ok=True)
    else:
        print(f"  (kept intermediate trace files in {workdir}/)")

    any_drift = any(per_sim[s].get("n_diff_rows", 0) > 0 or per_sim[s].get("error") for s in sims)
    return 1 if any_drift else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(_PRESETS), default="g1_29dof")
    ap.add_argument("--sim", choices=[*_SIMS, "all"], default="all")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=2, help="env.reset cycles per run.")
    ap.add_argument("--steps", type=int, default=20, help="env.step calls per rollout.")
    ap.add_argument("--out", default=_OUT_DEFAULT, help="Final report path (driver mode).")
    ap.add_argument("--keep", action="store_true", help="Keep intermediate trace files for inspection.")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.inner:
        if args.sim not in _SIMS:
            print(f"--inner requires --sim in {_SIMS}, got {args.sim!r}", file=sys.stderr)
            return 2
        return _run_inner(
            sim=args.sim,
            preset=args.preset,
            num_envs=args.num_envs,
            seed=args.seed,
            rollouts=args.rollouts,
            steps=args.steps,
            out_path=Path(args.out),
        )

    return _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
