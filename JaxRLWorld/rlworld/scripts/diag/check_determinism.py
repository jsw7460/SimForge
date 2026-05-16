"""Run ``check_feet_slip_trace`` twice with the same seed and report per-sim drift.

Use this to verify the seed fix landed: a fully deterministic sim should
produce byte-identical numeric columns across the two runs. mjwarp-based
sims (Newton's ``SolverMuJoCo`` and mjlab) carry inherent non-determinism
in their parallel reductions (``mujoco_warp#562``), so some drift is
expected there even with a correctly fixed seed.

Reports, per sim:
  * whether all numeric values are identical across the two runs
  * max absolute drift across every numeric value the trace prints
  * worst-offender metric (which row diverged the most)

Usage:
    python -m rlworld.scripts.diag.check_determinism
    python -m rlworld.scripts.diag.check_determinism --seed 7 --steps 10
    python -m rlworld.scripts.diag.check_determinism --keep    # keep trace files
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_DIAG_MODULE = "rlworld.scripts.diag.check_feet_slip_trace"
_SIMS = ("genesis", "newton", "mujoco")

# Per-step table lines look like:
#   "  metric_name                  |  +0.12345 |  -0.67890 |  +0.11111"
# Aggregate-summary rows look like:
#   "  metric                       |   +0.12345 |   +0.67890 |   +0.11111 |   5.3%"
# We pull the metric label + three numeric columns; trailing "Δ%" cell is ignored.
_VALUE_LINE = re.compile(
    r"^\s*(?P<metric>\S[^|]*?)\s*\|"
    r"\s*(?P<g>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\|"
    r"\s*(?P<n>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\|"
    r"\s*(?P<m>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:\|.*)?$"
)


def _run_diag(out: Path, *, seed: int, steps: int, num_envs: int, settle: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        _DIAG_MODULE,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--num-envs",
        str(num_envs),
        "--settle",
        str(settle),
        "--out",
        str(out),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _parse_metrics(path: Path) -> list[tuple[int, str, float, float, float]]:
    """Return ``[(line_no, metric_label, genesis_val, newton_val, mujoco_val), ...]``
    for every numeric three-column row we can recognise in the trace file.
    Line number is kept so the same metric appearing in different STEP blocks
    gets a unique key (otherwise STEP 0's ``cost`` would collide with STEP 1's).
    """
    rows: list[tuple[int, str, float, float, float]] = []
    for line_no, line in enumerate(path.read_text().splitlines()):
        m = _VALUE_LINE.match(line)
        if not m:
            continue
        rows.append(
            (
                line_no,
                m.group("metric").strip(),
                float(m.group("g")),
                float(m.group("n")),
                float(m.group("m")),
            )
        )
    return rows


def _per_sim_drift(
    rows_a: list[tuple[int, str, float, float, float]],
    rows_b: list[tuple[int, str, float, float, float]],
) -> dict[str, dict]:
    """Compute absolute drift per sim across all aligned rows.

    If the two traces have different line counts we still align by the
    shared prefix and warn.
    """
    if len(rows_a) != len(rows_b):
        print(f"  WARNING: row count differs (run A={len(rows_a)}, run B={len(rows_b)}); " "aligning on shared prefix.")
    n = min(len(rows_a), len(rows_b))
    per_sim: dict[str, dict] = {
        s: {"max_abs": 0.0, "max_metric": "", "max_line": -1, "max_a": 0.0, "max_b": 0.0, "n_diff": 0} for s in _SIMS
    }
    for i in range(n):
        line_a, metric_a, ga, na, ma = rows_a[i]
        line_b, metric_b, gb, nb, mb = rows_b[i]
        if metric_a != metric_b:
            # Structure mismatch — skip rather than mis-align.
            continue
        for sim_name, va, vb in (("genesis", ga, gb), ("newton", na, nb), ("mujoco", ma, mb)):
            d = abs(va - vb)
            if d > 0:
                per_sim[sim_name]["n_diff"] += 1
            if d > per_sim[sim_name]["max_abs"]:
                per_sim[sim_name]["max_abs"] = d
                per_sim[sim_name]["max_metric"] = metric_a
                per_sim[sim_name]["max_line"] = line_a
                per_sim[sim_name]["max_a"] = va
                per_sim[sim_name]["max_b"] = vb
    per_sim["_n_rows"] = n
    return per_sim


def _verdict(sim: str, info: dict) -> str:
    n_diff = info["n_diff"]
    if n_diff == 0:
        return "DETERMINISTIC (all values match)"
    return (
        f"DRIFT ({n_diff} values differ; "
        f"max |Δ|={info['max_abs']:.6g} at line {info['max_line']} "
        f"[{info['max_metric']}]: {info['max_a']:.6g} vs {info['max_b']:.6g})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--settle", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="Keep intermediate trace files for inspection.")
    args = ap.parse_args()

    out_a = Path("determinism_run_a.txt")
    out_b = Path("determinism_run_b.txt")

    print(f"\n{'=' * 78}\nRun 1/2\n{'=' * 78}")
    _run_diag(out_a, seed=args.seed, steps=args.steps, num_envs=args.num_envs, settle=args.settle)
    print(f"\n{'=' * 78}\nRun 2/2\n{'=' * 78}")
    _run_diag(out_b, seed=args.seed, steps=args.steps, num_envs=args.num_envs, settle=args.settle)

    rows_a = _parse_metrics(out_a)
    rows_b = _parse_metrics(out_b)
    drift = _per_sim_drift(rows_a, rows_b)

    print(f"\n{'=' * 78}\nDETERMINISM VERDICT  (seed={args.seed}, parsed {drift['_n_rows']} numeric rows)\n{'=' * 78}")
    for sim in _SIMS:
        info = drift[sim]
        print(f"  [{sim:>7s}] {_verdict(sim, info)}")
    print("=" * 78)

    if not args.keep:
        out_a.unlink(missing_ok=True)
        out_b.unlink(missing_ok=True)
    else:
        print(f"\n  (kept trace files: {out_a}, {out_b})")

    # exit code: 0 if every sim was deterministic; 1 otherwise so callers can gate on it
    any_drift = any(drift[s]["n_diff"] > 0 for s in _SIMS)
    return 1 if any_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
