"""K1 dof_pos_limits reward — full 3-sim divergence dump.

The k1_joystick preset's ``dof_pos_limits`` reward differs by more than
2x between mjlab and Genesis/Newton on the very first step. This diag
dumps EVERYTHING that flows into that term on every simulator so the
cause is pinned with certainty, into one combined report file:

  * the exact term wiring (function, weight, params) as built
  * actuated joint names in action-manager order, per sim
  * the soft limits the reward consumes (``rd.soft_joint_pos_limits``),
    per joint, per sim — plus hard limits where the backend exposes them
    (Genesis/Newton; mjlab stores only soft limits)
  * both candidate soft-limit derivations recomputed from the hard
    limits (``hard * factor`` — the Genesis/Newton robot_data formula —
    vs ``mid +- 0.5 * range * factor`` — the mjlab/IsaacLab formula) so
    the report shows which formula each sim's served limits match
  * default joint targets (``act_manager.offset``) and the post-reset
    joint state distribution per joint
  * every reward-manager CALL of the term, captured at the call site by
    shimming the module function BEFORE the config is built: the exact
    ``joint_pos`` / soft limits the manager's call consumed (so mjlab's
    stale-read convention is captured as trained, not as re-read), the
    per-joint lower/upper violation magnitudes, violating-env counts,
    and the returned per-env cost stats
  * a zero-action phase and a random-action phase, several steps each

The parent runs one subprocess per simulator and writes a combined
report with per-joint cross-sim tables and flags every quantity whose
cross-sim spread exceeds tolerance.

Usage (GPU box):
    python -m jaxrlworld.scripts.diag.k1.k1_dof_pos_limits_parity_diag
    python -m jaxrlworld.scripts.diag.k1.k1_dof_pos_limits_parity_diag --num-envs 2048
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "jaxrlworld.scripts.diag.k1.k1_dof_pos_limits_parity_diag"
_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_DEEP_CALLS = 12  # calls with full per-joint dumps (covers both phases' first steps)


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child ───────────────────────────────────────────────────────────


def run_cell(sim: str, num_envs: int, steps: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} (num_envs={num_envs}, steps={steps}/phase)")

    # Shim the reward function BEFORE the config is built so the
    # RewardTermConfig captures the shim and every manager call is
    # recorded exactly as training computes it (including any
    # stale-state convention of the backend).
    from jaxrlworld.rl.envs.mdp.rewards import k1_locomotion as k1_rf

    orig_fn = k1_rf.dof_pos_limits_soft
    calls: list[dict] = []

    def recorded(env, *a, **kw):
        rd = env.get_robot_data()
        soft_lo, soft_hi = rd.soft_joint_pos_limits
        q = rd.joint_pos
        low_viol = torch.clamp(soft_lo - q, min=0.0)
        high_viol = torch.clamp(q - soft_hi, min=0.0)
        out = orig_fn(env, *a, **kw)
        rec = {
            "call": len(calls),
            "value_mean": float(out.mean()),
            "value_min": float(out.min()),
            "value_max": float(out.max()),
            "violating_envs": int(((low_viol + high_viol).sum(dim=1) > 0).sum()),
        }
        if len(calls) < _DEEP_CALLS:
            rec.update(
                {
                    "soft_lo": soft_lo.tolist(),
                    "soft_hi": soft_hi.tolist(),
                    "q_mean": q.mean(dim=0).tolist(),
                    "q_min": q.min(dim=0).values.tolist(),
                    "q_max": q.max(dim=0).values.tolist(),
                    "low_viol_sum": low_viol.sum(dim=0).tolist(),
                    "high_viol_sum": high_viol.sum(dim=0).tolist(),
                    "low_viol_envs": (low_viol > 0).sum(dim=0).tolist(),
                    "high_viol_envs": (high_viol > 0).sum(dim=0).tolist(),
                }
            )
        calls.append(rec)
        return out

    k1_rf.dof_pos_limits_soft = recorded

    from jaxrlworld.rl.configs.presets.k1_joystick.base import K1JoystickConfig

    cfgs = K1JoystickConfig(sim_type=sim, num_envs=num_envs, seed=seed).build()

    # Term wiring as built (must show the shim + the weight actually used).
    term_cfg = cfgs.reward.dof_pos_limits
    term_dump = {
        "func": f"{term_cfg.func.__module__}.{term_cfg.func.__qualname__}",
        "is_recorded_shim": term_cfg.func is recorded,
        "weight": term_cfg.weight,
        "params": {k: repr(v) for k, v in (term_cfg.params or {}).items()},
    }

    # Robot articulation config: soft factor + actuator cfgs verbatim.
    robot_entity_cfg = cfgs.scene.entities["robot"]
    art = robot_entity_cfg.articulation
    articulation_dump = {
        "soft_joint_pos_limit_factor": getattr(art, "soft_joint_pos_limit_factor", "<no such field>"),
        "articulation_repr": repr(art),
    }

    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()
    _stage("env built + reset")

    rd = env.get_robot_data()
    joint_names = list(env.act_manager.actuated_joint_names)
    soft_lo, soft_hi = rd.soft_joint_pos_limits
    static = {
        "joint_names": joint_names,
        "soft_lo": soft_lo.tolist(),
        "soft_hi": soft_hi.tolist(),
        "offset_default_target": env.act_manager.offset[0].tolist(),
        "decimation": env.decimation,
        "physics_dt": env.physics_dt,
    }
    # Hard limits where the backend exposes them (mjlab: soft only).
    if sim in ("genesis", "newton"):
        hard_lo, hard_hi = rd.joint_pos_limits
        static["hard_lo"] = hard_lo.tolist()
        static["hard_hi"] = hard_hi.tolist()
    else:
        static["hard_lo"] = None
        static["hard_hi"] = None

    # Post-reset joint state distribution (before any step).
    q0 = rd.joint_pos
    static["reset_q_mean"] = q0.mean(dim=0).tolist()
    static["reset_q_min"] = q0.min(dim=0).values.tolist()
    static["reset_q_max"] = q0.max(dim=0).values.tolist()

    actions = torch.zeros((num_envs, env.num_actions), device=env.device)
    phase_marks = {}
    phase_marks["zero_action_calls_from"] = len(calls)
    for _ in range(steps):
        env.step(actions)
    phase_marks["random_action_calls_from"] = len(calls)
    for _ in range(steps):
        actions.uniform_(-1.0, 1.0)
        env.step(actions)
    phase_marks["calls_total"] = len(calls)

    k1_rf.dof_pos_limits_soft = orig_fn
    _stage(f"cell done: {len(calls)} recorded reward calls")
    return {
        "sim": sim,
        "num_envs": num_envs,
        "term": term_dump,
        "articulation": articulation_dump,
        "static": static,
        "phases": phase_marks,
        "calls": calls,
    }


# ── Parent ──────────────────────────────────────────────────────────


def _fmt_row(cells, widths):
    return "".join(str(c)[: w - 1].ljust(w) for c, w in zip(cells, widths))


def run_parent(args) -> int:
    out_path = Path(args.out).resolve()
    log_dir = out_path.parent / (out_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for sim in _SIMS:
        log_path = log_dir / f"{sim}.log"
        result_path = log_dir / f"{sim}.json"
        if result_path.exists():
            result_path.unlink()
        print(f"[diag] running {sim} ...", flush=True)
        t0 = time.perf_counter()
        with open(log_path, "w") as lf:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    "--cell",
                    sim,
                    "--result-json",
                    str(result_path),
                    "--num-envs",
                    str(args.num_envs),
                    "--steps",
                    str(args.steps),
                    "--seed",
                    str(args.seed),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        wall = time.perf_counter() - t0
        if result_path.exists():
            results[sim] = json.loads(result_path.read_text())
            print(f"[diag]   -> ok ({wall:.0f}s)", flush=True)
        else:
            print(f"[diag]   -> CRASH (see {log_path})", flush=True)

    L: list[str] = []
    L.append("=" * 120)
    L.append("K1 dof_pos_limits reward — 3-sim divergence dump")
    L.append("=" * 120)
    L.append(f"num_envs: {args.num_envs}   steps/phase: {args.steps}   seed: {args.seed}")
    L.append(f"cell logs/JSON: {log_dir}")
    L.append("")

    # ── per-sim wiring/static dumps ─────────────────────────────────
    for sim, d in results.items():
        L.append(f"── [{sim}] term wiring ─────────────────────────────────────────")
        L.append(f"  func: {d['term']['func']}   shim_active: {d['term']['is_recorded_shim']}")
        L.append(f"  weight: {d['term']['weight']}   params: {d['term']['params']}")
        L.append(f"  soft_joint_pos_limit_factor (cfg): {d['articulation']['soft_joint_pos_limit_factor']}")
        L.append(f"  decimation: {d['static']['decimation']}   physics_dt: {d['static']['physics_dt']}")
        L.append(f"  articulation cfg: {d['articulation']['articulation_repr']}")
        L.append("")

    sims = [s for s in _SIMS if s in results]
    if not sims:
        report = "\n".join(L)
        out_path.write_text(report + "\n")
        print(report)
        return 1
    ref = sims[0]
    names_by_sim = {s: results[s]["static"]["joint_names"] for s in sims}

    # ── joint name/order comparison ────────────────────────────────
    L.append("── actuated joint order (index: per-sim name; mismatches flagged) ──")
    n_joints = max(len(v) for v in names_by_sim.values())
    widths = [6] + [30] * len(sims) + [10]
    L.append(_fmt_row(["idx"] + sims + ["match"], widths))
    order_mismatch = False
    for i in range(n_joints):
        row = [str(i)]
        vals = []
        for s in sims:
            nm = names_by_sim[s][i] if i < len(names_by_sim[s]) else "<missing>"
            row.append(nm)
            vals.append(nm)
        same = len(set(vals)) == 1
        order_mismatch |= not same
        row.append("OK" if same else "**DIFF**")
        L.append(_fmt_row(row, widths))
    L.append("")

    def by_name(sim: str, key: str) -> dict[str, float]:
        vals = results[sim]["static"][key]
        if vals is None:
            return {}
        return dict(zip(names_by_sim[sim], vals))

    # ── soft/hard limit cross tables + formula attribution ─────────
    def cross_table(title: str, key: str, extra_cols=None):
        L.append(f"── {title} ──")
        maps = {s: by_name(s, key) for s in sims}
        cols = ["joint"] + sims + ["max|diff|", "flag"] + (list(extra_cols.keys()) if extra_cols else [])
        w = [26] + [14] * (len(cols) - 1)
        L.append(_fmt_row(cols, w))
        for nm in names_by_sim[ref]:
            vals = [maps[s].get(nm) for s in sims]
            present = [v for v in vals if v is not None]
            spread = max(present) - min(present) if len(present) > 1 else 0.0
            row = [nm] + [f"{v:.6f}" if v is not None else "N/A" for v in vals]
            row.append(f"{spread:.6f}")
            row.append("**DIFF**" if spread > 1e-4 else "")
            if extra_cols:
                for fn in extra_cols.values():
                    row.append(fn(nm))
            L.append(_fmt_row(row, w))
        L.append("")

    cross_table("soft limit LOWER (what the reward consumes)", "soft_lo")
    cross_table("soft limit UPPER (what the reward consumes)", "soft_hi")
    cross_table("hard limit LOWER (backend-exposed; mjlab N/A)", "hard_lo")
    cross_table("hard limit UPPER (backend-exposed; mjlab N/A)", "hard_hi")

    # Formula attribution from a sim that exposes hard limits.
    hard_sim = next((s for s in sims if results[s]["static"]["hard_lo"] is not None), None)
    if hard_sim is not None:
        L.append(
            "── soft-limit FORMULA attribution (from hard limits of " f"{hard_sim}; factor candidates 0.9 and 0.95) ──"
        )
        hl, hh = by_name(hard_sim, "hard_lo"), by_name(hard_sim, "hard_hi")
        cols = ["joint", "hard*0.9 lo/hi", "mid+-r*0.95 lo/hi"] + [f"{s} lo/hi" for s in sims]
        w = [26, 26, 26] + [26] * len(sims)
        L.append(_fmt_row(cols, w))
        for nm in names_by_sim[ref]:
            lo, hi = hl[nm], hh[nm]
            scale = (lo * 0.9, hi * 0.9)
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            midf = (mid - half * 0.95, mid + half * 0.95)
            row = [nm, f"{scale[0]:.4f}/{scale[1]:.4f}", f"{midf[0]:.4f}/{midf[1]:.4f}"]
            for s in sims:
                slo, shi = by_name(s, "soft_lo").get(nm), by_name(s, "soft_hi").get(nm)
                row.append(f"{slo:.4f}/{shi:.4f}")
            L.append(_fmt_row(row, w))
        L.append("")

    cross_table("default action target (act_manager.offset)", "offset_default_target")
    cross_table("post-reset joint_pos MEAN across envs", "reset_q_mean")
    cross_table("post-reset joint_pos MIN across envs", "reset_q_min")
    cross_table("post-reset joint_pos MAX across envs", "reset_q_max")

    # ── recorded reward calls ──────────────────────────────────────
    L.append("── recorded reward-manager calls (value mean/min/max, violating envs) ──")
    w = [8] + [34] * len(sims)
    L.append(_fmt_row(["call"] + sims, w))
    max_calls = max(len(results[s]["calls"]) for s in sims)
    for i in range(max_calls):
        row = [str(i)]
        for s in sims:
            cs = results[s]["calls"]
            if i < len(cs):
                c = cs[i]
                row.append(f"{c['value_mean']:.5f} [{c['value_min']:.4f},{c['value_max']:.4f}] v={c['violating_envs']}")
            else:
                row.append("-")
        L.append(_fmt_row(row, w))
    for s in sims:
        L.append(f"  {s} phases: {results[s]['phases']}")
    L.append("")

    # ── first deep call: per-joint violation breakdown ─────────────
    L.append("── FIRST reward call — per-joint violation breakdown (sum over envs; low|high, envs low|high) ──")
    cols = ["joint"] + sims
    w = [26] + [40] * len(sims)
    L.append(_fmt_row(cols, w))
    for j, nm in enumerate(names_by_sim[ref]):
        row = [nm]
        for s in sims:
            c = results[s]["calls"][0] if results[s]["calls"] else None
            if c and "low_viol_sum" in c and j < len(c["low_viol_sum"]):
                row.append(
                    f"{c['low_viol_sum'][j]:.4f}|{c['high_viol_sum'][j]:.4f} "
                    f"(e {c['low_viol_envs'][j]}|{c['high_viol_envs'][j]})"
                )
            else:
                row.append("-")
        L.append(_fmt_row(row, w))
    L.append("")

    # ── automated flags ────────────────────────────────────────────
    L.append("── FLAGS (anything with cross-sim spread) ──")
    flags = []
    if order_mismatch:
        flags.append("joint ORDER differs between sims — all per-index comparisons suspect")
    for key, label in (("soft_lo", "soft LOWER limit"), ("soft_hi", "soft UPPER limit")):
        maps = {s: by_name(s, key) for s in sims}
        for nm in names_by_sim[ref]:
            vals = [maps[s].get(nm) for s in sims if maps[s].get(nm) is not None]
            if len(vals) > 1 and max(vals) - min(vals) > 1e-4:
                flags.append(f"{label} '{nm}': " + " ".join(f"{s}={maps[s].get(nm):.6f}" for s in sims))
    for nm_idx, nm in enumerate(names_by_sim[ref]):
        vals = []
        for s in sims:
            c = results[s]["calls"][0] if results[s]["calls"] else None
            if c and "low_viol_sum" in c:
                vals.append(c["low_viol_sum"][nm_idx] + c["high_viol_sum"][nm_idx])
        if len(vals) > 1 and max(vals) - min(vals) > 1e-3 * args.num_envs:
            flags.append(f"first-call violation '{nm}' spread: {[f"{v:.4f}" for v in vals]}")
    if not flags:
        flags.append("(none above tolerance)")
    for f in flags:
        L.append(f"  !! {f}")

    report = "\n".join(L)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, help="internal: run one sim")
    ap.add_argument("--result-json", default=None, help="internal: child result path")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="k1_dof_pos_limits_parity_diag.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(args.cell, args.num_envs, args.steps, args.seed)
        Path(args.result_json).write_text(json.dumps(result))
        return 0

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
