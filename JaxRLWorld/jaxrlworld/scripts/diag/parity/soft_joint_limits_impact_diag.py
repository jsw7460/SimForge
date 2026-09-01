"""soft_joint_pos_limits fix — pre-implementation impact dump, every robot.

The planned fix changes Genesis/Newton ``RobotData.soft_joint_pos_limits``
from ``hard * 0.9`` (hardcoded factor, value scaling — introduced with the
T1 getup task for fallen-pose sampling) to the mjlab/IsaacLab formula
``mid +- 0.5 * range * soft_joint_pos_limit_factor`` (factor from the
entity's ArticulationCfg). Before touching the framework, this diag dumps,
for EVERY public preset on EVERY simulator:

  * the ``soft_joint_pos_limit_factor`` each sim's built config carries
    (flags cross-sim mismatches — e.g. an asset-level override)
  * per joint: hard limits (Genesis/Newton), the CURRENTLY SERVED soft
    limits, and the PROPOSED soft limits computed in-diag with the new
    formula — with per-joint deltas and a marker for symmetric joints
    (where the two formulas coincide and nothing changes)
  * for mjlab: whether the proposed formula applied to the (Genesis-read)
    hard limits reproduces mjlab's served limits exactly — i.e. proof the
    fix converges all three sims onto the same numbers
  * the ACTIVE CONSUMERS of ``soft_joint_pos_limits`` in that preset's
    built config (reward terms, command terms, event terms are all listed
    verbatim, and known consumers are flagged), so the report states
    exactly which presets change behavior on which path:
      - mdp/commands/motion.py  (MotionCommand joint clamp — tracking)
      - mdp/events/common.py reset_robot_fallen_state with
        fall_joint_noise_range == "soft_limit"  (getup pose sampling)
      - mujoco-only joint_pos_limits reward (reads mjlab data directly —
        NOT affected by the fix)
  * post-reset joint state margins against CURRENT vs PROPOSED limits
    (how many envs/joints sit outside each — the immediate reward/
    sampling impact at episode start)

Each (preset, sim) cell runs in its own subprocess; the parent writes one
combined report.

Usage (GPU box):
    python -m jaxrlworld.scripts.diag.parity.soft_joint_limits_impact_diag
    python -m jaxrlworld.scripts.diag.parity.soft_joint_limits_impact_diag --presets t1_getup --sims genesis,mujoco
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "jaxrlworld.scripts.diag.parity.soft_joint_limits_impact_diag"

_PRESETS: dict[str, tuple[str, str]] = {
    "go2": ("jaxrlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "g1_29dof": ("jaxrlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("jaxrlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
    "t1_tracking": ("jaxrlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
    "t1_getup": ("jaxrlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
}
_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child ───────────────────────────────────────────────────────────


def run_cell(preset: str, sim: str, num_envs: int, seed: int) -> dict:
    import importlib

    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {preset}:{sim} (num_envs={num_envs})")

    module, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(module), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs, seed=seed).build()

    # ---- config-level dump: factor + every term, verbatim ----------
    art = cfgs.scene.entities["robot"].articulation
    factor = art.soft_joint_pos_limit_factor

    from jaxrlworld.rl.configs.base_config import iter_terms
    from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig
    from jaxrlworld.rl.configs.rewards import RewardTermConfig

    def fq(fn) -> str:
        return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"

    reward_terms = {name: fq(t.func) for name, t in iter_terms(cfgs.reward, RewardTermConfig).items()}
    event_terms = {
        name: {"func": fq(t.func), "mode": t.mode, "params": {k: repr(v) for k, v in (t.params or {}).items()}}
        for name, t in iter_terms(cfgs.event, EventTermConfig).items()
    }
    command_terms = {name: type(t).__name__ for name, t in (cfgs.command.terms or {}).items()}

    # Known consumers of rd.soft_joint_pos_limits.
    consumers: list[str] = []
    for name, t in event_terms.items():
        if "reset_robot_fallen_state" in t["func"] and t["params"].get("fall_joint_noise_range") == "'soft_limit'":
            consumers.append(f"event '{name}' (fallen-pose sampling over soft range)")
    for name, cls in command_terms.items():
        if "Motion" in cls:
            consumers.append(f"command '{name}' ({cls} joint clamp)")
    for name, f in reward_terms.items():
        if "dof_pos_limits" in name or "dof_pos_limits" in f or "joint_pos_limits" in f:
            note = " [mjlab-data-direct, NOT via rd]" if ".mujoco." in f else ""
            consumers.append(f"reward '{name}' ({f}){note}")

    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()
    rd = env.get_robot_data()

    joint_names = list(env.act_manager.actuated_joint_names)
    cur_lo, cur_hi = rd.soft_joint_pos_limits
    out = {
        "preset": preset,
        "sim": sim,
        "factor": factor,
        "joint_names": joint_names,
        "cur_soft_lo": cur_lo.tolist(),
        "cur_soft_hi": cur_hi.tolist(),
        "reward_terms": reward_terms,
        "event_terms": event_terms,
        "command_terms": command_terms,
        "consumers": consumers,
    }

    if sim in ("genesis", "newton"):
        hard_lo, hard_hi = rd.joint_pos_limits
        mid = (hard_lo + hard_hi) / 2.0
        half = (hard_hi - hard_lo) / 2.0
        prop_lo, prop_hi = mid - half * factor, mid + half * factor
        out["hard_lo"] = hard_lo.tolist()
        out["hard_hi"] = hard_hi.tolist()
        out["prop_soft_lo"] = prop_lo.tolist()
        out["prop_soft_hi"] = prop_hi.tolist()
    else:
        out["hard_lo"] = None
        out["hard_hi"] = None
        out["prop_soft_lo"] = None
        out["prop_soft_hi"] = None

    # Post-reset margins: how many envs sit outside CURRENT vs PROPOSED.
    q = rd.joint_pos

    def outside(lo_t, hi_t):
        lo = torch.tensor(lo_t, device=q.device) if not torch.is_tensor(lo_t) else lo_t
        hi = torch.tensor(hi_t, device=q.device) if not torch.is_tensor(hi_t) else hi_t
        viol = torch.clamp(lo - q, min=0.0) + torch.clamp(q - hi, min=0.0)
        return {
            "viol_sum_per_joint": viol.sum(dim=0).tolist(),
            "viol_envs_per_joint": (viol > 0).sum(dim=0).tolist(),
            "viol_total": float(viol.sum()),
        }

    out["reset_outside_current"] = outside(cur_lo, cur_hi)
    if out["prop_soft_lo"] is not None:
        out["reset_outside_proposed"] = outside(out["prop_soft_lo"], out["prop_soft_hi"])
    else:
        out["reset_outside_proposed"] = None
    _stage("cell done")
    return out


# ── Parent ──────────────────────────────────────────────────────────


def _fmt(cells, widths):
    return "".join(str(c)[: w - 1].ljust(w) for c, w in zip(cells, widths))


def run_parent(args) -> int:
    out_path = Path(args.out).resolve()
    log_dir = out_path.parent / (out_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    presets = [p.strip() for p in args.presets.split(",")] if args.presets else list(_PRESETS)
    sims = [s.strip() for s in args.sims.split(",")] if args.sims else list(_SIMS)

    results: dict[tuple[str, str], dict] = {}
    for preset in presets:
        for sim in sims:
            tag = f"{preset}_{sim}"
            log_path = log_dir / f"{tag}.log"
            result_path = log_dir / f"{tag}.json"
            if result_path.exists():
                result_path.unlink()
            print(f"[diag] running {preset}:{sim} ...", flush=True)
            t0 = time.perf_counter()
            with open(log_path, "w") as lf:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        _MODULE,
                        "--cell",
                        f"{preset}:{sim}",
                        "--result-json",
                        str(result_path),
                        "--num-envs",
                        str(args.num_envs),
                        "--seed",
                        str(args.seed),
                    ],
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
            wall = time.perf_counter() - t0
            if result_path.exists():
                results[(preset, sim)] = json.loads(result_path.read_text())
                print(f"[diag]   -> ok ({wall:.0f}s)", flush=True)
            else:
                print(f"[diag]   -> CRASH (see {log_path})", flush=True)

    L: list[str] = []
    L.append("=" * 120)
    L.append("soft_joint_pos_limits fix — impact dump across all public presets")
    L.append("=" * 120)
    L.append(f"num_envs: {args.num_envs}   seed: {args.seed}   cells: {len(results)}")
    L.append(f"cell logs/JSON: {log_dir}")
    L.append("")
    L.append("PROPOSED formula: soft = mid +- 0.5 * (hard range) * soft_joint_pos_limit_factor (per-preset cfg)")
    L.append("CURRENT genesis/newton: soft = hard * 0.9 (hardcoded)")
    L.append("")

    for preset in presets:
        cells = {s: results.get((preset, s)) for s in sims}
        have = [s for s in sims if cells[s]]
        if not have:
            continue
        ref = cells[have[0]]
        L.append("=" * 120)
        L.append(f"[{preset}]")
        factors = {s: cells[s]["factor"] for s in have}
        fvals = set(factors.values())
        L.append(f"  soft_joint_pos_limit_factor per sim: {factors}" + ("   **MISMATCH**" if len(fvals) > 1 else ""))
        L.append("  consumers of rd.soft_joint_pos_limits:")
        any_consumer = False
        for s in have:
            for c in cells[s]["consumers"]:
                L.append(f"    [{s}] {c}")
                any_consumer |= "NOT via rd" not in c
        if not any(cells[s]["consumers"] for s in have):
            L.append("    (none — the fix cannot change this preset's behavior)")
        L.append("")

        # Per-joint table: hard, current, proposed, delta; mjlab served for convergence check.
        gn = next((cells[s] for s in ("genesis", "newton") if cells.get(s)), None)
        mj = cells.get("mujoco")
        if gn:
            L.append(
                _fmt(
                    [
                        "joint",
                        "hard lo/hi",
                        "CURRENT lo/hi",
                        "PROPOSED lo/hi",
                        "delta lo/hi",
                        "sym",
                        "mjlab served lo/hi",
                        "mj==prop",
                    ],
                    [24, 20, 20, 20, 18, 5, 22, 9],
                )
            )
            for j, nm in enumerate(gn["joint_names"]):
                hlo, hhi = gn["hard_lo"][j], gn["hard_hi"][j]
                clo, chi = gn["cur_soft_lo"][j], gn["cur_soft_hi"][j]
                plo, phi = gn["prop_soft_lo"][j], gn["prop_soft_hi"][j]
                dlo, dhi = plo - clo, phi - chi
                sym = "Y" if abs(hlo + hhi) < 1e-9 else ""
                if mj and nm in mj["joint_names"]:
                    k = mj["joint_names"].index(nm)
                    mlo, mhi = mj["cur_soft_lo"][k], mj["cur_soft_hi"][k]
                    mj_cell = f"{mlo:.4f}/{mhi:.4f}"
                    match = "OK" if abs(mlo - plo) < 1e-4 and abs(mhi - phi) < 1e-4 else "**NO**"
                else:
                    mj_cell, match = "N/A", "-"
                L.append(
                    _fmt(
                        [
                            nm,
                            f"{hlo:.3f}/{hhi:.3f}",
                            f"{clo:.4f}/{chi:.4f}",
                            f"{plo:.4f}/{phi:.4f}",
                            f"{dlo:+.4f}/{dhi:+.4f}",
                            sym,
                            mj_cell,
                            match,
                        ],
                        [24, 20, 20, 20, 18, 5, 22, 9],
                    )
                )
            L.append("")

        # Reset-state margins under current vs proposed.
        for s in have:
            d = cells[s]
            cur = d["reset_outside_current"]
            prop = d["reset_outside_proposed"]
            line = f"  [{s}] reset-state outside CURRENT soft: total {cur['viol_total']:.4f}"
            if prop is not None:
                line += f"   -> outside PROPOSED: total {prop['viol_total']:.4f}"
            L.append(line)
            worst = sorted(((v, i) for i, v in enumerate(cur["viol_sum_per_joint"]) if v > 0), reverse=True)[:4]
            for v, i in worst:
                nm = d["joint_names"][i]
                extra = ""
                if prop is not None:
                    extra = f" -> proposed {prop['viol_sum_per_joint'][i]:.4f} ({prop['viol_envs_per_joint'][i]} envs)"
                L.append(f"      {nm}: current {v:.4f} ({cur['viol_envs_per_joint'][i]} envs){extra}")
        L.append("")

        # Full term dump (verbatim) for the record.
        L.append(f"  reward terms: {ref['reward_terms']}")
        L.append(f"  command terms: {ref['command_terms']}")
        L.append(f"  event terms: { {k: v['func'] + '@' + v['mode'] for k, v in ref['event_terms'].items()} }")
        L.append("")

    report = "\n".join(L)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, help="internal: 'preset:sim'")
    ap.add_argument("--result-json", default=None, help="internal: child result path")
    ap.add_argument("--presets", default=None)
    ap.add_argument("--sims", default=None)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="soft_joint_limits_impact_diag.txt")
    args = ap.parse_args()

    if args.cell is not None:
        preset, sim = args.cell.split(":")
        result = run_cell(preset, sim, args.num_envs, args.seed)
        Path(args.result_json).write_text(json.dumps(result))
        return 0

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
