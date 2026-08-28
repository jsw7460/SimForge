"""Cross-sim evaluation of ONE trained K1 policy on all three simulators.

Purpose: decide whether late-training reward gaps (feet_slip /
feet_swing_height / feet_air_time diverging on genesis) come from the
framework/physics or from per-sim policy adaptation.

- Same reward numbers across sims for the SAME policy → the MDP is
  aligned; training-curve gaps are policies adapting to each sim.
- Same policy still showing the gap → a reproducible residual physics
  difference, with this exact rollout as the artifact to chase.

Each sim runs in its own subprocess (one CUDA context each) via
PolicyEvaluator with the checkpoint's preset auto-rebuilt for the
target sim (unified-config Strategy 2). Every env completes exactly
one episode; we report per-term mean episode returns side by side.

Run (server, from SimForge root):
    python -m rlworld.scripts.diag.k1.k1_cross_sim_policy_eval \\
        --wandb-run-path jsw7460/K1_Joystick/vzpzghww
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "rlworld.scripts.diag.k1.k1_cross_sim_policy_eval"
_SIMS = ("genesis", "newton", "mujoco")
_FOCUS_TERMS = ("feet_slip", "feet_swing_height", "feet_air_time")


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child: evaluate on one sim ──────────────────────────────────────


def run_cell(sim: str, wandb_run_path: str, num_envs: int, seed: int) -> dict:
    from rlworld.rl.evals import PolicyEvaluator

    _stage(f"cell start: {sim} num_envs={num_envs}")

    evaluator = PolicyEvaluator(
        wandb_run_path=wandb_run_path,
        eval_target=sim,
        num_evals=1,
        seed=seed,
        record_video=False,
        save_data=False,
        use_rich_display=False,
        extra_overrides={"env": {"num_envs": num_envs}},
    )
    _stage("evaluator ready")

    stats = evaluator.evaluate()
    _stage("evaluation done")

    per_type = {
        term: {"mean": round(d["mean"], 5), "std": round(d["std"], 5)} for term, d in stats["reward_breakdown"].items()
    }

    # ---- gait descriptors: same policy, fixed-length rollout ---------
    # Quantifies "the walk looks different": duty cycle, touchdown rate,
    # stance/air segment distributions, swing apex height, and the
    # per-STEP per-term reward means (the exact quantity wandb training
    # curves log). First 10 steps after each env reset are excluded
    # from gait stats; reward means include everything, like training.
    import torch

    from rlworld.rl.configs.robots.k1 import K1Config
    from rlworld.rl.configs.scene import SceneEntitySelector

    env = evaluator.env
    policy = evaluator.policy
    feet_ids = env.resolve_selector(
        SceneEntitySelector(name="robot", body_names=tuple(K1Config().foot_names), preserve_order=True)
    ).body_ids
    n_envs = env.num_envs
    ctrl_dt = env.control_dt
    steps, settle = 500, 10

    def sole_z() -> torch.Tensor:
        return env.get_robot_data().body_pos_w_by_ids(feet_ids)[..., 2] - 0.048

    def _stats(chunks: list) -> dict:
        if not chunks:
            return {"n": 0}
        t = torch.cat([x.flatten().float() for x in chunks])
        qs = torch.tensor([0.1, 0.5, 0.9], device=t.device)
        return {
            "n": int(t.numel()),
            "mean": round(float(t.mean()), 5),
            "q10_50_90": [round(float(v), 5) for v in torch.quantile(t, qs).tolist()],
        }

    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state()
    prev_c = env.contact_manager.is_contact("feet_ground_contact").clone()
    since_reset = torch.zeros(n_envs, device=env.device)
    stance_run = torch.zeros_like(prev_c, dtype=torch.float32)
    air_run = torch.zeros_like(stance_run)
    cur_apex = sole_z().clone()
    stance_lens: list = []
    air_lens: list = []
    apexes: list = []
    on_edges = 0
    contact_cnt = torch.zeros((), device=env.device)
    valid_cnt = torch.zeros((), device=env.device)
    term_sums: dict[str, float] = {}
    with torch.no_grad():
        for _ in range(steps):
            action = policy.get_action(obs, robot_states)
            obs, _rew, term, trunc, extras = env.step(action)
            robot_states = env.get_robot_state()
            reset_now = term | trunc
            if reset_now.any():
                policy.notify_reset(reset_now.cpu().numpy())
            for name, val in extras["rewards_per_type"].items():
                term_sums[name] = term_sums.get(name, 0.0) + float(val.mean())
            since_reset = torch.where(reset_now, torch.zeros_like(since_reset), since_reset + 1.0)
            valid = (since_reset > settle).unsqueeze(-1).expand_as(prev_c)
            c = env.contact_manager.is_contact("feet_ground_contact")
            fz = sole_z()
            rising = c & ~prev_c & valid
            falling = ~c & prev_c & valid
            if rising.any():
                air_lens.append(air_run[rising])
                apexes.append(cur_apex[rising])
                on_edges += int(rising.sum())
            if falling.any():
                stance_lens.append(stance_run[falling])
            # Swing-apex bookkeeping: restart tracking at liftoff, take
            # the running max while airborne, capture at touchdown.
            cur_apex = torch.where(falling, fz, cur_apex)
            cur_apex = torch.where(~c, torch.maximum(cur_apex, fz), cur_apex)
            stance_run = torch.where(c & valid, stance_run + 1.0, torch.zeros_like(stance_run))
            air_run = torch.where(~c & valid, air_run + 1.0, torch.zeros_like(air_run))
            contact_cnt += (c & valid).float().sum()
            valid_cnt += valid.float().sum()
            prev_c = c.clone()
    _stage("gait descriptor rollout done")

    gait = {
        "rollout_steps": steps,
        "duty_cycle": round(float(contact_cnt / valid_cnt.clamp(min=1.0)), 4),
        "touchdowns_per_foot_per_s": round(on_edges / max(float(valid_cnt) * ctrl_dt, 1e-9), 3),
        "stance_len_ctrl_steps": _stats(stance_lens),
        "air_len_ctrl_steps": _stats(air_lens),
        "swing_apex_m": _stats(apexes),
        "per_step_reward_means": {k: round(v / steps, 5) for k, v in sorted(term_sums.items())},
    }

    return {
        "sim": sim,
        "n_episodes": int(stats["num_episodes"]),
        "mean_return": round(float(stats["mean_return"]), 4),
        "std_return": round(float(stats["std_return"]), 4),
        "mean_episode_len": round(float(stats["mean_length"]), 1),
        "per_type": per_type,
        "gait": gait,
    }


# ── Parent ──────────────────────────────────────────────────────────


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
        print(f"[eval] running {sim} ...", flush=True)
        t0 = time.time()
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
                    "--wandb-run-path",
                    args.wandb_run_path,
                    "--num-envs",
                    str(args.num_envs),
                    "--seed",
                    str(args.seed),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                env={**os.environ},
            )
        ok = result_path.exists()
        print(f"[eval]   -> {'ok' if ok else 'CRASH (see log)'} ({time.time() - t0:.0f}s)", flush=True)
        if ok:
            results[sim] = json.loads(result_path.read_text())

    sims = [s for s in _SIMS if s in results]
    L: list[str] = []
    L.append("=" * 100)
    L.append(f"K1 cross-sim policy eval — one checkpoint on all sims ({args.wandb_run_path})")
    L.append("=" * 100)
    L.append(f"num_envs (episodes/sim): {args.num_envs}  seed: {args.seed}  logs: {log_dir}")
    L.append("")
    header = f"{'':34s}" + "".join(f"{s:>22s}" for s in sims)
    L.append(header)
    L.append("-" * len(header))
    L.append(
        f"{'mean_return':34s}"
        + "".join(f"{results[s]['mean_return']:>15.3f} ±{results[s]['std_return']:<5.1f}" for s in sims)
    )
    L.append(
        f"{'mean_episode_len [ctrl steps]':34s}" + "".join(f"{results[s]['mean_episode_len']:>22.1f}" for s in sims)
    )
    L.append("")

    all_terms = sorted({t for s in sims for t in results[s]["per_type"]})
    focus = [t for t in all_terms if any(f in t for f in _FOCUS_TERMS)]
    rest = [t for t in all_terms if t not in focus]
    L.append("-- FOCUS terms (per-episode return, mean ± std) --")
    for group in (focus, rest):
        for t in group:
            row = f"{t:34s}"
            for s in sims:
                d = results[s]["per_type"].get(t)
                row += f"{d['mean']:>15.4f} ±{d['std']:<5.1f}" if d else f"{'—':>22s}"
            L.append(row)
        if group is focus:
            L.append("")
            L.append("-- all other terms --")

    L.append("")
    L.append("-- GAIT DESCRIPTORS (same policy, 500-step rollout; 10 steps post-reset excluded) --")
    for s in sims:
        g = results[s].get("gait")
        if g is None:
            continue
        L.append(f"  [{s}] duty_cycle={g['duty_cycle']}  touchdowns/foot/s={g['touchdowns_per_foot_per_s']}")
        L.append(f"      stance len [ctrl steps]: {g['stance_len_ctrl_steps']}")
        L.append(f"      air len    [ctrl steps]: {g['air_len_ctrl_steps']}")
        L.append(f"      swing apex [m]: {g['swing_apex_m']}")
    L.append("")
    L.append("-- PER-STEP reward means (the wandb-curve quantity), same policy --")
    gait_terms = sorted({t for s in sims for t in results[s].get("gait", {}).get("per_step_reward_means", {})})
    for t in gait_terms:
        row = f"{t:34s}"
        for s in sims:
            v = results[s].get("gait", {}).get("per_step_reward_means", {}).get(t)
            row += f"{v:>22.5f}" if v is not None else f"{'—':>22s}"
        L.append(row)

    report = "\n".join(L)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"Report written to: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, choices=(None, *_SIMS))
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--wandb-run-path", required=True)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="k1_cross_sim_policy_eval.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(args.cell, args.wandb_run_path, args.num_envs, args.seed)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
