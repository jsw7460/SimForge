"""Cross-sim forensics for the go2 gait-conditioned (WTW) reward gap.

Training curves cannot separate "the sims measure the same behavior
differently" from "the policies learned different behavior". This diag
removes the policy variable: ONE trained checkpoint is rolled
CLOSED-LOOP in each simulator (one sim per process — the multi-sim
in-process contamination trap), while capturing, per control step:

  * every reward term's raw (pre weight*dt) mean — the exact quantities
    the wandb curves aggregate
  * the raw INPUTS behind the diverging terms, bucketed by the gait
    manager's desired contact state:
      - per-foot contact-force norms during swing (what
        tracking_contacts_shaped_force penalizes) and stance, with
        p50/p90/p99 (touchdown spikes are engine-specific)
      - per-foot velocity norms during stance in a SIM-NEUTRAL
        convention (finite difference of world foot positions) next to
        the term value itself (which embeds each sim's native read) —
        if the FD stats agree while the term values don't, the gap is
        the read convention (e.g. mjlab's substep-stale cvel); if both
        disagree, the gap is real dynamics
      - |dof_vel| mean, contact counts per sensor group
  * episode returns

Run once per sim, then compare:

    python -m jaxrlworld.scripts.diag.parity.go2_wtw_reward_forensics \
        --sim newton --wandb_run_path <entity>/<project>/<run_id> \
        --steps 1000 --num_envs 64 --out parity_out/wtw_newton.json
    (repeat with --sim genesis / mujoco, same checkpoint)
    python -m jaxrlworld.scripts.diag.parity.go2_wtw_reward_forensics \
        --compare parity_out/wtw_genesis.json parity_out/wtw_newton.json parity_out/wtw_mujoco.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _quantiles(x: torch.Tensor) -> dict:
    if x.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "n": 0}
    # Genesis switches torch's default device to CUDA; the q tensor must
    # live where the (cpu-collected) samples live.
    q = torch.quantile(x.float(), torch.tensor([0.5, 0.9, 0.99], device=x.device))
    return {
        "mean": float(x.mean()),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p99": float(q[2]),
        "n": int(x.numel()),
    }


def dump(args) -> None:
    from jaxrlworld.rl.evals import PolicyEvaluator

    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=args.sim,
        wandb_run_path=args.wandb_run_path,
        num_evals=1,
        record_video=False,
        extra_overrides={"env": {"num_envs": args.num_envs}},
    )
    env = evaluator.env
    policy = evaluator.policy
    rd = env.get_entity_data("robot")

    # Raw-term recovery: infos["rewards_per_type"] is weight*dt-scaled.
    weights = {name: cfg.weight for name, cfg in env.reward_manager.reward_terms.items()}
    dt = env.control_dt

    foot_names = list(env.gait_manager.foot_names)
    foot_ids = torch.tensor([rd.find_body_index(n) for n in foot_names], dtype=torch.long, device=env.device)

    term_sums: dict[str, float] = {}
    term_counts: dict[str, int] = {}
    force_swing, force_stance, vel_fd_stance = [], [], []
    dofvel_sum = 0.0
    contact_group_sums: dict[str, float] = {}
    n_steps = 0
    returns = []
    ep_ret = torch.zeros(env.num_envs, device=env.device)

    obs, _ = env.reset()
    robot_states = env.get_robot_state()
    prev_feet = rd.body_pos_w_by_ids(foot_ids).clone()

    for step in range(args.steps):
        action = policy.get_action(obs, robot_states)
        obs, rewards, terminated, truncated, infos = env.step(action)
        robot_states = env.get_robot_state()
        done = terminated | truncated
        ep_ret += rewards
        if done.any():
            returns.extend(ep_ret[done].tolist())
            ep_ret[done] = 0.0
            policy.notify_reset(done.cpu().numpy())

        for name, val in infos["rewards_per_type"].items():
            if name == "total_reward":
                continue
            w = weights.get(name, 0.0)
            raw = float(val.mean()) / (w * dt) if w != 0.0 else float(val.mean())
            term_sums[name] = term_sums.get(name, 0.0) + raw
            term_counts[name] = term_counts.get(name, 0) + 1

        desired = env.gait_manager.desired_contact_states  # (E, F)
        force = env.contact_manager.contact_force("feet_ground_contact")
        fnorm = force.norm(dim=-1)  # (E, F)
        feet = rd.body_pos_w_by_ids(foot_ids)
        vel_fd = ((feet - prev_feet) / dt).norm(dim=-1)  # (E, F), sim-neutral
        prev_feet = feet.clone()
        keep = ~done  # FD across a reset is meaningless
        swing = (desired < 0.5) & keep.unsqueeze(-1)
        stance = (desired > 0.5) & keep.unsqueeze(-1)
        force_swing.append(fnorm[swing].cpu())
        force_stance.append(fnorm[stance].cpu())
        vel_fd_stance.append(vel_fd[stance].cpu())

        dofvel_sum += float(rd.joint_vel.abs().mean())
        for group in env.contact_manager.group_names():
            f = env.contact_manager.contact_force(group).norm(dim=-1)
            contact_group_sums[group] = contact_group_sums.get(group, 0.0) + float((f > 1.0).float().sum(dim=-1).mean())
        n_steps += 1

    out = {
        "sim": args.sim,
        "steps": n_steps,
        "num_envs": args.num_envs,
        "terms_raw_mean": {k: term_sums[k] / term_counts[k] for k in sorted(term_sums)},
        "force_norm_swing": _quantiles(torch.cat(force_swing)),
        "force_norm_stance": _quantiles(torch.cat(force_stance)),
        "foot_vel_fd_stance": _quantiles(torch.cat(vel_fd_stance)),
        "dof_vel_abs_mean": dofvel_sum / n_steps,
        "contact_counts_over_1N": {k: v / n_steps for k, v in contact_group_sums.items()},
        "episode_return": {"mean": float(np.mean(returns)) if returns else None, "n": len(returns)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


def compare(paths: list[str]) -> None:
    data = [json.loads(Path(p).read_text()) for p in paths]
    ref = data[0]
    sims = [d["sim"] for d in data]
    print(f"reference: {sims[0]}  (ratios are sim/{sims[0]})\n")

    print("== raw reward terms (mean, pre weight*dt) ==")
    print(f"{'term':34s}" + "".join(f"{s:>14s}" for s in sims) + "".join(f"  {s}/{sims[0]:>6s}" for s in sims[1:]))
    for term in ref["terms_raw_mean"]:
        vals = [d["terms_raw_mean"].get(term) for d in data]
        line = f"{term:34s}" + "".join(f"{v:14.5f}" if v is not None else f"{'-':>14s}" for v in vals)
        for v in vals[1:]:
            r = v / vals[0] if v is not None and abs(vals[0]) > 1e-12 else float("nan")
            line += f"  {r:8.3f}"
        print(line)

    print("\n== inputs ==")
    for key in ("force_norm_swing", "force_norm_stance", "foot_vel_fd_stance"):
        print(f"  {key}:")
        for stat in ("mean", "p50", "p90", "p99"):
            vals = [d[key][stat] for d in data]
            ratios = "".join(f"  x{(v / vals[0]):.3f}" if abs(vals[0]) > 1e-12 else "  -" for v in vals[1:])
            print(f"    {stat:5s} " + "".join(f"{v:12.4f}" for v in vals) + ratios)
    vals = [d["dof_vel_abs_mean"] for d in data]
    print(
        "  dof_vel_abs_mean " + "".join(f"{v:12.4f}" for v in vals) + "".join(f"  x{v / vals[0]:.3f}" for v in vals[1:])
    )
    for group in ref.get("contact_counts_over_1N", {}):
        vals = [d["contact_counts_over_1N"].get(group) for d in data]
        if all(v is not None for v in vals):
            print(
                f"  contacts>1N [{group}] "
                + "".join(f"{v:10.3f}" for v in vals)
                + "".join(f"  x{v / vals[0]:.3f}" if vals[0] > 1e-9 else "  -" for v in vals[1:])
            )
    print("\n== episode return ==")
    for d in data:
        er = d["episode_return"]
        print(f"  {d['sim']:8s} mean {er['mean']}  (n={er['n']})")
    print(
        "\nreading guide: a shaped_force ratio driven by force_norm_swing p90/p99 is an\n"
        "engine contact-transient difference; a shaped_vel gap WITHOUT a matching\n"
        "foot_vel_fd_stance gap is a velocity READ-convention difference (e.g. mjlab's\n"
        "substep-stale cvel); matching FD gaps are real dynamics (friction/solver)."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["genesis", "newton", "mujoco"])
    ap.add_argument("--policy_path", type=str, default=None)
    ap.add_argument("--wandb_run_path", type=str, default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return
    if args.sim is None or (args.policy_path is None) == (args.wandb_run_path is None):
        ap.error("dump mode needs --sim and exactly one of --policy_path / --wandb_run_path")
    if args.out is None:
        args.out = f"parity_out/wtw_{args.sim}.json"
    dump(args)


if __name__ == "__main__":
    main()
