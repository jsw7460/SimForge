"""Prove, on a live simulator, which foot's contact gates which foot's slip.

The mujoco go2 ``feet_slip`` curve sat 30x below newton / genesis while
every policy walked. The cause is a column pairing, not physics: the
mujoco preset selects the feet as sites in ``(FR, FL, RR, RL)`` order and
the contact sensor tracks the foot geoms in model order ``(FL, FR, RL,
RR)``; the term multiplied column ``i`` of one by column ``i`` of the
other, so the swing foot's speed was gated by its lateral neighbour's
stance. ``common._feet_contact_order`` now matches contact columns to the
feet by name. This diag measures the pairing on a rolling robot so the
fix is judged against the physics rather than against the code.

ONE trained checkpoint is rolled closed-loop (one sim per process) and,
per control step, each foot's finite-difference xy speed is taken from
the reward's own selector and crossed with the contact state of

  * its OWN contact column, found by name (``FR`` -> ``FR_foot_collision``),
  * its LATERAL NEIGHBOUR's column (``FR`` <-> ``FL``, ``RR`` <-> ``RL``),
  * the column at the SAME INDEX in the group's natural order — what the
    term used to read.

Physics decides: a stance foot creeps (cm/s), a swing foot flies (m/s).
The own-contact gate must give the small number on every backend, the
neighbour gate the large one, and the term the env computes must equal
the own-contact sum. On mujoco the same-index gate equals the neighbour
gate (the bug, measured); on newton / genesis it equals the own gate.

Run once per sim, then compare::

    python -m jaxrlworld.scripts.diag.parity.go2_feet_slip_column_alignment_diag \\
        --sim mujoco --wandb_run_path <entity>/<project>/<run_id> \\
        --steps 500 --num_envs 64 --out parity_out/slip_align_mujoco.json
    (repeat with --sim newton, the newton run of the same preset)
    python -m jaxrlworld.scripts.diag.parity.go2_feet_slip_column_alignment_diag \\
        --compare parity_out/slip_align_mujoco.json parity_out/slip_align_newton.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jaxrlworld.rl.envs.mdp.rewards.common.reward_terms import _feet_contact_order, _foot_pos_vel

GROUP = "feet_ground_contact"
TERM = "feet_slip"
LEGS = ("FR", "FL", "RR", "RL")
NEIGHBOUR = {"FR": "FL", "FL": "FR", "RR": "RL", "RL": "RR"}


def leg_of(name: str) -> str:
    """The leg tag a foot site / body / geom name carries (exactly one)."""
    tags = [leg for leg in LEGS if leg in name]
    if len(tags) != 1:
        raise ValueError(f"Foot name {name!r} carries leg tags {tags}; expected exactly one of {LEGS}.")
    return tags[0]


def _stats(x: torch.Tensor) -> dict:
    if x.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "n": 0}
    q = torch.quantile(x.float(), torch.tensor([0.5, 0.9], device=x.device))
    return {"mean": float(x.mean()), "p50": float(q[0]), "p90": float(q[1]), "n": int(x.numel())}


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
    dt = env.control_dt

    term_cfg = env.reward_manager.reward_terms[TERM]
    asset_cfg = term_cfg.params["asset_cfg"]
    weight = term_cfg.weight
    foot_names = list(asset_cfg.site_names if asset_cfg.site_ids is not None else asset_cfg.body_names)
    foot_legs = [leg_of(n) for n in foot_names]
    tracked = list(env.contact_manager.tracked_names(GROUP))
    tracked_legs = [leg_of(n) for n in tracked]
    own_col = [tracked_legs.index(leg) for leg in foot_legs]
    neighbour_col = [tracked_legs.index(NEIGHBOUR[leg]) for leg in foot_legs]
    same_index_col = list(range(len(foot_legs)))
    matched_order = _feet_contact_order(env, GROUP, asset_cfg, None)
    print(f"feet (selector order):     {foot_names}")
    print(f"contact (natural order):   {tracked}")
    print(f"_feet_contact_order:       {matched_order}")
    print(f"own / neighbour / index columns per foot: {own_col} / {neighbour_col} / {same_index_col}")

    sums = {"own": 0.0, "neighbour": 0.0, "same_index": 0.0, "term": 0.0}
    per_leg_own: dict[str, list[torch.Tensor]] = {leg: [] for leg in LEGS}
    per_leg_neighbour: dict[str, list[torch.Tensor]] = {leg: [] for leg in LEGS}
    stance_frac = torch.zeros(len(foot_legs), device=env.device)
    n_steps = 0
    n_valid = 0

    obs, _ = env.reset()
    robot_states = env.get_robot_state()
    prev_pos, _ = _foot_pos_vel(env, asset_cfg)
    prev_pos = prev_pos.clone()

    for _ in range(args.steps):
        action = policy.get_action(obs, robot_states)
        obs, _, terminated, truncated, infos = env.step(action)
        robot_states = env.get_robot_state()
        done = terminated | truncated
        if done.any():
            policy.notify_reset(done.cpu().numpy())

        pos, _ = _foot_pos_vel(env, asset_cfg)
        vel_xy_sq = torch.sum(torch.square((pos - prev_pos)[..., :2] / dt), dim=-1)  # (E, F)
        prev_pos = pos.clone()
        contact = env.contact_manager.is_contact(GROUP).float()  # (E, N) natural order
        # Same mask as the term: a finite difference across a reset is not a speed.
        valid = env.episode_length_buf > 1
        cmd = env.command_manager
        active = (torch.norm(torch.stack([cmd.lin_vel_x, cmd.lin_vel_y], dim=1), dim=1) + cmd.ang_vel.abs()) > 0.05
        keep = valid & active
        if keep.any():
            v = vel_xy_sq[keep]
            own = torch.sum(v * contact[keep][:, own_col], dim=1)
            neighbour = torch.sum(v * contact[keep][:, neighbour_col], dim=1)
            same_index = torch.sum(v * contact[keep][:, same_index_col], dim=1)
            sums["own"] += float(own.sum())
            sums["neighbour"] += float(neighbour.sum())
            sums["same_index"] += float(same_index.sum())
            for i, leg in enumerate(foot_legs):
                in_own = contact[keep][:, own_col[i]] > 0.5
                in_neigh = contact[keep][:, neighbour_col[i]] > 0.5
                per_leg_own[leg].append(v[in_own, i].sqrt().cpu())
                per_leg_neighbour[leg].append(v[in_neigh, i].sqrt().cpu())
            stance_frac += contact[keep][:, own_col].mean(dim=0)
            n_valid += int(keep.sum())
            # The env's own term value, raw (pre weight*dt), over the SAME envs.
            raw = infos["rewards_per_type"][TERM] / (weight * dt)
            sums["term"] += float((-raw[keep]).sum())
        n_steps += 1

    means = {k: v / max(n_valid, 1) for k, v in sums.items()}
    out = {
        "sim": args.sim,
        "steps": n_steps,
        "num_envs": args.num_envs,
        "valid_env_steps": n_valid,
        "feet_selector_order": foot_names,
        "contact_natural_order": tracked,
        "feet_contact_order": matched_order,
        "same_index_pairs": {leg: tracked_legs[i] for i, leg in enumerate(foot_legs)},
        "slip_cost_mean": means,
        "stance_fraction_per_foot": dict(zip(foot_legs, (stance_frac / max(n_steps, 1)).tolist())),
        "stance_speed_own_gate": {leg: _stats(torch.cat(per_leg_own[leg])) for leg in foot_legs},
        "speed_under_neighbour_gate": {leg: _stats(torch.cat(per_leg_neighbour[leg])) for leg in foot_legs},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    report([out])
    print(f"\nwrote {args.out}")


def report(data: list[dict]) -> None:
    sims = [d["sim"] for d in data]
    print("\n== column pairing ==")
    for d in data:
        print(f"  {d['sim']:8s} feet {d['feet_selector_order']}")
        print(f"  {'':8s} contact natural order {d['contact_natural_order']}")
        print(f"  {'':8s} same-index pairs foot->contact {d['same_index_pairs']}")
        print(f"  {'':8s} _feet_contact_order {d['feet_contact_order']}")

    print("\n== slip cost, raw mean over active env-steps (sum_feet v_xy^2 * gate) ==")
    print(f"  {'gate':12s}" + "".join(f"{s:>14s}" for s in sims))
    for key in ("own", "neighbour", "same_index", "term"):
        print(f"  {key:12s}" + "".join(f"{d['slip_cost_mean'][key]:14.6f}" for d in data))
    print("  ratios:")
    for d in data:
        m = d["slip_cost_mean"]
        own = m["own"] if abs(m["own"]) > 1e-12 else float("nan")
        print(
            f"    {d['sim']:8s} neighbour/own {m['neighbour'] / own:8.2f}   "
            f"same_index/own {m['same_index'] / own:8.2f}   term/own {m['term'] / own:8.4f}"
        )

    print("\n== per-foot xy speed (m/s) while gated by its OWN contact vs its NEIGHBOUR's ==")
    for d in data:
        print(f"  {d['sim']}:")
        print(f"    {'foot':5s} {'stance%':>8s} {'own p50':>9s} {'own p90':>9s} {'neigh p50':>10s} {'neigh p90':>10s}")
        for leg, frac in d["stance_fraction_per_foot"].items():
            o = d["stance_speed_own_gate"][leg]
            n = d["speed_under_neighbour_gate"][leg]
            print(f"    {leg:5s} {100 * frac:7.1f}% {o['p50']:9.4f} {o['p90']:9.4f} {n['p50']:10.4f} {n['p90']:10.4f}")

    print(
        "\nreading guide: 'own' is the physical slip (stance foot creep, cm/s);\n"
        "'neighbour' is the swing foot's flight speed charged as slip. The term\n"
        "must track 'own' (term/own ~ 1) on every backend. same_index/own >> 1\n"
        "on a backend means its natural contact order is not the selector order,\n"
        "which is the pairing the term used before _feet_contact_order matched by name."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["genesis", "newton", "mujoco"])
    ap.add_argument("--policy_path", type=str, default=None)
    ap.add_argument("--wandb_run_path", type=str, default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    if args.compare:
        report([json.loads(Path(p).read_text()) for p in args.compare])
        return
    if args.sim is None or (args.policy_path is None) == (args.wandb_run_path is None):
        ap.error("dump mode needs --sim and exactly one of --policy_path / --wandb_run_path")
    if args.out is None:
        args.out = f"parity_out/slip_align_{args.sim}.json"
    dump(args)


if __name__ == "__main__":
    main()
