"""Why does the trained K1 walk with a low, mincing (총총) gait instead of a
bold, high-clearance stride like G1 — under (near-)identical rewards?

Loads ONE trained checkpoint, drives it at a FIXED forward command, and dumps
the numbers that discriminate the three hypotheses:

  H1  the policy never even attempts to lift → feet_clearance / feet_swing_height
      contributions ~0 AND swing apex barely above the sole.
  H2  it lifts a bit but the clearance reward is outcompeted by the action-rate /
      effort penalty → apex low WHILE raw_action_rate_l2 is a large negative
      contribution.
  H3  it lifts fine but steps fast & short (pure cadence/morphology) → apex near
      the target BUT touchdown rate high / step length short.

Measured over a fixed-command rollout (obs noise + push events off via eval
defaults):
  - achieved forward speed vs commanded (is it even tracking?)
  - swing APEX of the foot-link origin vs the feet reward target (0.15) — the
    reward measures the foot origin, which sits ~0.038 m above the sole, so the
    sole clearance is apex-0.038.
  - cadence: touchdowns/foot/s, stance & air durations, duty cycle, step length.
  - per-term reward contributions (the wandb-curve quantity), ranked — so you can
    SEE whether feet_clearance/feet_swing_height are winning or being cancelled.

Mac can't run this (needs the sim + torch). Run on the training box, from the
SimForge root::

    python -m rlworld.scripts.diag.k1.k1_gait_diagnosis_diag \\
        --policy_path outputs/models/2026-08-03/21-23-47/checkpoint_latest --vx 0.5

``--sim`` defaults to the checkpoint's training sim; pass genesis/newton/mujoco
to force a cross-sim rollout. ``--vx`` is the forward command (m/s).
"""

from __future__ import annotations

import argparse

# feet reward target (feet_clearance / feet_swing_height target_height in the
# K1 g1_recipe) and the foot-origin-to-sole offset from that recipe's docstring.
_TARGET_H = 0.15
_SOLE_OFFSET = 0.038
_FOCUS = ("feet_clearance", "feet_swing_height", "feet_air_time", "raw_action_rate", "feet_slip")


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


def run(
    policy_path: str | None,
    wandb_run_path: str | None,
    sim: str | None,
    foot_bodies: tuple[str, ...],
    target_h: float,
    sole_offset: float,
    vx: float,
    num_envs: int,
    steps: int,
    settle: int,
    seed: int,
) -> int:
    import torch

    from rlworld.rl.configs.scene import SceneEntitySelector
    from rlworld.rl.evals import PolicyEvaluator
    from rlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz, yaw_quat_wxyz

    src = policy_path or wandb_run_path
    _stage(f"loading checkpoint: {src}  (sim={sim or 'training-sim'})")
    evaluator = PolicyEvaluator(
        policy_path=policy_path,
        wandb_run_path=wandb_run_path,
        eval_target=sim,  # None → training sim
        num_evals=1,
        seed=seed,
        record_video=False,
        save_data=False,
        use_rich_display=False,
        extra_overrides={"env": {"num_envs": num_envs}},
    )
    env = evaluator.env
    policy = evaluator.policy
    dev = env.device
    ctrl_dt = env.control_dt
    _stage(f"env ready: sim={evaluator.sim_type} num_envs={env.num_envs} control_dt={ctrl_dt*1000:.1f} ms")

    feet_ids = env.resolve_selector(
        SceneEntitySelector(name="robot", body_names=tuple(foot_bodies), preserve_order=True)
    ).body_ids
    all_ids = torch.arange(env.num_envs, device=dev)
    vel_term = env.command_manager.get_term("velocity")
    cmd = torch.zeros(env.num_envs, 3, device=dev)
    cmd[:, 0] = vx  # fixed forward command; set_command locks it against resampling

    def foot_origin_z() -> torch.Tensor:
        return env.get_robot_data().body_pos_w_by_ids(feet_ids)[..., 2]  # (N, 2)

    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state()
    prev_c = env.contact_manager.is_contact("feet_ground_contact").clone()
    since_reset = torch.zeros(env.num_envs, device=dev)
    stance_run = torch.zeros_like(prev_c, dtype=torch.float32)
    air_run = torch.zeros_like(stance_run)
    cur_apex = foot_origin_z().clone()
    stance_lens: list = []
    air_lens: list = []
    apexes: list = []
    touchdowns = 0
    contact_cnt = torch.zeros((), device=dev)
    valid_cnt = torch.zeros((), device=dev)
    fwd_speed_sum = torch.zeros((), device=dev)
    lat_speed_sum = torch.zeros((), device=dev)
    fwd_n = torch.zeros((), device=dev)
    term_sums: dict[str, float] = {}

    def heading_frame_vel() -> torch.Tensor:
        """Base velocity in the yaw-aligned (heading) frame — the SAME frame the
        velocity command is tracked in. [:,0]=forward, [:,1]=lateral. Averaging
        world-x instead cancels across envs with different yaw (the earlier bug)."""
        rd = env.get_robot_data()
        return quat_rotate_inverse_wxyz(yaw_quat_wxyz(rd.root_link_quat_w), rd.root_link_lin_vel_w)

    _stage(f"rollout: {steps} steps at vx={vx} m/s")
    with torch.no_grad():
        for _ in range(steps):
            vel_term.set_command(all_ids, cmd)  # hold fixed command (also re-locks post-reset)
            obs = env.obs_manager.get_observation()  # so the policy sees the fixed command
            action = policy.get_action(obs, robot_states)
            obs, _rew, term, trunc, extras = env.step(action)
            robot_states = env.get_robot_state()
            reset_now = term | trunc
            if reset_now.any():
                policy.notify_reset(reset_now.cpu().numpy())

            for name, val in extras["rewards_per_type"].items():
                term_sums[name] = term_sums.get(name, 0.0) + float(val.mean())

            since_reset = torch.where(reset_now, torch.zeros_like(since_reset), since_reset + 1.0)
            valid_row = since_reset > settle
            valid = valid_row.unsqueeze(-1).expand_as(prev_c)

            c = env.contact_manager.is_contact("feet_ground_contact")
            fz = foot_origin_z()
            rising = c & ~prev_c & valid  # touchdown
            falling = ~c & prev_c & valid  # liftoff
            if rising.any():
                air_lens.append(air_run[rising])
                apexes.append(cur_apex[rising])
                touchdowns += int(rising.sum())
            if falling.any():
                stance_lens.append(stance_run[falling])
            cur_apex = torch.where(falling, fz, cur_apex)
            cur_apex = torch.where(~c, torch.maximum(cur_apex, fz), cur_apex)
            stance_run = torch.where(c & valid, stance_run + 1.0, torch.zeros_like(stance_run))
            air_run = torch.where(~c & valid, air_run + 1.0, torch.zeros_like(air_run))
            contact_cnt += (c & valid).float().sum()
            valid_cnt += valid.float().sum()

            vh = heading_frame_vel()  # yaw-frame: [:,0]=forward, [:,1]=lateral
            fwd_speed_sum += vh[valid_row, 0].sum()
            lat_speed_sum += vh[valid_row, 1].abs().sum()
            fwd_n += valid_row.float().sum()
            prev_c = c.clone()
    _stage("rollout done")

    def q(chunks: list) -> dict:
        if not chunks:
            return {"n": 0}
        t = torch.cat([x.flatten().float() for x in chunks])
        qs = torch.tensor([0.1, 0.5, 0.9], device=t.device)
        return {
            "n": int(t.numel()),
            "mean": float(t.mean()),
            "q10_50_90": [float(v) for v in torch.quantile(t, qs).tolist()],
        }

    apex = q(apexes)
    stance = q(stance_lens)
    air = q(air_lens)
    valid_seconds = max(float(valid_cnt) * ctrl_dt, 1e-9)
    td_per_foot_s = touchdowns / (2.0 * valid_seconds) if touchdowns else 0.0  # 2 feet
    fwd_speed = float(fwd_speed_sum / fwd_n.clamp(min=1.0))
    lat_speed = float(lat_speed_sum / fwd_n.clamp(min=1.0))
    step_len = fwd_speed / (2.0 * td_per_foot_s) if td_per_foot_s > 1e-6 else float("nan")
    per_step_rew = {k: v / steps for k, v in term_sums.items()}

    # ── report ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(
        f"K1 GAIT DIAGNOSIS — {evaluator.sim_type} | cmd vx={vx} m/s | {steps} steps " f"({settle} post-reset excluded)"
    )
    print("=" * 78)

    print(
        f"\n[tracking] commanded vx={vx:.2f}  achieved fwd (heading frame)={fwd_speed:.3f} m/s  ({100.0 * fwd_speed / vx if vx else 0.0:.0f}% of command)"
        f"  | |lateral|={lat_speed:.3f} m/s"
    )

    print(f"\n[foot lift] swing APEX of foot-origin z (reward target = {target_h:.3f}):")
    if apex.get("n"):
        am, a10, a50, a90 = apex["mean"], *apex["q10_50_90"]
        print(f"   apex  mean={am:.3f}  q10/50/90={a10:.3f}/{a50:.3f}/{a90:.3f}  (n={apex['n']} touchdowns)")
        print(f"   => sole clearance apex (apex-{sole_offset}) mean={am-sole_offset:.3f} m")
        print(f"   => apex reaches {100.0*am/target_h:.0f}% of the {target_h} target")
    else:
        print("   (no swing detected — feet never left the ground)")

    print(
        f"\n[cadence] touchdowns/foot/s={td_per_foot_s:.2f}  duty_cycle={float(contact_cnt / valid_cnt.clamp(min=1.0)):.3f}  step_len≈{step_len:.3f} m"
    )
    print(
        f"   stance [ctrl steps] mean={stance.get('mean', float('nan')):.2f} "
        f"({stance.get('mean', 0)*ctrl_dt*1000:.0f} ms)  air mean={air.get('mean', float('nan')):.2f} "
        f"({air.get('mean', 0)*ctrl_dt*1000:.0f} ms)"
    )

    print("\n[per-term reward contributions] (per-step mean, ranked by |value|):")
    for name, v in sorted(per_step_rew.items(), key=lambda kv: -abs(kv[1])):
        star = "  <<" if any(f in name for f in _FOCUS) else ""
        print(f"   {name:32s} {v:+.5f}{star}")

    # ── interpretation ────────────────────────────────────────────────
    print("\n[interpretation]")
    clear = abs(per_step_rew.get("feet_clearance", 0.0)) + abs(per_step_rew.get("feet_swing_height", 0.0))
    arate = min((v for k, v in per_step_rew.items() if "action_rate" in k), default=0.0)
    dominant = max(
        (abs(per_step_rew.get(k, 0.0)) for k in ("track_lin_vel", "variable_posture", "flat_orientation")), default=1e-9
    )
    apex_mean = apex.get("mean", 0.0)
    sole_clear = apex_mean - sole_offset
    track_pct = 100.0 * fwd_speed / vx if vx else 0.0

    if track_pct < 30.0:
        print(f"   Forward tracking only {track_pct:.0f}% of command — is this a measurement issue")
        print("   (heading frame) or genuinely not walking? Confirm in the viewer before reading gait.")
    else:
        print(f"   Walks OK: tracks {track_pct:.0f}% of the {vx} m/s command (heading frame).")
        if sole_clear < 0.06:
            print(
                f"   Foot lift LOW: sole clearance apex {sole_clear*100:.1f} cm "
                f"({100.0*apex_mean/target_h:.0f}% of the {target_h} target)."
            )
            if clear < 0.2 * dominant:
                print(
                    f"   → CAUSE: feet_clearance+swing contribution ({clear:.4f}) is ~{dominant/max(clear,1e-9):.0f}x"
                )
                print(f"     smaller than the dominant terms ({dominant:.4f}) — almost no incentive to lift.")
                print("     LEVER: raise feet_swing_height / feet_clearance weight (currently 0.25 / 2.0).")
            if arate < -0.5 * clear:
                print(
                    f"   → raw_action_rate penalty ({arate:.4f}) also competes with the (tiny) lift reward"
                    " — consider lowering it."
                )
        else:
            print(f"   Foot lift OK ({sole_clear*100:.1f} cm sole): the 'mincing' is then cadence/morphology")
            print(f"     (touchdowns/foot/s={td_per_foot_s:.2f}, step_len={step_len:.3f} m), not a lift problem.")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--policy_path", default=None, help="Local checkpoint dir, e.g. outputs/models/.../checkpoint_latest"
    )
    ap.add_argument("--wandb_run_path", default=None, help="e.g. jsw7460/G1_29Dof/5qsj5crv (instead of --policy_path)")
    ap.add_argument("--sim", choices=("genesis", "newton", "mujoco"), default=None, help="Default: training sim.")
    # Robot-specific knobs. Defaults = K1 g1_recipe. For G1: --foot_bodies
    # left_foot_frame,right_foot_frame --target 0.1 --sole_offset 0.0
    ap.add_argument(
        "--foot_bodies",
        default="left_foot_link,right_foot_link",
        help="Comma-separated feet-reward body names (the apex is read at these).",
    )
    ap.add_argument("--target", type=float, default=_TARGET_H, help="feet_clearance/swing target_height [m].")
    ap.add_argument("--sole_offset", type=float, default=_SOLE_OFFSET, help="Foot-origin-above-sole offset [m].")
    ap.add_argument("--vx", type=float, default=0.5, help="Fixed forward command [m/s].")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--settle", type=int, default=10, help="Post-reset control steps excluded from gait stats.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if bool(args.policy_path) == bool(args.wandb_run_path):
        ap.error("provide exactly one of --policy_path or --wandb_run_path")
    foot_bodies = tuple(s.strip() for s in args.foot_bodies.split(",") if s.strip())
    return run(
        args.policy_path,
        args.wandb_run_path,
        args.sim,
        foot_bodies,
        args.target,
        args.sole_offset,
        args.vx,
        args.num_envs,
        args.steps,
        args.settle,
        args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
