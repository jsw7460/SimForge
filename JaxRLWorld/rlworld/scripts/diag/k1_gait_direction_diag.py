"""K1 gait direction asymmetry / lateral-forward drift diagnostic (sim).

On real hardware the user sees: forward walk = fast pattering + near-fall,
backward = smooth; lateral move drifts forward. Mirror symmetry only enforces
LEFT-RIGHT symmetry (+vy<->-vy, +wz<->-wz), so front-back asymmetry and a
lateral->forward coupling are NOT what mirror fixes. This diagnostic tells
whether those effects show up IN SIM, which is the only way to attribute them:

  - reproduced in sim  -> policy / dynamics  (COM is NOT the culprit)
  - clean in sim, only on real  -> sim2real   (COM / mass / latency)

It rolls out under the policy's own random commands (NO pinning — pinning breaks
the gait, a prior lesson) and buckets each (step, env) sample by command
direction, reporting the achieved body-frame velocity, the CROSS-drift
(lateral command -> forward velocity), and forward tilt per direction.

Run (JAX -> jaxpy):
    jaxpy -m rlworld.scripts.diag.k1_gait_direction_diag \\
        --wandb-run-path jsw7460/K1_Joystick/ql6fzhj9 --sim mujoco \\
        --num-envs 256 --steps 2000
"""

from __future__ import annotations

import argparse


def main() -> int:
    import numpy as np
    import torch

    from rlworld.rl.evals import PolicyEvaluator

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-run-path", required=True)
    ap.add_argument("--sim", default="mujoco", choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--com-x",
        type=float,
        default=0.0,
        help="forced Trunk COM forward offset (m); +ve pushes COM forward (tests forward tipping)",
    )
    ap.add_argument("--com-y", type=float, default=0.0, help="forced Trunk COM lateral offset (m)")
    args = ap.parse_args()

    print(f"[gait-dir] loading {args.wandb_run_path} on {args.sim} (num_envs={args.num_envs})")
    ev = PolicyEvaluator(
        wandb_run_path=args.wandb_run_path,
        eval_target=args.sim,
        num_evals=1,
        seed=args.seed,
        record_video=False,
        save_data=False,
        use_rich_display=False,
        extra_overrides={"env": {"num_envs": args.num_envs}},
    )
    env, policy = ev.env, ev.policy

    from rlworld.rl.configs.scene import SceneEntitySelector
    from rlworld.rl.envs.mdp.events.dr import unified as unified_dr

    # DR backends expect a RESOLVED entity (body_ids); resolve once like the event manager does.
    trunk_resolved = env.resolve_selector(SceneEntitySelector(name="robot", body_names=("Trunk",)))

    def _force_com() -> None:
        """Pin the Trunk COM to a fixed forced offset (added onto the MJCF default)."""
        if args.com_x == 0.0 and args.com_y == 0.0:
            return
        ranges = {}
        if args.com_x != 0.0:
            ranges[0] = (args.com_x, args.com_x)
        if args.com_y != 0.0:
            ranges[1] = (args.com_y, args.com_y)
        unified_dr.randomize_body_com_offset(
            env,
            env_ids=torch.arange(env.num_envs, device=env.device),
            asset_cfg=trunk_resolved,
            ranges=ranges,
            operation="add",
        )

    _force_com()
    if args.com_x or args.com_y:
        print(f"[gait-dir] forced Trunk COM offset: x={args.com_x:+.3f} y={args.com_y:+.3f} m")

    obs = env.obs_manager.get_observation()
    rs = env.get_robot_state()

    CMD, VB, WB, G = [], [], [], []
    n_term = 0
    with torch.no_grad():
        for t in range(args.steps):
            action = policy.get_action(obs, rs)
            obs, _rew, term, trunc, _extras = env.step(action)
            rs = env.get_robot_state()
            n_term += int(term.sum().item())
            done = term | trunc
            if done.any():
                policy.notify_reset(done.cpu().numpy())
                _force_com()  # reset restores the baseline COM; re-apply the forced offset
            rd = env.get_robot_data()
            CMD.append(env.command_manager.get_commands_tensor()[:, :3].cpu().numpy())  # (N,3) body
            VB.append(rd.root_link_lin_vel_b.cpu().numpy())  # (N,3) body [fwd, lat, up]
            WB.append(rd.root_link_ang_vel_b.cpu().numpy())  # (N,3) body
            G.append(rd.projected_gravity_b.cpu().numpy())  # (N,3) body
            if t % 500 == 0:
                print(f"  step {t}/{args.steps}")

    cmd = np.concatenate(CMD)
    vb = np.concatenate(VB)
    wb = np.concatenate(WB)
    g = np.concatenate(G)
    cx, cy, cw = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    fwd, lat = vb[:, 0], vb[:, 1]
    wz = wb[:, 2]
    tilt_fwd = g[:, 0]  # forward component of projected gravity ~ forward tilt (pitch proxy)

    thr, eps = 0.3, 0.15
    F = (cx > thr) & (np.abs(cy) < eps) & (np.abs(cw) < eps)
    B = (cx < -thr) & (np.abs(cy) < eps) & (np.abs(cw) < eps)
    L = (cy > thr) & (np.abs(cx) < eps) & (np.abs(cw) < eps)
    R = (cy < -thr) & (np.abs(cx) < eps) & (np.abs(cw) < eps)

    def grp(mask, name):
        n = int(mask.sum())
        if n < 50:
            print(f"  {name:<9} n={n} (too few — raise --steps/--num-envs)")
            return
        print(
            f"  {name:<9} n={n:6d} | cmd=({cx[mask].mean():+.2f},{cy[mask].mean():+.2f},{cw[mask].mean():+.2f})"
            f" | achieved fwd={fwd[mask].mean():+.3f} lat={lat[mask].mean():+.3f} wz={wz[mask].mean():+.3f}"
            f" | tilt_fwd={tilt_fwd[mask].mean():+.3f}"
        )

    com_tag = f", forced COM x={args.com_x:+.3f} y={args.com_y:+.3f}" if (args.com_x or args.com_y) else ", COM=default"
    print(f"\n=== gait direction ({args.sim}, samples={cmd.shape[0]}{com_tag}) ===")
    print("(achieved = body-frame velocity; drift = nonzero orthogonal component)")
    grp(F, "FORWARD")
    grp(B, "BACKWARD")
    grp(L, "LEFT")
    grp(R, "RIGHT")

    print("\n[Tipping / stability]")
    print(
        f"  terminations (falls): {n_term} over {args.steps} steps x {args.num_envs} envs"
        f"  |  mean forward tilt (all)={tilt_fwd.mean():+.4f}"
    )
    print("  -> run COM=default vs --com-x 0.03/0.05 and compare: rising falls / forward tilt = COM tips it forward")

    print("\n[Front-back asymmetry]  (mirror does NOT address this axis)")
    if F.sum() > 50 and B.sum() > 50:
        print(
            f"  forward tilt {tilt_fwd[F].mean():+.3f}  vs  backward tilt {tilt_fwd[B].mean():+.3f}"
            f"   (large gap = forward leans/unstable -> 'near-fall pattering')"
        )
        print(
            f"  fwd speed tracking: forward {fwd[F].mean():+.3f}/{cx[F].mean():+.2f}"
            f"   backward {fwd[B].mean():+.3f}/{cx[B].mean():+.2f}"
        )

    print("\n[Lateral -> forward drift]  (COM discriminator)")
    for m, name in ((L, "LEFT"), (R, "RIGHT")):
        if m.sum() > 50:
            fd = fwd[m].mean()
            print(
                f"  {name}: lateral cmd {cy[m].mean():+.2f} -> forward vel {fd:+.3f}"
                f"   ({'DRIFT' if abs(fd) > 0.05 else 'clean'})"
            )
    print("  -> forward drift IN SIM  => policy coupling (NOT COM).")
    print("     clean in SIM but drifts on REAL => COM / mass sim2real mismatch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
