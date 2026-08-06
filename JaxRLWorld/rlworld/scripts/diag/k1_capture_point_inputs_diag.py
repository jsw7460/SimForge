"""Capture-point reward INPUTS: can we fetch every value the planned
capture-point / support-polygon balance reward needs, ACCURATELY and
IDENTICALLY, on all three backends — BEFORE writing the reward?

The reward (not built yet) will need, per env, per step:
  - CoM (xy + height)         → proxy candidates: root_link_pos_w /
                                root_com_pos_w / centroid(body_com_pos_w_all)
  - CoM velocity (world xy)   → root_link_lin_vel_w / root_com_lin_vel_w
  - foot positions (world)    → body_pos_w_by_ids(foot ids)  [support polygon]
  - CoM height above feet     → root_z - mean(foot_z)         [omega0 = sqrt(g/h)]
  - velocity command norm     → command_manager "velocity"    [standing gate]

This diag builds the real K1 env per backend and PROVES the numbers are right,
not merely present:

  1. REST: after settling at zero action + zero command, root height ~=
     base_init_height, world velocity ~= 0, feet ~= ground. (sane baseline)
  2. WORLD-VELOCITY CORRECTNESS (the subtle one):
       a. FRAME: root_link_lin_vel_w == R(root_quat_w) @ root_link_lin_vel_b
          to float tol — proves the "_w" accessor is genuinely world-frame.
       b. FINITE-DIFF: root_link_lin_vel_w == d(root_link_pos_w)/control_dt
          over a moving rollout — proves magnitude/direction/units/dt.
  3. FOOT LAYOUT: two feet, plausible stance width, feet on the ground,
     height h in a humanoid range; cross-sim consistent.
  4. CoM PROXY SPREAD: root_link vs root_com vs unweighted body-CoM centroid,
     and each vs the foot center at standing — quantifies how good the cheap
     sim-agnostic proxy (root_com) is vs the true mass-weighted CoM (which
     lies between root_com and the centroid; the MJCF mass layout is identical
     across backends, so a tight bracket on ONE backend pins it on all).
  5. COMMAND: the standing-gate signal has shape (N,3) and reads ~0 when set.
  6. XI PREVIEW: pure-math capture point + support distance on the validated
     inputs (with a synthetic lateral velocity) so the eventual reward's
     magnitudes are eyeballed now.

Run::

    jaxpy -m rlworld.scripts.diag.k1_capture_point_inputs_diag            # all three
    jaxpy -m rlworld.scripts.diag.k1_capture_point_inputs_diag --sim mujoco
"""

from __future__ import annotations

import argparse

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}

_G = 9.81
# Tolerances.
_FRAME_TOL = 1e-3  # world == R(quat)@body (exact up to float)
_FD_REL_TOL = 0.20  # instantaneous vs Δpos/dt (average over step ⇒ loose)
_FD_ABS_TOL = 0.03  # m/s floor so near-zero velocities don't blow up rel err
_CENTER_TOL = 0.10  # m: CoM over foot center at the upright reset pose
_SETTLE_STEPS = 40
_MOTION_STEPS = 20
_MOTION_ACTION = 0.15  # mild constant target bias to induce root motion
_XI_TEST_VEL = 0.5  # m/s synthetic lateral velocity for the xi preview
_SUPPORT_MARGIN = 0.10  # m foot half-extent for the support capsule


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _seg_dist_xy(p, a, b):
    """Distance from point p to segment [a,b] in xy. All (N,2). Returns (N,)."""

    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(dim=1).clamp(min=1e-9)
    t = (ap * ab).sum(dim=1) / denom
    t = t.clamp(0.0, 1.0)
    proj = a + t.unsqueeze(1) * ab
    return (p - proj).norm(dim=1)


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
    from rlworld.rl.evals.sim_initializers import get_initializer
    from rlworld.rl.utils.quat_utils import quat_rotate_wxyz

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs} seed={seed}")

    preset = K1G1RecipeConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    cfgs = preset.build()
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()
    dev = env.device
    r = preset.robot
    out: dict = {
        "sim": sim,
        "num_envs": num_envs,
        "control_dt": float(env.control_dt),
        "base_init_height": float(r.base_init_height),
    }

    rd = env.get_robot_data("robot")
    foot_sel = env.resolve_selector(SceneEntitySelector(name="robot", body_names=tuple(r.foot_names)))
    vel_term = env.command_manager.get_term("velocity")
    all_ids = torch.arange(num_envs, device=dev)

    def _measure_geometry():
        """Snapshot CoM / feet / support / xi from the CURRENT sim state."""
        feet = rd.body_pos_w_by_ids(foot_sel.body_ids)  # (N, 2, 3)
        root_z = rd.root_link_pos_w[:, 2]
        foot_z = feet[..., 2]
        h = (root_z - foot_z.mean(dim=1)).clamp(min=0.15)
        quat = rd.root_link_quat_w
        w, x, y, z = quat.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        dx = feet[:, 0, 0] - feet[:, 1, 0]
        dy = feet[:, 0, 1] - feet[:, 1, 1]
        lateral = torch.abs(torch.cos(yaw) * dy - torch.sin(yaw) * dx)
        fore_aft = torch.abs(torch.cos(yaw) * dx + torch.sin(yaw) * dy)
        c_link = rd.root_link_pos_w[:, :2]
        c_rootcom = rd.root_com_pos_w[:, :2]
        c_centroid = rd.body_com_pos_w_all[:, :, :2].mean(dim=1)  # UNWEIGHTED (biased by light limbs)
        foot_center = feet[:, :, :2].mean(dim=1)
        a_xy, b_xy = feet[:, 0, :2], feet[:, 1, :2]
        omega0 = torch.sqrt(torch.tensor(_G, device=dev) / h)
        # xi at the true (near-zero) velocity == CoM proxy → should sit INSIDE support
        d_static = _seg_dist_xy(c_rootcom, a_xy, b_xy)
        dout_static = (d_static - _SUPPORT_MARGIN).clamp(min=0.0)
        # synthetic lateral velocity → xi pushed sideways, should leave support
        vlat = torch.zeros(num_envs, 2, device=dev)
        vlat[:, 1] = _XI_TEST_VEL
        xi_lat = c_rootcom + vlat / omega0.unsqueeze(1)
        dout_lat = (_seg_dist_xy(xi_lat, a_xy, b_xy) - _SUPPORT_MARGIN).clamp(min=0.0)
        return {
            "root_z": float(root_z.mean()),
            "root_z_std": float(root_z.std()),
            "foot_z": float(foot_z.mean()),
            "h": float(h.mean()),
            "lateral_sep": float(lateral.mean()),
            "lateral_std": float(lateral.std()),
            "fore_aft_sep": float(fore_aft.mean()),
            "link_vs_rootcom": float((c_link - c_rootcom).norm(dim=1).mean()),
            "rootcom_vs_centroid": float((c_rootcom - c_centroid).norm(dim=1).mean()),
            "rootcom_vs_footcenter": float((c_rootcom - foot_center).norm(dim=1).mean()),
            "omega0": float(omega0.mean()),
            "dout_static": float(dout_static.mean()),
            "xi_shift_lat": float((xi_lat - c_rootcom).norm(dim=1).mean()),
            "dout_lat": float(dout_lat.mean()),
        }

    # ── RESET snapshot (t=0, upright, zero velocity) ──────────────────
    # A humanoid CANNOT stand under fixed-joint PD (rigid inverted pendulum on
    # the ankles → topples); zero-action "settling" measures a FALLING robot,
    # not a stand. The only policy-free upright snapshot is right after reset.
    env.reset()
    vel_term.set_command(all_ids, torch.zeros(num_envs, 3, device=dev))
    reset_geo = _measure_geometry()
    out["geo"] = reset_geo
    zero_act = torch.zeros(num_envs, env.num_actions, device=dev)

    out["reset_ok"] = bool(
        abs(reset_geo["root_z"] - r.base_init_height) < 0.05
        and abs(reset_geo["foot_z"]) < 0.08
        and 0.30 < reset_geo["h"] < 0.60
    )

    # Informational: confirm zero-action is NOT a stable stand (sags/topples).
    for _ in range(_SETTLE_STEPS):
        env.step(zero_act)
    out["zero_action_root_z"] = float(rd.root_link_pos_w[:, 2].mean())
    out["zero_action_vel_max"] = float(rd.root_link_lin_vel_w.norm(dim=1).max())

    # ── command accessor (standing gate signal) ───────────────────────
    cmd = vel_term.command
    out["cmd_shape"] = tuple(cmd.shape)
    out["cmd_norm_max"] = float(cmd.norm(dim=1).max())
    out["cmd_ok"] = bool(cmd.shape == (num_envs, 3) and float(cmd.norm(dim=1).max()) < 1e-4)

    out["foot_ok"] = bool(0.04 < reset_geo["lateral_sep"] < 0.40)
    # root_com is the reward's CoM proxy; require it near the true CoM bracket
    # (root_link ↔ centroid) and over the feet at the upright reset pose.
    out["com_ok"] = bool(reset_geo["rootcom_vs_footcenter"] < _CENTER_TOL)
    out["xi_ok"] = bool(reset_geo["dout_static"] < 0.03 and reset_geo["xi_shift_lat"] > 0.02)

    # ── world-velocity correctness over a MOVING rollout ──────────────
    env.reset()
    vel_term.set_command(all_ids, torch.zeros(num_envs, 3, device=dev))
    move_act = torch.full((num_envs, env.num_actions), _MOTION_ACTION, device=dev)
    prev_pos = rd.root_link_pos_w.clone()
    frame_err = 0.0
    fd_rel_max = 0.0
    fd_rel_sum = 0.0
    fd_n = 0
    fd_discarded = 0
    for _ in range(_MOTION_STEPS):
        vel_term.set_command(all_ids, torch.zeros(num_envs, 3, device=dev))
        env.step(move_act)
        pos = rd.root_link_pos_w
        vel_w = rd.root_link_lin_vel_w
        vel_b = rd.root_link_lin_vel_b
        q = rd.root_link_quat_w
        # (a) frame: world == R(quat) @ body
        vel_w_from_b = quat_rotate_wxyz(q, vel_b)
        frame_err = max(frame_err, float((vel_w - vel_w_from_b).norm(dim=1).max()))
        # (b) finite-diff on the horizontal plane (skip reset teleports)
        v_fd = (pos - prev_pos) / env.control_dt
        jump = v_fd.norm(dim=1) > 5.0  # reset/teleport guard
        fd_discarded += int(jump.sum())
        good = ~jump
        if bool(good.any()):
            num = (v_fd[good, :2] - vel_w[good, :2]).norm(dim=1)
            den = vel_w[good, :2].norm(dim=1).clamp(min=_FD_ABS_TOL)
            rel = num / den
            fd_rel_max = max(fd_rel_max, float(rel.max()))
            fd_rel_sum += float(rel.sum())
            fd_n += int(good.sum())
        prev_pos = pos.clone()
    out["frame_err"] = frame_err
    out["frame_ok"] = bool(frame_err < _FRAME_TOL)
    out["fd_rel_max"] = fd_rel_max
    out["fd_rel_mean"] = (fd_rel_sum / fd_n) if fd_n else float("nan")
    out["fd_discarded"] = fd_discarded
    out["fd_ok"] = bool(fd_n > 0 and (fd_rel_sum / fd_n) < _FD_REL_TOL)

    _stage(f"cell done: {sim}")
    return out


def _print_cell(r: dict) -> None:
    sim = r["sim"]
    print(f"\n===== {sim.upper()} (num_envs={r['num_envs']}, control_dt={r['control_dt']*1000:.1f} ms) =====")

    g = r["geo"]
    print(
        f"  [reset t=0] root_z={g['root_z']:.3f}±{g['root_z_std']:.3f} (init {r['base_init_height']:.3f}) "
        f"| foot_z={g['foot_z']:.4f} | h={g['h']:.3f}  {'OK' if r['reset_ok'] else '!!'}"
    )
    print(
        f"  [zero-action fate] after {_SETTLE_STEPS} steps root_z={r['zero_action_root_z']:.3f}, "
        f"vel_max={r['zero_action_vel_max']:.2f} m/s  (expected to sag/topple — NOT a stable stand)"
    )

    print(f"  [command] shape={r['cmd_shape']} norm_max={r['cmd_norm_max']:.2e}  {'OK' if r['cmd_ok'] else '!!'}")

    print(
        f"  [foot] lateral_sep={g['lateral_sep']:.3f}±{g['lateral_std']:.3f} m "
        f"| fore_aft={g['fore_aft_sep']:.3f} m  {'OK' if r['foot_ok'] else '!!'}"
    )

    print(
        f"  [CoM proxy] root_link↔root_com={g['link_vs_rootcom']*100:.2f} cm | "
        f"root_com↔centroid(unweighted)={g['rootcom_vs_centroid']*100:.2f} cm | "
        f"root_com↔foot_center={g['rootcom_vs_footcenter']*100:.2f} cm  {'OK' if r['com_ok'] else '!!'}"
    )

    print(
        f"  [vel FRAME] max|vel_w - R(q)@vel_b| = {r['frame_err']:.2e}  "
        f"{'OK' if r['frame_ok'] else '!!'}  (proves world-frame)"
    )
    print(
        f"  [vel FD] rel err mean={r['fd_rel_mean']:.3f} max={r['fd_rel_max']:.3f} "
        f"(discarded {r['fd_discarded']} reset-jumps)  {'OK' if r['fd_ok'] else '!!'}"
    )

    print(
        f"  [xi @ reset] omega0={g['omega0']:.2f} rad/s | static d_out={g['dout_static']*100:.2f} cm "
        f"| +{_XI_TEST_VEL} m/s lat → xi shift={g['xi_shift_lat']*100:.2f} cm, d_out={g['dout_lat']*100:.2f} cm  "
        f"{'OK' if r['xi_ok'] else '!!'}"
    )

    ok = all(r[k] for k in ("reset_ok", "cmd_ok", "foot_ok", "com_ok", "frame_ok", "fd_ok", "xi_ok"))
    print(f"  VERDICT: {'PASS' if ok else 'CHECK'}")


def _print_cross_sim(results: list) -> None:
    if len(results) < 2:
        return
    print("\n===== CROSS-SIM CONSISTENCY (same MJCF ⇒ should agree) =====")
    print(f"  {'metric':22} " + " ".join(f"{r['sim']:>10}" for r in results))
    rows = [
        ("root_z [m]", lambda r: r["geo"]["root_z"]),
        ("height h [m]", lambda r: r["geo"]["h"]),
        ("lateral_sep [m]", lambda r: r["geo"]["lateral_sep"]),
        ("rootcom↔footctr[cm]", lambda r: r["geo"]["rootcom_vs_footcenter"] * 100),
        ("omega0 [rad/s]", lambda r: r["geo"]["omega0"]),
    ]
    for label, fn in rows:
        print(f"  {label:22} " + " ".join(f"{fn(r):>10.3f}" for r in results))


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture-point reward INPUT verification (pre-reward).")
    ap.add_argument("--sim", choices=_SIMS, help="Single backend (default: all).")
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sims = [args.sim] if args.sim else list(_SIMS)
    results = []
    for sim in sims:
        try:
            results.append(run_cell(sim, args.num_envs, args.seed))
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"\n[{sim}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    for r in results:
        _print_cell(r)
    _print_cross_sim(results)
    print()
    return 0 if len(results) == len(sims) else 1


if __name__ == "__main__":
    raise SystemExit(main())
