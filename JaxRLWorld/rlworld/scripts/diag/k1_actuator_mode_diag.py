"""K1 ``K1_ACTUATOR_MODE`` toggle: does the selected actuator bundle land
correctly on the LIVE PD actuator of ALL THREE backends?

``K1_ACTUATOR_MODE`` (read once at import in ``rlworld/.../robots/k1.py``)
swaps the whole per-joint bundle:

  legacy   – hand-tuned PD (``K1_PD_PROFILE``, stiff_legs): kp/kd from the
             profile, derated MJCF effort, recipe action_scale, plain PD.
  physical – Booster motor spec (booster_train): kp = J·ω_n², kd = 2ζ·J·ω_n,
             real motor effort (2-3× MJCF), action_scale = 0.25·effort/kp, and
             the piecewise-linear torque-speed (T-N) curve.

The mode is frozen at import, so this diag verifies the CURRENTLY-selected mode
against an INDEPENDENTLY re-derived bundle, and additionally cross-checks the
live gains against the OTHER mode's expected constants to prove the two modes
are numerically distinct — all in one process. Run it once per mode:

  jaxpy -m rlworld.scripts.diag.k1_actuator_mode_diag                      # legacy (default)
  K1_ACTUATOR_MODE=physical jaxpy -m rlworld.scripts.diag.k1_actuator_mode_diag

Add ``--sim mujoco`` (or newton/genesis) to restrict to one backend.

Per sim it proves:
  1. GAINS: NOMINAL kp/kd/effort/armature (re-resolved from the actuator cfg,
     pre-DR) match the selected mode's bundle. The recipe DR-randomises kp/kd
     per env AND per joint, so the live tensor is jittered (left!=right); the
     nominal source is what the builders wired. Live DR spread is reported too.
  2. ACTION_SCALE: physical ⇒ am._scale == 0.25·effort/kp per joint; legacy ⇒
     am._scale differs from the physical scale (recipe scale is reported).
  3. T-N CURVE: physical ⇒ actuator._use_tn, per-joint vel_limit/knee match, and
     a joint-velocity sweep saturates torque to the piecewise-linear ceiling to
     float tol; legacy ⇒ _use_tn is False (plain box clip).
  4. KAPPA: tau_scale disabled in both (sim-only tanh removed for sim2real).
  5. DISTINCT: live leg kp differs from the other mode's expected leg kp.
"""

from __future__ import annotations

import argparse
import math

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}

# ── Independent expected bundles (re-derived here; NOT imported from k1.py) ──
# Per group: (kp, kd, effort, armature, vel_limit, knee_point).
# LEGACY = stiff_legs profile + derated MJCF effort; no T-N curve.
_LEGACY = {
    "Head": (15.0, 2.0, 6.0, 0.002, None, None),
    "Shoulder": (15.0, 2.0, 14.0, 0.001, None, None),
    "Elbow": (15.0, 2.0, 14.0, 0.001, None, None),
    "Hip_Pitch": (50.0, 5.0, 30.0, 0.0478125, None, None),
    "Hip_Roll": (50.0, 5.0, 35.0, 0.0339552, None, None),
    "Hip_Yaw": (50.0, 5.0, 20.0, 0.0282528, None, None),
    "Knee": (50.0, 5.0, 40.0, 0.095625, None, None),
    "Ankle": (15.0, 5.0, 20.0, 0.0565, None, None),
}
# PHYSICAL = Booster motor spec (armature, effort, vel_limit, knee, freq, zeta).
_BOOSTER = {
    "Head": (0.001, 6.0, 7.85, 10.47, 10.0, 2.0),
    "Shoulder": (0.001, 14.0, 33.51, 5.24, 10.0, 2.0),
    "Elbow": (0.001, 14.0, 33.51, 5.24, 10.0, 2.0),
    "Hip_Pitch": (0.0478125, 68.0, 14.66, 1.88, 4.0, 1.5),
    "Hip_Roll": (0.0339552, 76.0, 12.57, 2.62, 4.0, 1.5),
    "Hip_Yaw": (0.0282528, 38.3, 17.59, 7.85, 4.0, 1.5),
    "Knee": (0.095625, 112.0, 12.57, 2.09, 4.0, 1.0),
    "Ankle": (0.0565056, 38.3, 17.59, 7.85, 4.0, 1.5),
}


def _physical(group: str):
    a, e, vlim, knee, f, zeta = _BOOSTER[group]
    wn = 2.0 * math.pi * f
    kp = a * wn * wn
    kd = 2.0 * zeta * a * wn
    return (kp, kd, e, a, vlim, knee)


_EXPECTED = {"legacy": _LEGACY, "physical": {g: _physical(g) for g in _BOOSTER}}
# Group keys ordered so the LAST rfind match wins on full-chain joint names.
_GROUP_KEYS = ("Head", "Shoulder", "Elbow", "Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee", "Ankle")
_ATOL = 1e-3
_RTOL = 1e-3
# Joint-velocity sweep (rad/s) for the T-N ceiling check.
_VEL_SWEEP = [0.0, 1.0, 3.0, 6.0, 10.0, 20.0]


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _group_of(name: str) -> str | None:
    best, best_pos = None, -1
    for key in _GROUP_KEYS:
        pos = name.rfind(key)
        if pos > best_pos:
            best_pos, best = pos, key
    return best


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    import rlworld.rl.configs.robots.k1 as k1

    mode = k1.K1_ACTUATOR_MODE
    torch.manual_seed(seed)
    _stage(f"cell start: {sim} mode={mode} num_envs={num_envs} seed={seed}")

    from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    preset = K1G1RecipeConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    cfgs = preset.build()
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()
    _stage("env built + first reset")

    am = env.act_manager
    names = list(am.actuated_joint_names)
    actuator, _ = am.actuators[0]
    exp = _EXPECTED[mode]
    other = "physical" if mode == "legacy" else "legacy"

    out: dict = {"sim": sim, "mode": mode, "num_envs": num_envs, "actuator": type(actuator).__name__, "names": names}

    # 1. gains vs expected. The recipe DR-randomises kp/kd per env AND per joint
    # (live ``actuator.stiffness[env]`` is a jittered, left!=right draw), so the
    # bundle-landing check reads the NOMINAL gains re-resolved from the actuator
    # cfg (pre-DR) — same source the builders wired. The live DR spread is
    # reported separately below so it is visible, not silently hidden.
    kp = actuator._resolve_per_joint_param(actuator.cfg.stiffness, default=0.0)[0].detach().float().cpu()
    kd = actuator._resolve_per_joint_param(actuator.cfg.damping, default=0.0)[0].detach().float().cpu()
    eff = actuator._resolve_per_joint_param(actuator.cfg.effort_limit, default=0.0)[0].detach().float().cpu()
    arm = actuator._resolve_per_joint_param(actuator.cfg.armature, default=0.0)[0].detach().float().cpu()
    # Live (post-DR) stiffness spread across envs, for transparency.
    live_kp = actuator.stiffness.detach().float()
    kp_nom_t = actuator._resolve_per_joint_param(actuator.cfg.stiffness, default=0.0).detach().float()
    dr_active = bool((live_kp != kp_nom_t).any())
    dr_mean_dev = float(((live_kp.mean(0) - kp_nom_t[0]).abs() / kp_nom_t[0].clamp(min=1e-6)).max())
    rows = []
    gains_ok = True
    distinct_ok = True
    for i, n in enumerate(names):
        g = _group_of(n)
        e_kp, e_kd, e_eff, e_arm, _, _ = exp[g]
        o_kp = _EXPECTED[other][g][0]
        row_ok = (
            math.isclose(float(kp[i]), e_kp, rel_tol=_RTOL, abs_tol=_ATOL)
            and math.isclose(float(kd[i]), e_kd, rel_tol=_RTOL, abs_tol=_ATOL)
            and math.isclose(float(eff[i]), e_eff, rel_tol=_RTOL, abs_tol=_ATOL)
            and math.isclose(float(arm[i]), e_arm, rel_tol=_RTOL, abs_tol=_ATOL)
        )
        gains_ok = gains_ok and row_ok
        # leg gains must differ from the other mode (proves the toggle bites).
        if g in ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee", "Ankle"):
            distinct_ok = distinct_ok and not math.isclose(float(kp[i]), o_kp, rel_tol=1e-2)
        rows.append((n, float(kp[i]), e_kp, float(kd[i]), e_kd, float(eff[i]), e_eff, float(arm[i]), e_arm, row_ok))
    out.update(gains_ok=gains_ok, distinct_ok=distinct_ok, rows=rows, dr_active=dr_active, dr_mean_dev=dr_mean_dev)

    # 2. action_scale. Force CPU on every tensor: a global torch device context
    # (cuda:0) is active during env build, so bare ``torch.tensor(...)`` would
    # land on cuda and mismatch the CPU-copied scale.
    scale = am._scale.detach().float().cpu()
    scale = scale[0] if scale.ndim == 2 else scale
    if mode == "physical":
        phys_scale = torch.tensor([0.25 * exp[_group_of(n)][2] / exp[_group_of(n)][0] for n in names], device="cpu")
        scale_ok = bool(torch.allclose(scale, phys_scale, rtol=_RTOL, atol=_ATOL))
        out["scale_detail"] = [(n, float(scale[i]), float(phys_scale[i])) for i, n in enumerate(names)]
    else:
        # legacy: scale is recipe-driven; only require it is NOT the physical scale.
        phys_legacy = torch.tensor(
            [0.25 * _physical(_group_of(n))[2] / _physical(_group_of(n))[0] for n in names], device="cpu"
        )
        scale_ok = not bool(torch.allclose(scale, phys_legacy, rtol=1e-2, atol=1e-2))
        out["scale_sample"] = [(n, float(scale[i])) for i, n in enumerate(names[:6])]
    out["scale_ok"] = scale_ok

    # 3. T-N curve.
    use_tn = bool(getattr(actuator, "_use_tn", False))
    out["use_tn"] = use_tn
    if mode == "physical":
        tn_wire_ok = use_tn
        tn_math_ok = True
        if use_tn:
            vlim = actuator._vel_limit[0].detach().float().cpu()
            knee = actuator._knee_point[0].detach().float().cpu()
            # per-joint vel_limit/knee wiring
            for i, n in enumerate(names):
                _, _, _, _, e_vl, e_kn = exp[_group_of(n)]
                tn_wire_ok = tn_wire_ok and math.isclose(float(vlim[i]), e_vl, rel_tol=_RTOL, abs_tol=_ATOL)
                tn_wire_ok = tn_wire_ok and math.isclose(float(knee[i]), e_kn, rel_tol=_RTOL, abs_tol=_ATOL)
            # velocity sweep: force a huge PD torque, verify applied == T-N ceiling.
            dev, ne, nj = actuator._device, actuator._num_envs, actuator._num_joints
            elim = actuator.effort_limit
            tn_sweep = []
            for v in _VEL_SWEEP:
                q = torch.zeros(ne, nj, device=dev)
                dq = torch.full((ne, nj), float(v), device=dev)  # +vel ⇒ PD wants -torque; push target high +
                tgt = torch.full((ne, nj), 100.0, device=dev)  # saturate the raw PD torque
                for _ in range(getattr(actuator, "_max_delay", 1) + 1):
                    applied = actuator.compute(tgt, q, dq)
                # expected ceiling per joint (env 0)
                ceil = torch.minimum(
                    torch.clamp(elim * (actuator._vel_limit - dq.abs()) / actuator._tn_denom, min=0.0),
                    elim,
                )
                err = float((applied - torch.clamp(actuator.computed_effort, -ceil, ceil)).abs().max())
                tn_math_ok = tn_math_ok and err < 1e-3
                tn_sweep.append((v, float(applied[0].abs().mean()), float(ceil[0].abs().mean()), err))
            out["tn_sweep"] = tn_sweep
        out["tn_wire_ok"] = tn_wire_ok
        out["tn_math_ok"] = tn_math_ok
    else:
        out["tn_wire_ok"] = not use_tn  # legacy: T-N must be OFF

    # 4. kappa disabled in both.
    out["kappa_off"] = not bool(getattr(actuator, "_use_tau_scale", False))

    _stage(f"cell done: {sim}")
    return out


def _print_cell(r: dict) -> None:
    sim, mode = r["sim"], r["mode"]
    print(f"\n===== {sim.upper()} | mode={mode} (num_envs={r['num_envs']}, actuator={r['actuator']}) =====")

    print("  [gains] NOMINAL (pre-DR) joint | kp(exp) | kd(exp) | eff(exp) | arm(exp) | ok")
    for n, kp, ekp, kd, ekd, eff, eeff, arm, earm, ok in r["rows"]:
        flag = "" if ok else "  !!"
        print(
            f"    {n:22} {kp:7.3f}({ekp:.3f}) {kd:6.3f}({ekd:.3f}) "
            f"{eff:6.2f}({eeff:.2f}) {arm:.5f}({earm:.5f}){flag}"
        )
    print(f"  [gains] nominal bundle matches {mode}: {r['gains_ok']}  {'OK' if r['gains_ok'] else '!!'}")
    print(
        f"  [DR] kp/kd randomised per-env: {r['dr_active']}  "
        f"(live mean vs nominal max dev {100*r['dr_mean_dev']:.1f}% — sampling noise, not a bundle error)"
    )
    print(f"  [distinct] nominal leg kp != other-mode leg kp: {r['distinct_ok']}  {'OK' if r['distinct_ok'] else '!!'}")

    if mode == "physical":
        print(f"  [action_scale] am._scale == 0.25·eff/kp: {r['scale_ok']}  {'OK' if r['scale_ok'] else '!!'}")
        if not r["scale_ok"]:
            for n, s, e in r["scale_detail"]:
                fl = "" if math.isclose(s, e, rel_tol=_RTOL, abs_tol=_ATOL) else "  !!"
                print(f"      {n:22} scale={s:.4f} exp={e:.4f}{fl}")
    else:
        print(
            f"  [action_scale] differs from physical scale (recipe-driven): "
            f"{r['scale_ok']}  {'OK' if r['scale_ok'] else '!!'}"
        )
        print(f"      sample: {[(n, round(s,4)) for n,s in r['scale_sample']]}")

    if mode == "physical":
        print(
            f"  [T-N] wired (_use_tn + per-joint vel_limit/knee): "
            f"{r['tn_wire_ok']}  {'OK' if r['tn_wire_ok'] else '!!'}"
        )
        if "tn_sweep" in r:
            print("  [T-N sweep]  vel |applied| | ceiling | max_err")
            for v, ap, ce, err in r["tn_sweep"]:
                print(f"      {v:5.1f} | {ap:7.3f} | {ce:7.3f} | {err:.2e}")
            print(
                f"  [T-N] applied == piecewise-linear ceiling: "
                f"{r['tn_math_ok']}  {'OK' if r['tn_math_ok'] else '!!'}"
            )
    else:
        print(f"  [T-N] disabled (legacy plain PD): {r['tn_wire_ok']}  {'OK' if r['tn_wire_ok'] else '!!'}")

    print(f"  [kappa] tanh saturation OFF: {r['kappa_off']}  {'OK' if r['kappa_off'] else '!!'}")

    ok = (
        r["gains_ok"]
        and r["distinct_ok"]
        and r["scale_ok"]
        and r["tn_wire_ok"]
        and r.get("tn_math_ok", True)
        and r["kappa_off"]
    )
    print(f"  VERDICT: {'PASS' if ok else 'CHECK'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 K1_ACTUATOR_MODE (legacy/physical) wiring + correctness diag.")
    ap.add_argument("--sim", choices=_SIMS, help="Single backend (default: all).")
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import rlworld.rl.configs.robots.k1 as k1

    print(f"K1_ACTUATOR_MODE = {k1.K1_ACTUATOR_MODE!r}  " f"(set K1_ACTUATOR_MODE=physical to verify the other bundle)")

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
    print()
    return 0 if len(results) == len(sims) else 1


if __name__ == "__main__":
    raise SystemExit(main())
