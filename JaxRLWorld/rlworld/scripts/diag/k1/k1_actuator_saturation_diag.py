"""K1 tanh actuator-saturation model: is ``tau = kappa * tanh(tau_PD / kappa)``
correctly wired into the live PD actuator on ALL THREE backends?

The saturation lives in ``IdealPDActuator.compute`` (sim-agnostic Python, run
for the explicit DelayedPD actuator on every backend), so correctness is shared
— but each preset's sim builder must pass ``tau_scale=r.tau_scale`` into the
actuator cfg, or that backend silently falls back to the plain hard clip. This
diag builds the real env per sim, grabs the live actuator, and proves, per sim:

  1. WIRING: the actuator has tau_scale set (``_use_tau_scale``) and the
     per-joint kappa matches the K1 config (head 6, arm 14, hip 30/35/yaw 20,
     knee 40, ankle 20). A missing builder wiring shows up as kappa absent.
  2. EXACT MATH: for a sweep of PD-torque magnitudes, the applied torque equals
     ``clip(kappa * tanh(tau_PD / kappa), +/- effort_limit)`` to float tol —
     tau_PD read from the actuator's own ``computed_effort`` (so DR'd kp/kd are
     accounted for automatically).
  3. REGIMES: small tau_PD passes ~linearly (applied ~= tau_PD); large tau_PD
     saturates toward +/-kappa AND rolls off below tau_PD (torque decay), all
     within the effort limit.
  4. NOT A NO-OP: at high torque the tanh output is strictly below what the old
     hard clip would deliver (reported as the decay vs clip).

Run::

    jaxpy -m rlworld.scripts.diag.k1.k1_actuator_saturation_diag --sim mujoco
    jaxpy -m rlworld.scripts.diag.k1.k1_actuator_saturation_diag            # all three
"""

from __future__ import annotations

import argparse

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}

# Expected kappa per joint group (K1 config: tau_scale == effort limits).
_EXP_KAPPA = {
    "Hip_Pitch": 30.0,
    "Hip_Roll": 35.0,
    "Hip_Yaw": 20.0,
    "Knee": 40.0,
    "Ankle": 20.0,
    "Head": 6.0,
    "Shoulder": 14.0,
    "Elbow": 14.0,
}
# PD-torque sweep is generated via target error (rad); actual tau_PD is read
# back from the actuator, so exact kp is irrelevant here.
_ERR_SWEEP = [0.002, 0.01, 0.05, 0.2, 0.5, 1.0, 5.0]
_MATCH_ATOL = 1e-5
_MATCH_RTOL = 1e-4


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _exp_kappa(name: str) -> float | None:
    """Expected kappa by LAST-occurring group key (handles full-chain names)."""
    best, best_pos = None, -1
    for key, val in _EXP_KAPPA.items():
        pos = name.rfind(key)
        if pos > best_pos:
            best_pos, best = pos, val
    return best


def _probe(actuator, err: float):
    """Drive the live actuator to a steady PD torque at target error ``err``
    (q=0, dq=0), flushing the delay buffer. Returns env-0 (tau_PD, applied)."""
    import torch

    dev, ne, nj = actuator._device, actuator._num_envs, actuator._num_joints
    q = torch.zeros(ne, nj, device=dev)
    dq = torch.zeros(ne, nj, device=dev)
    tgt = torch.full((ne, nj), float(err), device=dev)
    for _ in range(getattr(actuator, "_max_delay", 1) + 1):
        actuator.compute(tgt, q, dq)
    return actuator.computed_effort.detach().clone(), actuator.applied_effort.detach().clone()


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs} seed={seed}")

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
    out: dict = {"sim": sim, "num_envs": num_envs, "actuator": type(actuator).__name__, "names": names}

    # 1. wiring + per-joint kappa
    use = bool(getattr(actuator, "_use_tau_scale", False))
    out["use_tau_scale"] = use
    if use:
        kappa0 = actuator.tau_scale[0].detach().float().cpu()  # (nj,) — env 0 (DR draw)
        exp = torch.tensor([_exp_kappa(n) if _exp_kappa(n) is not None else float("nan") for n in names], device="cpu")
        out["kappa"] = [float(x) for x in kappa0]
        # kappa is DR-scaled to (0, base]; check bounded by the configured base
        # (== effort limit) and positive, not exact equality.
        out["kappa_match"] = bool(((kappa0 > 0) & (kappa0 <= exp * 1.001)).all())
        out["kappa_exp"] = [float(x) for x in exp]

    # 2-4. sweep: exact math + regimes + decay-vs-clip
    if use:
        elim = actuator.effort_limit  # (ne, nj)
        kap = actuator.tau_scale
        sweep = []
        max_math_err = 0.0
        for err in _ERR_SWEEP:
            computed, applied = _probe(actuator, err)  # (ne, nj)
            expected = torch.clamp(kap * torch.tanh(computed / kap), -elim, elim)
            hard_clip = torch.clamp(computed, -elim, elim)
            math_err = float((applied - expected).abs().max())
            max_math_err = max(max_math_err, math_err)
            ratio = (applied.abs() / kap).clamp(max=2.0)  # applied / kappa
            lin_rel = float(((applied - computed).abs() / computed.abs().clamp(min=1e-6)).mean())
            decay = float((hard_clip.abs() - applied.abs()).mean())  # >0 ⇒ tanh delivers less
            sweep.append(
                {
                    "err": err,
                    "tauPD_absmean": float(computed.abs().mean()),
                    "applied_absmean": float(applied.abs().mean()),
                    "ratio_mean": float(ratio.mean()),
                    "ratio_max": float(ratio.max()),
                    "math_err": math_err,
                    "lin_rel": lin_rel,
                    "decay_vs_clip": decay,
                    "within_kappa": bool((applied.abs() <= kap * 1.0001).all()),
                }
            )
        out["sweep"] = sweep
        out["max_math_err"] = max_math_err

    _stage(f"cell done: {sim}")
    return out


def _print_cell(r: dict) -> None:
    sim = r["sim"]
    print(f"\n===== {sim.upper()} (num_envs={r['num_envs']}, actuator={r['actuator']}) =====")

    use = r.get("use_tau_scale", False)
    print(f"  [wiring] tau_scale active: {use}  {'OK' if use else '!! FAIL (builder missing tau_scale=r.tau_scale)'}")
    if not use:
        print("  VERDICT: CHECK")
        return

    kmatch = r["kappa_match"]
    print(f"  [kappa] per-joint == config: {kmatch}  {'OK' if kmatch else '!!'}")
    if not kmatch:
        for n, k, e in zip(r["names"], r["kappa"], r["kappa_exp"]):
            flag = "" if abs(k - e) < 1e-4 else "  !!"
            print(f"      {n:24} kappa={k:.3f} exp={e:.3f}{flag}")

    print(
        f"  [exact math] max|applied - clip(kappa*tanh(tauPD/kappa))| = {r['max_math_err']:.2e}"
        f"  {'OK' if r['max_math_err'] < 1e-4 else '!!'}"
    )
    print("  [sweep]  err |  tauPD |applied| ratio(a/k) | math_err | lin_rel | decay_vs_clip | <=kappa")
    lin_ok = sat_ok = math_ok = True
    for s in r["sweep"]:
        print(
            f"    {s['err']:5.3f} | {s['tauPD_absmean']:6.2f} | {s['applied_absmean']:6.2f} |"
            f" {s['ratio_mean']:.3f}/{s['ratio_max']:.3f} | {s['math_err']:.1e} |"
            f" {s['lin_rel']:.4f} | {s['decay_vs_clip']:+.3f} | {s['within_kappa']}"
        )
        math_ok = math_ok and s["math_err"] < 1e-4
    # smallest err ⇒ near-linear (applied ~= tauPD); largest err ⇒ saturated
    lo, hi = r["sweep"][0], r["sweep"][-1]
    lin_ok = lo["lin_rel"] < 0.01
    sat_ok = hi["ratio_mean"] > 0.99 and hi["within_kappa"] and hi["decay_vs_clip"] > 0.0
    print(
        f"  [regimes] small tauPD linear (lin_rel={lo['lin_rel']:.4f}<0.01): {'OK' if lin_ok else '!!'}"
        f"  |  large tauPD saturates to kappa & decays (ratio={hi['ratio_mean']:.3f}, "
        f"decay={hi['decay_vs_clip']:+.2f}): {'OK' if sat_ok else '!!'}"
    )
    ok = use and kmatch and math_ok and lin_ok and sat_ok
    print(f"  VERDICT: {'PASS' if ok else 'CHECK'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 tanh actuator-saturation wiring / correctness diag.")
    ap.add_argument("--sim", choices=_SIMS, help="Single backend (default: all).")
    ap.add_argument("--num_envs", type=int, default=64)
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
    print()
    return 0 if len(results) == len(sims) else 1


if __name__ == "__main__":
    raise SystemExit(main())
