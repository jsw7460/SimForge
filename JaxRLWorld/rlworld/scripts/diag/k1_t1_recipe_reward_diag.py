"""K1 T1-recipe reward parity/health diag — 100%-confidence check that EVERY
T1-flat reward term is present, finite, correctly SIGNED, and produces sensible
values in ALL THREE backends.

The T1 recipe's reward functions are sim-agnostic, but a term can still silently
break per backend (a missing state field, a wrong selector, a sign-convention
slip that turns a penalty into a bonus). This diag drives the REAL env per sim
with random actions (so every term is exercised — feet lift/slide, single
stance, torque/acc spikes, terminations) and reads the per-term WEIGHTED
contribution the trainer actually sums (``env.rew_buf_per_type``, populated by
the reward manager during ``step`` — no re-call, so stateful terms
(feet_air_time / joint_acc) are not double-advanced).

Per sim it checks, for each of the 16 terms:
  1. PRESENT — the term is registered and produced a value.
  2. FINITE — no NaN / inf across all envs and steps.
  3. SIGN — the weighted contribution never has the wrong sign: reward terms
     (track_lin/ang, feet_air_time, upward) stay >= 0, penalty terms stay <= 0.
     This is the load-bearing check: it catches a flipped weight or a
     penalize_* function's negative-return convention being mishandled.
  4. ACTIVE — the always-on terms (upward, orientation, joint_pos_penalty,
     tracking) are non-trivially non-zero (not silently short-circuited).

A cross-sim table then shows each term's mean side by side so the three
backends can be eyeballed as consistent in sign and magnitude.

Run::

    jaxpy -m rlworld.scripts.diag.k1_t1_recipe_reward_diag --sim mujoco
    jaxpy -m rlworld.scripts.diag.k1_t1_recipe_reward_diag              # all three
"""

from __future__ import annotations

import argparse

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}

# name -> expected sign of the WEIGHTED contribution ("+" reward, "-" penalty).
_EXPECTED_SIGN = {
    "track_lin_vel": "+",
    "track_ang_vel": "+",
    "feet_air_time": "+",
    "upward": "+",
    "flat_orientation": "-",
    "ang_vel_xy": "-",
    "lin_vel_z": "-",
    "joint_torques": "-",
    "joint_acc": "-",
    "joint_deviation_hip": "-",
    "joint_deviation_arms": "-",
    "joint_pos_limits": "-",
    "joint_pos_penalty": "-",
    "action_rate": "-",
    "feet_slide": "-",
    "is_terminated": "-",
}
# Terms that must be non-trivially active every rollout (not gait-dependent).
_ALWAYS_ACTIVE = ("upward", "flat_orientation", "joint_pos_penalty", "track_lin_vel", "track_ang_vel")
_SIGN_TOL = 1e-6
_ACTIVE_TOL = 1e-9
_N_STEPS = 40


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs} seed={seed}")

    from rlworld.rl.configs.presets.k1_joystick.t1_recipe import K1T1RecipeConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    preset = K1T1RecipeConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    cfgs = preset.build()
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()
    n_act = len(env.act_manager.actuated_joint_names)

    registered = list(env.reward_manager.reward_terms.keys())
    # Accumulate per-term weighted values over the rollout.
    acc: dict[str, list] = {n: [] for n in registered}
    for _ in range(_N_STEPS):
        action = torch.randn(num_envs, n_act, device=env.device) * 0.5
        env.step(action)
        for name, val in env.rew_buf_per_type.items():
            if name == "total_reward":
                continue
            acc.setdefault(name, []).append(val.detach().float().cpu())

    out: dict = {"sim": sim, "num_envs": num_envs, "registered": registered, "terms": {}}
    for name, vals in acc.items():
        if not vals:
            out["terms"][name] = {"present": False}
            continue
        v = torch.stack(vals)  # (steps, envs)
        finite = bool(torch.isfinite(v).all())
        exp = _EXPECTED_SIGN.get(name)
        if exp == "+":
            sign_ok = bool((v >= -_SIGN_TOL).all())
        elif exp == "-":
            sign_ok = bool((v <= _SIGN_TOL).all())
        else:
            sign_ok = True
        out["terms"][name] = {
            "present": True,
            "finite": finite,
            "sign_ok": sign_ok,
            "exp_sign": exp,
            "mean": float(v.mean()),
            "min": float(v.min()),
            "max": float(v.max()),
            "absmean": float(v.abs().mean()),
        }
    _stage(f"cell done: {sim}")
    return out


def _print_cell(r: dict) -> dict:
    sim = r["sim"]
    print(f"\n===== {sim.upper()} (num_envs={r['num_envs']}) =====")
    expected = set(_EXPECTED_SIGN)
    got = set(n for n, t in r["terms"].items() if t.get("present"))
    missing = expected - got
    extra = got - expected
    print(
        f"  terms: {len(got)} present  (expected {len(expected)})"
        f"  missing={sorted(missing) or '-'}  extra={sorted(extra) or '-'}"
    )

    print(f"    {'term':22}{'exp':>4}{'mean':>12}{'min':>12}{'max':>12}  fin sign act")
    all_ok = not missing
    per_term_mean = {}
    for name in _EXPECTED_SIGN:
        t = r["terms"].get(name, {"present": False})
        if not t.get("present"):
            print(f"    {name:22}{'':>4}{'MISSING':>12}")
            all_ok = False
            continue
        per_term_mean[name] = t["mean"]
        fin = t["finite"]
        sgn = t["sign_ok"]
        act = (t["absmean"] > _ACTIVE_TOL) if name in _ALWAYS_ACTIVE else True
        all_ok = all_ok and fin and sgn and act
        print(
            f"    {name:22}{t['exp_sign']:>4}{t['mean']:>12.3e}{t['min']:>12.3e}{t['max']:>12.3e}"
            f"  {'OK' if fin else '!!'} {'OK' if sgn else '!!'} {'OK' if act else '!!'}"
        )
    print(f"  VERDICT: {'PASS' if all_ok else 'CHECK'}")
    return per_term_mean


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 T1-recipe reward health / 3-sim parity diag.")
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

    means = {}
    for r in results:
        means[r["sim"]] = _print_cell(r)

    if len(means) > 1:
        print("\n===== CROSS-SIM (weighted per-term mean) =====")
        sims_done = list(means)
        print(f"    {'term':22}" + "".join(f"{s:>12}" for s in sims_done))
        for name in _EXPECTED_SIGN:
            row = "".join(f"{means[s].get(name, float('nan')):>12.3e}" for s in sims_done)
            print(f"    {name:22}{row}")
        print("  (same sign + comparable magnitude across columns ⇒ 3-sim consistent)")

    print()
    return 0 if len(results) == len(sims) else 1


if __name__ == "__main__":
    raise SystemExit(main())
