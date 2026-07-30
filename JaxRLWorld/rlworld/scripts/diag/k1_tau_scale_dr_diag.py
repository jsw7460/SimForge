"""K1 actuator-saturation DR: does ``randomize_tau_scale`` randomize the tanh
scale kappa per env, in [0.5, 1.0]x the configured value, re-applied each reset?

kappa lives on the explicit Python PD actuator (sim-agnostic), so the DR is the
same code on every backend — this diag builds the real env per sim, reads the
live ``actuator.tau_scale`` across resets, and checks, per sim:

  1. per-env spread: std across envs > 0 — kappa is randomized per env, not one
     shared value (the mjlab per-world footgun does not apply here since kappa
     is an actuator tensor, not a model field).
  2. in band: kappa / base in [0.5, 1.0] — the configured scale range, and never
     above the base (scale<=1.0).
  3. across resets: the per-env values CHANGE — reset_dr re-samples each reset.

Companion to ``k1_actuator_saturation_diag`` (which proves the tanh math). Base
kappa is the pre-DR value the term captures (``_dr_base_tau_scale``).

Run::

    jaxpy -m rlworld.scripts.diag.k1_tau_scale_dr_diag --sim mujoco
    jaxpy -m rlworld.scripts.diag.k1_tau_scale_dr_diag              # all three
"""

from __future__ import annotations

import argparse

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_EXPECTED = (0.5, 1.0)  # dr_tau_scale scale range in base.py
_TOL = 1e-3
_N_RESETS = 3


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _read_ratio(env):
    """Per-env kappa / base_kappa over the actuated joints. (num_envs, n_joint)."""

    actuator, _ = env.act_manager.actuators[0]
    kap = actuator.tau_scale.detach().float().cpu()  # (num_envs, nj), DR'd
    base = actuator._dr_base_tau_scale.detach().float().cpu()  # (num_envs, nj), pre-DR
    return kap / base.clamp(min=1e-6)


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

    lo, hi = _EXPECTED
    out: dict = {"sim": sim, "num_envs": num_envs, "n_joint": None, "resets": []}
    prev = None
    for k in range(_N_RESETS):
        r = _read_ratio(env)  # (num_envs, n_joint)
        out["n_joint"] = r.shape[1]
        per_dof_std = r.std(dim=0)
        rec = {
            "min": float(r.min()),
            "max": float(r.max()),
            "mean": float(r.mean()),
            "in_band_frac": float(((r >= lo - _TOL) & (r <= hi + _TOL)).float().mean()),
            "max_per_env_std": float(per_dof_std.max()),
            "min_per_env_std": float(per_dof_std.min()),
        }
        if prev is not None:
            rec["changed_vs_prev_frac"] = float((r != prev).float().mean())
        out["resets"].append(rec)
        prev = r
        env.reset()
    _stage(f"cell done: {sim}")
    return out


def _print_cell(r: dict) -> None:
    lo, hi = _EXPECTED
    print(f"\n===== {r['sim'].upper()} (num_envs={r['num_envs']}, {r['n_joint']} actuated joints) =====")
    print(f"  expected kappa/base band: [{lo}, {hi}]  (scale DR of the configured kappa)")
    ok_band = ok_spread = ok_resample = True
    for i, rec in enumerate(r["resets"]):
        line = (
            f"  reset {i}: ratio min={rec['min']:.4f} max={rec['max']:.4f} mean={rec['mean']:.4f}  "
            f"in-band={rec['in_band_frac'] * 100:.1f}%  "
            f"per-env-std[min={rec['min_per_env_std']:.4f} max={rec['max_per_env_std']:.4f}]"
        )
        if "changed_vs_prev_frac" in rec:
            line += f"  changed_vs_prev={rec['changed_vs_prev_frac'] * 100:.1f}%"
            ok_resample = ok_resample and rec["changed_vs_prev_frac"] > 0.5
        print(line)
        ok_band = ok_band and rec["in_band_frac"] > 0.99 and rec["max"] <= hi + _TOL
        ok_spread = ok_spread and rec["max_per_env_std"] > 1e-3
    print(
        f"  → randomized per-env: {ok_spread} | in [0.5,1.0]x base: {ok_band} "
        f"| re-applied each reset: {ok_resample}"
    )
    print(f"  VERDICT: {'PASS' if (ok_band and ok_spread and ok_resample) else 'CHECK'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 tau_scale (kappa) DR landing / reset re-apply diag.")
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
