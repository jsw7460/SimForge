"""K1 passive-damping DR: does ``randomize_joint_damping`` actually OVERRIDE the
model's ``dof_damping`` (XML 3 legs / 2 arms) with a per-env draw in [0, 1], and
is it RE-APPLIED on every reset (``reset_dr``)? Proven on all three backends.

Reads the per-env, per-actuated-joint ``dof_damping`` the SOLVER steps (mjwarp
``dof_damping`` for mujoco/newton, ``get_dofs_damping`` for genesis) across
several resets and checks, per sim:

  1. after build+reset: every value is in [0, 1] — the XML 3/2 is GONE
     (override landed, not a silent no-op leaving the baseline).
  2. per-env spread: std across envs > 0 — the value is randomized per env,
     not a single constant broadcast to all.
  3. across 3 resets: the per-env values CHANGE — ``reset_dr`` re-samples AND
     the new value reaches the model each reset.

Companion to ``k1_passive_damping_diag`` (which proves the un-randomized 3/2 is
present). Per-sim model access mirrors ``k1_friction_dr_diag``.

Run::

    jaxpy -m rlworld.scripts.diag.k1.k1_damping_dr_diag --sim mujoco
    jaxpy -m rlworld.scripts.diag.k1.k1_damping_dr_diag              # all three
"""

from __future__ import annotations

import argparse

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_EXPECTED = (0.0, 1.0)  # dr_joint_damping abs range in base.py
_TOL = 1e-3
_N_RESETS = 3


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _stats(t) -> dict:
    t = t.detach().float().reshape(-1)
    return {"min": float(t.min()), "max": float(t.max()), "mean": float(t.mean()), "std": float(t.std())}


def _hinge_dofadrs(mj) -> list[int]:
    import mujoco

    return [int(mj.jnt_dofadr[j]) for j in range(mj.njnt) if int(mj.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)]


def _read_damping(sim: str, env):
    """Per-env passive dof_damping over the actuated joints. (num_envs, n_joint)."""
    import torch

    sm = env.scene_manager
    if sim == "genesis":
        d = torch.as_tensor(sm["robot"].get_dofs_damping()).detach().float().cpu()
        if d.dim() == 1:
            d = d.unsqueeze(0)
        cols = torch.as_tensor(env.act_manager.indexing.sim_indices).long().cpu()
        return d[:, cols]

    if sim == "newton":
        import warp as wp

        mj = sm.solver.mj_model
        nv = int(mj.nv)
        d = wp.to_torch(sm.solver.mjw_model.dof_damping).detach().float().cpu().reshape(-1, nv)
        return d[:, _hinge_dofadrs(mj)]

    # mujoco / mjlab
    mj = sm.mj_model
    nv = int(mj.nv)
    d = torch.as_tensor(sm.model.dof_damping).detach().float().cpu().reshape(-1, nv)
    return d[:, _hinge_dofadrs(mj)]


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
        d = _read_damping(sim, env).float().cpu()  # (num_envs, n_joint)
        out["n_joint"] = d.shape[1]
        per_dof_std = d.std(dim=0)  # spread across envs, per joint
        rec = {
            "stats": _stats(d),
            "in_band_frac": float(((d >= lo - _TOL) & (d <= hi + _TOL)).float().mean()),
            "max_per_env_std": float(per_dof_std.max()),
            "min_per_env_std": float(per_dof_std.min()),
        }
        if prev is not None:
            rec["changed_vs_prev_frac"] = float((d != prev).float().mean())
        out["resets"].append(rec)
        prev = d
        env.reset()
    _stage(f"cell done: {sim}")
    return out


def _print_cell(r: dict) -> None:
    lo, hi = _EXPECTED
    print(f"\n===== {r['sim'].upper()} (num_envs={r['num_envs']}, {r['n_joint']} actuated joints) =====")
    print(f"  expected dof_damping band: [{lo}, {hi}]  (XML baseline was leg 3 / arm 2)")
    ok_band = ok_spread = ok_resample = True
    for i, rec in enumerate(r["resets"]):
        s = rec["stats"]
        line = (
            f"  reset {i}: damp min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} "
            f"std={s['std']:.4f}  in-band={rec['in_band_frac'] * 100:.1f}%  "
            f"per-env-std[min={rec['min_per_env_std']:.4f} max={rec['max_per_env_std']:.4f}]"
        )
        if "changed_vs_prev_frac" in rec:
            line += f"  changed_vs_prev={rec['changed_vs_prev_frac'] * 100:.1f}%"
            ok_resample = ok_resample and rec["changed_vs_prev_frac"] > 0.5
        print(line)
        ok_band = ok_band and rec["in_band_frac"] > 0.99 and s["max"] <= hi + _TOL
        ok_spread = ok_spread and rec["max_per_env_std"] > 1e-3
    print(
        f"  → overridden to [0,1] (no 3/2 left): {ok_band} | randomized per-env: {ok_spread} "
        f"| re-applied each reset: {ok_resample}"
    )
    print(f"  VERDICT: {'PASS' if (ok_band and ok_spread and ok_resample) else 'CHECK'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 passive-damping DR landing / reset re-apply diag.")
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
