"""K1 foot-friction DR: does the randomized ground friction actually LAND in
the solver, and is it RE-APPLIED on every reset (reset_dr)?

Answers the concrete question: on each ``env.reset()`` the ``randomize_friction``
term (mode ``reset_dr``) samples a new per-env foot friction and writes it to the
sim model — but does the value the SOLVER actually uses change, per backend? DR
has a history of silently not reaching the solver (foot mu pinned to a default;
per-env expand bugs). This reads the model's foot-geom friction directly, per
sim, across several resets:

  1. after build+reset: per-env foot mu spread matches the DR band (landed, not
     a silent no-op / single value broadcast to all envs)
  2. across 3 resets: the per-env values CHANGE (reset_dr re-samples AND the new
     value reaches the model each reset — the whole point of reset_dr)
  3. every value stays inside the configured band [expected_lo, expected_hi]

Reads WRITTEN model friction (mjwarp ``geom_friction`` / Genesis base×ratio) —
the value the contact solver reads — not just the DR term's intent. Reuses the
per-sim geom-id / conversion helpers from ``k1_contact_mu_parity_diag``.

Run per cell on the training box::

    jaxpy -m jaxrlworld.scripts.diag.k1.k1_friction_dr_diag --sim newton
    jaxpy -m jaxrlworld.scripts.diag.k1.k1_friction_dr_diag            # all three
"""

from __future__ import annotations

import argparse

from jaxrlworld.scripts.diag.k1.k1_contact_mu_parity_diag import _host_geom_ids

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
# Expected effective foot-ground friction band (matches base.py randomize_friction).
_EXPECTED = (0.9, 1.2)
_TOL = 0.02  # allow small solver-side rounding at the band edges
_N_RESETS = 3


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _stats(t) -> dict:
    t = t.detach().float().reshape(-1)
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
    }


def _read_foot_mu(sim: str, env, foot_names):
    """Per-env foot-geom friction the SOLVER uses. Shape (num_envs, n_foot)."""
    import torch

    sm = env.scene_manager
    if sim == "genesis":
        from genesis.utils.misc import qd_to_torch

        solver = sm.scene.sim.rigid_solver
        robot = sm["robot"]
        foot_geoms = []
        for link in robot.links:
            if any(fn in link.name for fn in foot_names):
                foot_geoms.extend(range(link.geom_start, link.geom_end))
        if not foot_geoms:
            raise RuntimeError(f"no genesis foot geoms matched {foot_names}")
        base = qd_to_torch(solver.dyn_info.geoms.friction, copy=True)[foot_geoms]
        ratio = solver.get_geoms_friction_ratio(geoms_idx=foot_geoms)  # (B, n_foot)
        return base.unsqueeze(0).to(ratio.device) * ratio

    if sim == "newton":
        import warp as wp

        solver = sm.solver
        foot_gids, _ = _host_geom_ids(solver.mj_model, list(foot_names))
        gf = wp.to_torch(solver.mjw_model.geom_friction)  # (nworld, ngeom, 3)
        return gf[:, sorted(foot_gids), 0]

    # mujoco / mjlab
    foot_gids, _ = _host_geom_ids(sm.mj_model, list(foot_names))
    gf = torch.as_tensor(sm.model.geom_friction)  # (nworld, ngeom, 3)
    return gf[:, sorted(foot_gids), 0]


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs} seed={seed}")

    from jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    preset = K1G1RecipeConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    foot_names = list(preset.robot.foot_names)
    cfgs = preset.build()
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()
    _stage("env built + first reset")

    out: dict = {"sim": sim, "num_envs": num_envs, "n_foot": None, "resets": []}

    prev = None
    for k in range(_N_RESETS):
        # To CPU: reused _stats() builds its quantile q-tensor on CPU, and the
        # per-sim reads land on GPU (mjwarp / genesis).
        mu = _read_foot_mu(sim, env, foot_names).detach().float().cpu()
        out["n_foot"] = mu.shape[1] if mu.dim() > 1 else 1
        flat = mu.reshape(mu.shape[0], -1)
        per_env = flat.mean(dim=1)  # one mu per env (feet share the DR draw)
        rec = {
            "stats": _stats(mu),
            "in_band_frac": float(((mu >= _EXPECTED[0] - _TOL) & (mu <= _EXPECTED[1] + _TOL)).float().mean()),
        }
        if prev is not None:
            rec["changed_vs_prev_frac"] = float((per_env != prev).float().mean())
        out["resets"].append(rec)
        prev = per_env
        env.reset()
    _stage(f"cell done: {sim}")
    return out


def _print_cell(r: dict) -> None:
    lo, hi = _EXPECTED
    print(f"\n===== {r['sim'].upper()} (num_envs={r['num_envs']}, {r['n_foot']} foot geoms) =====")
    print(f"  expected foot-ground mu band: [{lo}, {hi}]")
    ok_band = True
    ok_resample = True
    for i, rec in enumerate(r["resets"]):
        s = rec["stats"]
        line = (
            f"  reset {i}: mu min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} "
            f"std={s['std']:.4f}  in-band={rec['in_band_frac'] * 100:.1f}%"
        )
        if "changed_vs_prev_frac" in rec:
            line += f"  changed_vs_prev={rec['changed_vs_prev_frac'] * 100:.1f}%"
            ok_resample = ok_resample and rec["changed_vs_prev_frac"] > 0.5
        print(line)
        ok_band = ok_band and rec["in_band_frac"] > 0.98 and s["std"] > 1e-4
    verdict = "PASS" if (ok_band and ok_resample) else "CHECK"
    print(f"  → landed & per-env varied: {ok_band} | re-applied each reset: {ok_resample}")
    print(f"  VERDICT: {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 foot-friction DR landing / reset re-apply diag.")
    ap.add_argument("--sim", choices=_SIMS, help="Single backend (default: all).")
    ap.add_argument("--num_envs", type=int, default=4096)
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
