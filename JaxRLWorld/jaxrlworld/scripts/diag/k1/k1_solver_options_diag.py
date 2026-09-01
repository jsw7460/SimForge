"""K1 solver-option provenance: what integrator / timestep / impratio / cone /
iterations / contact-solref does the RUNNING sim ACTUALLY use, and where do
those values come from?

Three candidate sources disagree:

  * Doc2 = our MJCF ``k1_mjx_feetonly.xml`` <option>: Euler, eulerdamp off,
    timestep 0.002, iterations 3 / ls 5, pyramidal, impratio 1 (defaults).
  * Doc1 = Booster's ``K1_22dof.xml`` <option>: Newton, tol 1e-6,
    timestep 0.001, impratio 10, geom solref 0.001 1.
  * CONFIG = jaxrlworld's mjlab solver recipe (``_mujoco_builders`` mjlab_sim_cfg):
    implicitfast, timestep = substep_dt, impratio 100, elliptic, iterations
    100 / ls 50 — which mjlab writes straight onto ``model.opt`` at build
    (Mjlab/sim.py), OVERRIDING whatever the XML <option> said.

So neither XML's numbers are what runs. This diag reads the LIVE compiled model
the solver steps (host ``mj_model.opt`` + the mjwarp model) per mjwarp backend
and prints the actual value of each option next to all three candidates, so you
can see exactly which one won. It also dumps the per-geom ``solref`` / ``solimp``
(foot spheres + ground plane), the contact-stiffness knobs neither the mjlab
solver recipe nor this preset overrides — those DO still come from the XML.

Genesis is a different solver (no MuJoCo <option>); its RigidOptions are printed
separately for reference.

Run::

    jaxpy -m jaxrlworld.scripts.diag.k1.k1_solver_options_diag --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.k1.k1_solver_options_diag            # all three
"""

from __future__ import annotations

import argparse

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}

# Candidate option sets (— = not pinned by that source).
_DOC2_XML = {
    "integrator": "euler",
    "timestep": 0.002,
    "impratio": 1.0,
    "cone": "pyramidal",
    "iterations": 3,
    "ls_iterations": 5,
}
_DOC1_BOOSTER = {
    "integrator": "—",
    "timestep": 0.001,
    "impratio": 10.0,
    "cone": "—",
    "iterations": "—",
    "ls_iterations": "—",
}
_CONFIG = {
    "integrator": "implicitfast",
    "timestep": "substep_dt",
    "impratio": 100.0,
    "cone": "elliptic",
    "iterations": 100,
    "ls_iterations": 50,
}

_INTEGRATOR = {0: "euler", 1: "rk4", 2: "implicit", 3: "implicitfast"}
_CONE = {0: "pyramidal", 1: "elliptic"}
_SOLVER = {0: "pgs", 1: "cg", 2: "newton"}


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _read_opt(mj) -> dict:
    """Solver options off a host mujoco MjModel."""
    o = mj.opt
    return {
        "integrator": _INTEGRATOR.get(int(o.integrator), f"?{int(o.integrator)}"),
        "solver": _SOLVER.get(int(o.solver), f"?{int(o.solver)}"),
        "timestep": float(o.timestep),
        "impratio": float(o.impratio),
        "cone": _CONE.get(int(o.cone), f"?{int(o.cone)}"),
        "iterations": int(o.iterations),
        "ls_iterations": int(o.ls_iterations),
        "tolerance": float(o.tolerance),
    }


def _read_geom_contact(mj) -> dict:
    """Per-geom solref / solimp for the foot spheres and the ground plane."""
    import mujoco
    import numpy as np

    solref = np.asarray(mj.geom_solref)  # (ngeom, 2 [or mjNREF])
    solimp = np.asarray(mj.geom_solimp)  # (ngeom, mjNIMP)
    rows = []
    for g in range(mj.ngeom):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        low = name.lower()
        if ("foot" in low) or ("floor" in low) or ("ground" in low) or ("plane" in low):
            rows.append((name, solref[g].tolist(), solimp[g].tolist()))
    # distinct solref values across ALL geoms (to spot a global default)
    uniq = sorted({tuple(round(x, 5) for x in solref[g]) for g in range(mj.ngeom)})
    return {"rows": rows[:8], "uniq_solref": uniq}


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim}")

    from jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    preset = K1G1RecipeConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    cfgs = preset.build()
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()
    sm = env.scene_manager
    out: dict = {"sim": sim}

    if sim == "mujoco":
        out["host_opt"] = _read_opt(sm.mj_model)
        out["contact"] = _read_geom_contact(sm.mj_model)
    elif sim == "newton":
        out["host_opt"] = _read_opt(sm.solver.mj_model)
        out["contact"] = _read_geom_contact(sm.solver.mj_model)
    else:  # genesis — different solver
        ro = getattr(sm.scene.sim, "rigid_solver", None)
        opts = getattr(getattr(sm, "scene", None), "_sim_options", None)
        out["genesis_note"] = (
            "Genesis uses its own RigidOptions (integrator=approximate_implicitfast, "
            "constraint_timeconst, elliptic cone, signorini) — not a MuJoCo <option>. "
            f"rigid_solver={type(ro).__name__ if ro else 'n/a'}"
        )
    _stage(f"cell done: {sim}")
    return out


def _cmp(actual, cand) -> str:
    if cand == "—":
        return " "
    if isinstance(cand, str):
        return "✓" if str(actual) == cand else "✗"
    try:
        return "✓" if abs(float(actual) - float(cand)) <= 1e-9 else "✗"
    except (TypeError, ValueError):
        return "?"


def _print_cell(r: dict) -> None:
    sim = r["sim"]
    print(f"\n===== {sim.upper()} =====")
    if "genesis_note" in r:
        print(f"  {r['genesis_note']}")
        return

    opt = r["host_opt"]
    print(f"  {'option':14}{'ACTUAL':>16}   {'Doc2(ours xml)':>16} {'Doc1(booster)':>16} {'CONFIG':>16}")
    for key in ("integrator", "timestep", "impratio", "cone", "iterations", "ls_iterations"):
        a = opt[key]
        d2, d1, cf = _DOC2_XML[key], _DOC1_BOOSTER[key], _CONFIG[key]
        print(
            f"  {key:14}{str(a):>16}   {str(d2):>13}{_cmp(a, d2):>3} {str(d1):>13}{_cmp(a, d1):>3} {str(cf):>13}{_cmp(a, cf):>3}"
        )
    print(f"  {'solver':14}{opt['solver']:>16}   (mjwarp always Newton internally)")
    print(f"  {'tolerance':14}{opt['tolerance']:>16.2e}")

    match = (
        "CONFIG override"
        if opt["integrator"] == "implicitfast" and opt["impratio"] == 100.0
        else ("Doc2 XML" if opt["integrator"] == "euler" else "?")
    )
    print(f"  → running solver options come from: {match}")

    c = r["contact"]
    print("  geom solref/solimp (foot + ground; NOT overridden — from XML):")
    for name, sr, si in c["rows"]:
        print(f"    {name:28} solref={sr}  solimp={[round(x, 3) for x in si]}")
    print(f"  distinct solref across all geoms: {c['uniq_solref']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 running solver-option provenance diag.")
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
    print()
    return 0 if len(results) == len(sims) else 1


if __name__ == "__main__":
    raise SystemExit(main())
