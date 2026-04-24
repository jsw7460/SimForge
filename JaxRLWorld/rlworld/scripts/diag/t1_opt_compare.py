"""Dump mj_model.opt settings for Newton and Mjlab, side-by-side.

Usage:
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_opt_compare.py --sim newton
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_opt_compare.py --sim mujoco

Compare the two outputs line-by-line to spot which MuJoCo option differs.
"""
from __future__ import annotations

import argparse

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner


OPT_ATTRS = (
    "timestep", "gravity", "integrator", "solver",
    "iterations", "ls_iterations", "tolerance", "ls_tolerance",
    "impratio", "cone", "jacobian",
    "wind", "density", "viscosity",
    "noslip_iterations", "noslip_tolerance",
    "ccd_iterations", "ccd_tolerance",
    "sdf_iterations", "sdf_initpoints",
    "o_margin", "o_solref", "o_solimp",
    "apirate", "magnetic",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=("newton", "mujoco"), required=True)
    args = parser.parse_args()

    cfg = T1GetupConfig(sim_type=args.sim, num_envs=1)
    cfgs = cfg.build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env

    # Resolve mj_model: different attribute location per sim
    if args.sim == "newton":
        mj = env.scene_manager.solver.mj_model
    else:
        mj = env.scene_manager.mj_model

    print(f"\n=== [{args.sim}] mj_model.opt ===")
    for attr in OPT_ATTRS:
        try:
            v = getattr(mj.opt, attr)
        except AttributeError:
            v = "N/A"
        print(f"  {attr:<22} = {v}")

    # Newton stepping uses mujoco-warp's mjw_model (GPU), not mj_model (CPU).
    # mj_model.opt.timestep is the MJCF compile-time default (unused); the
    # effective timestep is in mjw_model.opt.timestep, set every step by
    # SolverMuJoCo.step(..., dt) → mjw_model.opt.timestep.fill_(dt).
    if args.sim == "newton":
        mjw = getattr(env.scene_manager.solver, "mjw_model", None)
        if mjw is not None:
            print("\n=== [newton] mjw_model.opt (effective, GPU-side) ===")
            ts = mjw.opt.timestep
            try:
                ts_val = ts.numpy() if hasattr(ts, "numpy") else ts
            except Exception:
                ts_val = ts
            print(f"  timestep (mjw_model)   = {ts_val}")
            print(f"  iterations             = {mjw.opt.iterations}")
            print(f"  ls_iterations          = {mjw.opt.ls_iterations}")
            print(f"  impratio               = {mjw.opt.impratio}")
            print(f"  cone                   = {mjw.opt.cone}")

    # Enum name lookups for integrator/solver/cone/jacobian (they print as ints)
    try:
        import mujoco
        print("\n=== Enum interpretation ===")
        enums = {
            "integrator": mujoco.mjtIntegrator,
            "solver":     mujoco.mjtSolver,
            "cone":       mujoco.mjtCone,
            "jacobian":   mujoco.mjtJacobian,
        }
        for attr, enum_cls in enums.items():
            try:
                v = int(getattr(mj.opt, attr))
                name = enum_cls(v).name
                print(f"  {attr:<22} = {v} ({name})")
            except Exception as e:
                print(f"  {attr:<22} = err: {e}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
