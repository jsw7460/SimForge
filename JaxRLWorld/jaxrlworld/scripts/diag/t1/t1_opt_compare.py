"""Dump mj_model.opt settings for Newton and Mjlab, side-by-side.

Usage:
    uv run python JaxRLWorld/jaxrlworld/scripts/diag/t1_opt_compare.py --sim newton
    uv run python JaxRLWorld/jaxrlworld/scripts/diag/t1_opt_compare.py --sim mujoco

Compare the two outputs line-by-line to spot which MuJoCo option differs.
"""

from __future__ import annotations

import argparse

import mujoco

from jaxrlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from jaxrlworld.rl.runners import BaseRunner

OPT_ATTRS = (
    "timestep",
    "gravity",
    "integrator",
    "solver",
    "iterations",
    "ls_iterations",
    "tolerance",
    "ls_tolerance",
    "impratio",
    "cone",
    "jacobian",
    "wind",
    "density",
    "viscosity",
    "noslip_iterations",
    "noslip_tolerance",
    "ccd_iterations",
    "ccd_tolerance",
    "sdf_iterations",
    "sdf_initpoints",
    "o_margin",
    "o_solref",
    "o_solimp",
    "magnetic",
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
        v = getattr(mj.opt, attr)  # dynamic-attr: option named in the OPT_ATTRS table
        print(f"  {attr:<22} = {v}")

    # Newton stepping uses mujoco-warp's mjw_model (GPU), not mj_model (CPU).
    # mj_model.opt.timestep is the MJCF compile-time default (unused); the
    # effective timestep is in mjw_model.opt.timestep, set every step by
    # SolverMuJoCo.step(..., dt) → mjw_model.opt.timestep.fill_(dt).
    # The CPU solver path builds no mujoco-warp model.
    if args.sim == "newton" and not env.scene_manager.config.solver_cfg.use_mujoco_cpu:
        mjw = env.scene_manager.solver.mjw_model
        print("\n=== [newton] mjw_model.opt (effective, GPU-side) ===")
        print(f"  timestep (mjw_model)   = {mjw.opt.timestep.numpy()}")
        print(f"  iterations             = {mjw.opt.iterations}")
        print(f"  ls_iterations          = {mjw.opt.ls_iterations}")
        print(f"  impratio               = {mjw.opt.impratio}")
        print(f"  cone                   = {mjw.opt.cone}")

    # Enum name lookups for integrator/solver/cone/jacobian (they print as ints)
    print("\n=== Enum interpretation ===")
    enums = {
        "integrator": mujoco.mjtIntegrator,
        "solver": mujoco.mjtSolver,
        "cone": mujoco.mjtCone,
        "jacobian": mujoco.mjtJacobian,
    }
    for attr, enum_cls in enums.items():
        v = int(getattr(mj.opt, attr))  # dynamic-attr: enum-valued option named in the table
        print(f"  {attr:<22} = {v} ({enum_cls(v).name})")


if __name__ == "__main__":
    main()
