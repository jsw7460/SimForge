"""Dump joint friction / damping / mass and solver settings for mjlab T1 getup."""
from __future__ import annotations

import mujoco

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = T1GetupConfig(sim_type="mujoco", num_envs=4).build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    mj = env.scene_manager.mj_model

    print("\n=== Mjlab T1 — joint friction / damping ===")
    print(f"{'Idx':<4} {'Joint':<30} {'friction':>12} {'damping':>12} {'armature':>12}")
    for j in range(mj.njnt):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) or "?"
        dof_start = int(mj.jnt_dofadr[j])
        f = float(mj.dof_frictionloss[dof_start])
        d = float(mj.dof_damping[dof_start])
        a = float(mj.dof_armature[dof_start])
        print(f"{j:<4} {name:<30} {f:>12.5f} {d:>12.5f} {a:>12.5f}")

    print("\n=== Mjlab T1 — body masses ===")
    total = 0.0
    for b in range(mj.nbody):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b) or "?"
        mass = float(mj.body_mass[b])
        total += mass
        print(f"  {b:<3} {name:<30} {mass:>10.4f} kg")
    print(f"  TOTAL: {total:.4f} kg")

    print("\n=== Mjlab T1 — solver / timing ===")
    # MjModel option attrs
    print(f"  timestep (dt): {mj.opt.timestep}")
    print(f"  integrator: {mj.opt.integrator}")
    print(f"  solver: {mj.opt.solver}")
    print(f"  iterations: {mj.opt.iterations}")
    print(f"  ls_iterations: {mj.opt.ls_iterations}")
    print(f"  tolerance: {mj.opt.tolerance}")
    print(f"  ls_tolerance: {mj.opt.ls_tolerance}")
    print(f"  impratio: {mj.opt.impratio}")
    print(f"  cone: {mj.opt.cone}")
    print(f"  jacobian: {mj.opt.jacobian}")
    print(f"  gravity: {mj.opt.gravity}")
    sm = env.scene_manager
    cfg = sm.config
    print(f"  [rlworld cfg] physics_dt: {cfg.physics_dt}")
    print(f"  [rlworld cfg] substeps: {cfg.substeps}")


if __name__ == "__main__":
    main()
