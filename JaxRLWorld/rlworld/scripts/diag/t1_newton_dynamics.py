"""Dump joint friction / damping / mass and solver settings for Newton T1 getup."""
from __future__ import annotations

import warp as wp

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.envs.utils.newton.label import leaf_name
from rlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = T1GetupConfig(sim_type="newton", num_envs=4).build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    m = env.scene_manager.model

    num_worlds = m.world_count
    joints_per_world = len(m.joint_label) // num_worlds
    bodies_per_world = len(m.body_label) // num_worlds

    qd_start = wp.to_torch(m.joint_qd_start).cpu().numpy()
    jfric = wp.to_torch(m.joint_friction).cpu().numpy() if hasattr(m, "joint_friction") else None
    jdamp = (
        wp.to_torch(m.joint_damping).cpu().numpy()
        if hasattr(m, "joint_damping") else None
    )
    body_mass = wp.to_torch(m.body_mass).cpu().numpy()

    print("\n=== Newton T1 — joint friction / damping (world 0) ===")
    print(f"{'Idx':<4} {'Joint':<30} {'friction':>12} {'damping':>12}")
    for j in range(joints_per_world):
        name = leaf_name(m.joint_label[j])
        d0 = int(qd_start[j])
        f = jfric[d0] if jfric is not None else float('nan')
        d = jdamp[d0] if jdamp is not None else float('nan')
        print(f"{j:<4} {name:<30} {f:>12.5f} {d:>12.5f}")

    print("\n=== Newton T1 — body masses (world 0) ===")
    total = 0.0
    for b in range(bodies_per_world):
        name = leaf_name(m.body_label[b])
        mv = float(body_mass[b])
        total += mv
        print(f"  {b:<3} {name:<30} {mv:>10.4f} kg")
    print(f"  TOTAL: {total:.4f} kg")

    print("\n=== Newton T1 — solver / timing ===")
    sm = env.scene_manager
    cfg = sm.config
    print(f"  dt (physics): {cfg.dt}")
    print(f"  substeps: {cfg.substeps}")
    print(f"  solver_type: {cfg.solver_type}")
    # Try to peek into the MuJoCo solver
    if hasattr(sm, "solver") and sm.solver is not None:
        s = sm.solver
        for attr in ("njmax", "nconmax", "impratio", "iterations", "ls_iterations",
                     "ccd_iterations", "use_mujoco_contacts", "ls_parallel"):
            v = getattr(s, attr, None)
            if v is not None:
                print(f"  solver.{attr}: {v}")


if __name__ == "__main__":
    main()
