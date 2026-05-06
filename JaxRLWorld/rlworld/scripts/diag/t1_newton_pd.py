"""Diagnose actual PD gains / effort limits applied by Newton for T1 getup.

Builds the T1GetupConfig(sim_type='newton') env and prints each joint's
``joint_target_ke`` / ``joint_target_kd`` / ``joint_armature`` /
``joint_effort_limit`` as seen by the MuJoCo-Warp solver, alongside the
leaf joint name so we can compare against T1Config's
``STIFFNESS_*`` / ``EFFORT_*`` constants.

Usage:
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_newton_pd.py
"""

from __future__ import annotations

import warp as wp

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.envs.utils.newton.label import leaf_name
from rlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = T1GetupConfig(sim_type="newton", num_envs=4).build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    model = env.scene_manager.model

    num_worlds = model.world_count
    joints_per_world = len(model.joint_label) // num_worlds
    dofs_per_world = model.joint_dof_count // num_worlds

    ke = wp.to_torch(model.joint_target_ke).cpu().numpy()
    kd = wp.to_torch(model.joint_target_kd).cpu().numpy()
    arm = wp.to_torch(model.joint_armature).cpu().numpy()
    frc = wp.to_torch(model.joint_effort_limit).cpu().numpy()
    qd_start = wp.to_torch(model.joint_qd_start).cpu().numpy()

    print("\n=== Newton T1 — actual PD / effort values (world 0) ===")
    print(f"{'Idx':<4} {'Joint':<30} {'ke':>10} {'kd':>10} {'armature':>10} {'effort':>10}")
    print("-" * 78)
    for j in range(joints_per_world):
        name = leaf_name(model.joint_label[j])
        dof_start = int(qd_start[j])
        if j + 1 < joints_per_world:
            dof_count = int(qd_start[j + 1]) - dof_start
        else:
            dof_count = dofs_per_world - dof_start
        if dof_count == 0:
            continue
        # Take the first DOF's values (revolute has 1 DOF; free joint has 6)
        d0 = dof_start
        print(f"{j:<4} {name:<30} {ke[d0]:>10.3f} {kd[d0]:>10.3f} {arm[d0]:>10.5f} {frc[d0]:>10.2f}")


if __name__ == "__main__":
    main()
