"""Diagnose actual PD gains / effort limits applied by mjlab for T1 getup.

Builds the T1GetupConfig(sim_type='mujoco') env and inspects the
compiled MjModel's actuator arrays. For mjlab's <position> actuator
(gaintype=FIXED, biastype=AFFINE) the effective PD law is:

    force = gainprm[0] * ctrl + biasprm[1] * qpos + biasprm[2] * qvel
          = stiffness*(ctrl - qpos) - damping*qvel        (when biasprm[1] = -gainprm[0])

So prints: gainprm[0] (= kp), -biasprm[2] (= kd), the actuator's
forcerange, plus the joint's armature for comparison with T1Config.

Usage:
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_mujoco_pd.py
"""
from __future__ import annotations

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = T1GetupConfig(sim_type="mujoco", num_envs=4).build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    mj = env.scene_manager.mj_model

    import mujoco

    print("\n=== Mjlab T1 — actual actuator PD / effort values ===")
    print(
        f"{'ActIdx':<7} {'Actuator':<30} {'Joint':<30} "
        f"{'kp':>10} {'kd':>10} {'frc_lo':>10} {'frc_hi':>10} {'arm':>10}"
    )
    print("-" * 120)

    for a in range(mj.nu):
        act_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "?"
        jnt_id = int(mj.actuator_trnid[a, 0])
        jnt_name = (
            mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or "?"
        ) if jnt_id >= 0 else "(non-joint)"

        kp = float(mj.actuator_gainprm[a, 0])
        kd = -float(mj.actuator_biasprm[a, 2])  # biasprm[2] = -damping
        frc_lo = float(mj.actuator_forcerange[a, 0])
        frc_hi = float(mj.actuator_forcerange[a, 1])

        # DOF-level armature for the target joint (mj.jnt_dofadr maps joint to DOF start)
        if jnt_id >= 0:
            dof_start = int(mj.jnt_dofadr[jnt_id])
            armature = float(mj.dof_armature[dof_start])
        else:
            armature = float("nan")

        print(
            f"{a:<7} {act_name:<30} {jnt_name:<30} "
            f"{kp:>10.3f} {kd:>10.3f} {frc_lo:>10.2f} {frc_hi:>10.2f} "
            f"{armature:>10.5f}"
        )


if __name__ == "__main__":
    main()
