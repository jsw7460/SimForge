"""Dump every compiled physics parameter G1 uses, in canonical joint order.

Both sims produce the exact same row layout so ``diff -u newton.txt mujoco.txt``
reveals every value that disagrees — PD gains, armature, joint damping /
frictionloss, effort limit, per-body mass / inertia, sim-wide dt / integrator
/ gravity, and the 6-DoF free joint parameters for the floating base.

The goal is *confirmatory*: no heuristics, no recovery, no silent fallbacks.
Attributes that don't exist on a given sim print as ``nan`` so both sides
always produce the same number of rows.

Usage::

    jaxpy JaxRLWorld/rlworld/scripts/diag/g1_physics_params_dump.py \\
        --policy_path ./outputs/.../checkpoint_latest \\
        --eval_sim mujoco --out /tmp/g1_phys_mujoco.txt

    jaxpy JaxRLWorld/rlworld/scripts/diag/g1_physics_params_dump.py \\
        --policy_path ./outputs/.../checkpoint_latest \\
        --eval_sim newton --out /tmp/g1_phys_newton.txt

    diff -u /tmp/g1_phys_mujoco.txt /tmp/g1_phys_newton.txt

Every non-empty diff line is a real physics mismatch between the two sims.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from rlworld.rl.evals.evaluator import PolicyEvaluator


_NAN = float("nan")


def _fmt(x, w: int = 11, prec: int = 6) -> str:
    if x is None:
        return f"{'nan':>{w}}"
    try:
        f = float(x)
    except Exception:
        return f"{'?':>{w}}"
    if np.isnan(f):
        return f"{'nan':>{w}}"
    if np.isinf(f):
        return f"{('+inf' if f > 0 else '-inf'):>{w}}"
    return f"{f:>+{w}.{prec}f}"


def _safe_to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    try:
        return np.asarray(x)
    except Exception:
        return None


# ── MuJoCo side ────────────────────────────────────────────────────
def dump_mujoco(env, canonical_names) -> None:
    import mujoco

    mj = env.scene_manager.mj_model

    print("[SIM META]")
    print(f"  sim_type                = mujoco")
    print(f"  physics_dt              = {float(mj.opt.timestep):.6f}")
    # mjtIntegrator: 0=EULER, 1=RK4, 2=IMPLICIT, 3=IMPLICITFAST
    print(f"  integrator (mjtInt)     = {int(mj.opt.integrator)}")
    print(f"  solver (mjtSolver)      = {int(mj.opt.solver)}")
    print(f"  iterations              = {int(mj.opt.iterations)}")
    print(f"  ls_iterations           = {int(getattr(mj.opt, 'ls_iterations', -1))}")
    print(f"  gravity                 = [{mj.opt.gravity[0]:+.4f}, "
          f"{mj.opt.gravity[1]:+.4f}, {mj.opt.gravity[2]:+.4f}]")
    print(f"  nu (actuators)          = {int(mj.nu)}")
    print(f"  njnt                    = {int(mj.njnt)}")
    print(f"  nbody                   = {int(mj.nbody)}")
    print(f"  nv (total dofs)         = {int(mj.nv)}")
    print(f"  nq (total qpos dim)     = {int(mj.nq)}")

    # Build actuator → joint_id reverse map (only transmission id=0 is the joint).
    act_by_jid = {int(mj.actuator_trnid[a, 0]): a for a in range(mj.nu)}

    # mjlab attaches entities with an entity-name prefix (``robot/``) on every
    # body + joint + actuator. Build a leaf-name → jid map that matches both
    # prefixed and raw joint names so the canonical-order lookup works.
    all_jnt_names = [
        mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for j in range(mj.njnt)
    ]
    leaf_to_jid: dict[str, int] = {}
    for j, nm in enumerate(all_jnt_names):
        if not nm:
            continue
        leaf = nm.rsplit("/", 1)[-1]
        leaf_to_jid[leaf] = j
        leaf_to_jid[nm] = j  # also accept fully-qualified

    print("\n[PER-JOINT PHYSICS — canonical order]")
    header = (
        f"{'Idx':<4} {'joint_name':<32} "
        f"{'kp':>11} {'kd':>11} {'armature':>11} {'damping':>11} "
        f"{'friction':>11} {'eff_lo':>11} {'eff_hi':>11}"
    )
    print(header)
    print("-" * len(header))

    for i, name in enumerate(canonical_names):
        jid = leaf_to_jid.get(name, -1)
        if jid < 0:
            print(f"{i:<4} {name:<32} [JOINT NOT FOUND — tried raw + robot/ prefix]")
            continue
        dof_start = int(mj.jnt_dofadr[jid])
        armature = float(mj.dof_armature[dof_start])
        damping = float(mj.dof_damping[dof_start])
        friction = float(mj.dof_frictionloss[dof_start])

        a = act_by_jid.get(jid)
        if a is not None:
            kp = float(mj.actuator_gainprm[a, 0])
            kd = -float(mj.actuator_biasprm[a, 2])
            eff_lo = float(mj.actuator_forcerange[a, 0])
            eff_hi = float(mj.actuator_forcerange[a, 1])
        else:
            kp = kd = eff_lo = eff_hi = _NAN

        print(
            f"{i:<4} {name:<32} "
            f"{_fmt(kp)} {_fmt(kd)} {_fmt(armature)} {_fmt(damping)} "
            f"{_fmt(friction)} {_fmt(eff_lo)} {_fmt(eff_hi)}"
        )

    # Root free joint — first 6 DoF on a floating-base MuJoCo model.
    print("\n[ROOT FREE JOINT — first 6 DoF]")
    for d in range(min(6, mj.nv)):
        a_v = float(mj.dof_armature[d])
        d_v = float(mj.dof_damping[d])
        f_v = float(mj.dof_frictionloss[d])
        print(f"  dof[{d}] armature={_fmt(a_v)} damping={_fmt(d_v)} "
              f"frictionloss={_fmt(f_v)}")

    # Per-body mass + inertia.
    print("\n[PER-BODY MASS + INERTIA]")
    header = (
        f"{'Idx':<4} {'body_name':<36} "
        f"{'mass[kg]':>11} {'Ixx':>11} {'Iyy':>11} {'Izz':>11}"
    )
    print(header)
    print("-" * len(header))
    for b in range(mj.nbody):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b) or "?"
        mass = float(mj.body_mass[b])
        ixx = float(mj.body_inertia[b, 0])
        iyy = float(mj.body_inertia[b, 1])
        izz = float(mj.body_inertia[b, 2])
        print(f"{b:<4} {str(name):<36} "
              f"{_fmt(mass)} {_fmt(ixx)} {_fmt(iyy)} {_fmt(izz)}")


# ── Newton side ────────────────────────────────────────────────────
def dump_newton(env, canonical_names) -> None:
    import warp as wp
    from rlworld.rl.envs.utils.newton.label import leaf_name

    model = env.scene_manager.model

    print("[SIM META]")
    print(f"  sim_type                = newton")
    print(f"  physics_dt (env)        = {float(env.physics_dt):.6f}")
    print(f"  control_dt (env)        = {float(env.control_dt):.6f}")
    print(f"  decimation              = {int(env.decimation)}")
    print(f"  num_worlds              = {int(model.world_count)}")

    # Probe whatever Newton exposes about integrator / solver / gravity.
    def _probe(attr_name: str):
        obj = getattr(model, attr_name, None)
        if obj is None:
            return None
        arr = _safe_to_numpy(obj)
        if arr is None:
            return obj
        return arr

    for a in ("gravity", "soft_contact_ke", "soft_contact_kd",
              "soft_contact_mu", "soft_contact_kf",
              "rigid_contact_margin", "rigid_contact_torsional_friction"):
        v = _probe(a)
        if v is not None:
            if isinstance(v, np.ndarray):
                v = v.tolist() if v.size <= 6 else f"<ndarray shape={v.shape}>"
            print(f"  {a:<24}= {v}")

    num_worlds = int(model.world_count)
    joints_per_world = len(model.joint_label) // num_worlds
    dofs_per_world = int(model.joint_dof_count) // num_worlds
    print(f"  joints_per_world        = {joints_per_world}")
    print(f"  dofs_per_world          = {dofs_per_world}")

    # Parameter arrays — per-world, stride of dofs_per_world.
    ke = _safe_to_numpy(model.joint_target_ke)
    kd = _safe_to_numpy(model.joint_target_kd)
    armature = _safe_to_numpy(model.joint_armature)
    effort_limit = _safe_to_numpy(model.joint_effort_limit)
    qd_start = _safe_to_numpy(model.joint_qd_start)

    # damping / frictionloss: probe both the obvious names.
    damping = _safe_to_numpy(getattr(model, "joint_damping", None))
    friction = _safe_to_numpy(getattr(model, "joint_friction", None))
    if friction is None:
        friction = _safe_to_numpy(getattr(model, "joint_frictionloss", None))

    print(f"  joint_damping exposed?  = {damping is not None}")
    print(f"  joint_friction exposed? = {friction is not None}")

    # Build leaf-name → newton joint index for world 0 only.
    name_to_jidx = {}
    for j in range(joints_per_world):
        name_to_jidx[leaf_name(model.joint_label[j])] = j

    print("\n[PER-JOINT PHYSICS — canonical order]")
    header = (
        f"{'Idx':<4} {'joint_name':<32} "
        f"{'kp':>11} {'kd':>11} {'armature':>11} {'damping':>11} "
        f"{'friction':>11} {'eff_lo':>11} {'eff_hi':>11}"
    )
    print(header)
    print("-" * len(header))

    for i, name in enumerate(canonical_names):
        j = name_to_jidx.get(name)
        if j is None:
            print(f"{i:<4} {name:<32} [JOINT NOT FOUND IN NEWTON MODEL]")
            continue
        dof_start = int(qd_start[j])
        kp_v = float(ke[dof_start])
        kd_v = float(kd[dof_start])
        arm_v = float(armature[dof_start])
        eff = float(effort_limit[dof_start])
        dmp_v = float(damping[dof_start]) if damping is not None else _NAN
        frn_v = float(friction[dof_start]) if friction is not None else _NAN
        print(
            f"{i:<4} {name:<32} "
            f"{_fmt(kp_v)} {_fmt(kd_v)} {_fmt(arm_v)} {_fmt(dmp_v)} "
            f"{_fmt(frn_v)} {_fmt(-eff)} {_fmt(eff)}"
        )

    # Root free joint — first 6 DoF (world 0).
    print("\n[ROOT FREE JOINT — first 6 DoF]")
    for d in range(min(6, dofs_per_world)):
        a_v = float(armature[d])
        d_v = float(damping[d]) if damping is not None else _NAN
        f_v = float(friction[d]) if friction is not None else _NAN
        print(f"  dof[{d}] armature={_fmt(a_v)} damping={_fmt(d_v)} "
              f"frictionloss={_fmt(f_v)}")

    # Per-body mass + inertia (world 0).
    print("\n[PER-BODY MASS + INERTIA]")
    body_label = model.body_label
    bodies_per_world = len(body_label) // num_worlds
    body_mass = _safe_to_numpy(getattr(model, "body_mass", None))
    body_inertia = _safe_to_numpy(getattr(model, "body_inertia", None))
    header = (
        f"{'Idx':<4} {'body_name':<36} "
        f"{'mass[kg]':>11} {'Ixx':>11} {'Iyy':>11} {'Izz':>11}"
    )
    print(header)
    print("-" * len(header))
    for b in range(bodies_per_world):
        name = leaf_name(body_label[b])
        mass = float(body_mass[b]) if body_mass is not None else _NAN
        ixx = iyy = izz = _NAN
        if body_inertia is not None:
            inert = body_inertia[b]
            shape = np.asarray(inert).shape
            if shape == (3, 3):
                ixx, iyy, izz = float(inert[0, 0]), float(inert[1, 1]), float(inert[2, 2])
            elif shape == (6,):
                # Common warp inertia layout: diag + off-diag.
                ixx, iyy, izz = float(inert[0]), float(inert[1]), float(inert[2])
            elif shape == (9,):
                # Flattened row-major 3x3.
                ixx, iyy, izz = float(inert[0]), float(inert[4]), float(inert[8])
        print(f"{b:<4} {name:<36} "
              f"{_fmt(mass)} {_fmt(ixx)} {_fmt(iyy)} {_fmt(izz)}")


# ── main ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--eval_sim", required=True, choices=("newton", "mujoco"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or f"g1_phys_{args.eval_sim}.txt"
    log_fh = open(out_path, "w")
    orig_stdout = sys.stdout
    sys.stdout = log_fh

    print(f"# g1_physics_params_dump sim={args.eval_sim} "
          f"policy={args.policy_path}")

    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=args.eval_sim,
        extra_overrides={"env": {"num_envs": 1}},
        record_video=False,
    )
    env = evaluator.env
    canonical_names = list(env.act_manager.actuated_joint_names)

    if args.eval_sim == "mujoco":
        dump_mujoco(env, canonical_names)
    else:
        dump_newton(env, canonical_names)

    sys.stdout = orig_stdout
    log_fh.close()
    print(f"[g1_physics_params_dump] wrote {out_path}")


if __name__ == "__main__":
    main()
