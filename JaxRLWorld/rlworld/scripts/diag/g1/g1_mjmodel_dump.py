"""Dump the *native* MjModel used by each sim, in a deterministic canonical order.

Both sims expose a ``mujoco.MjModel`` instance:
    mujoco  : env.scene_manager.mj_model          (compiled from MjSpec by mjlab)
    newton  : env.scene_manager.solver.mj_model   (built by SolverMuJoCo from newton.Model)

Because both are the same Python type, we can print every attribute with
identical formatting on both sides — so ``diff -u mujoco.txt newton.txt``
reveals *every* compiled physics parameter that disagrees between the two
builds (dof_armature / dof_solref / dof_solimp / body_inertia / actuator_*,
opt.* flags, etc).

We sort per-object rows by leaf name so that mjlab's ``robot/foo`` prefix and
newton's unprefixed ``foo`` line up row-for-row.

Usage::

    jaxpy JaxRLWorld/rlworld/scripts/diag/g1_mjmodel_dump.py \\
        --policy_path ./outputs/.../checkpoint_latest \\
        --eval_sim mujoco --out /tmp/g1_mjm_mujoco.txt

    jaxpy JaxRLWorld/rlworld/scripts/diag/g1_mjmodel_dump.py \\
        --policy_path ./outputs/.../checkpoint_latest \\
        --eval_sim newton --out /tmp/g1_mjm_newton.txt

    diff -u /tmp/g1_mjm_mujoco.txt /tmp/g1_mjm_newton.txt
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import numpy as np

from rlworld.rl.evals.evaluator import PolicyEvaluator


def _fmt(x, w: int = 13, prec: int = 6) -> str:
    if x is None:
        return f"{'nan':>{w}}"
    try:
        f = float(x)
    except Exception:
        return f"{str(x):>{w}}"
    if np.isnan(f):
        return f"{'nan':>{w}}"
    if np.isinf(f):
        return f"{('+inf' if f > 0 else '-inf'):>{w}}"
    return f"{f:>+{w}.{prec}g}"


def _fmt_vec(v, prec: int = 6) -> str:
    return "[" + ", ".join(_fmt(x, w=0, prec=prec).strip() for x in v) + "]"


def _leaf(name: str) -> str:
    """Strip mjlab's ``robot/`` prefix so both sides sort identically."""
    return name.rsplit("/", 1)[-1] if name else name


def _names(mj: mujoco.MjModel, obj_type) -> list[str]:
    n = {
        mujoco.mjtObj.mjOBJ_BODY: mj.nbody,
        mujoco.mjtObj.mjOBJ_JOINT: mj.njnt,
        mujoco.mjtObj.mjOBJ_ACTUATOR: mj.nu,
        mujoco.mjtObj.mjOBJ_GEOM: mj.ngeom,
        mujoco.mjtObj.mjOBJ_SITE: mj.nsite,
    }[obj_type]
    return [mujoco.mj_id2name(mj, obj_type, i) or f"<{obj_type}_{i}>" for i in range(n)]


# ── opt ────────────────────────────────────────────────────────────────
def dump_opt(mj: mujoco.MjModel) -> None:
    o = mj.opt
    print("[OPT]")
    # Integrator / solver
    print(f"  timestep              = {_fmt(o.timestep)}")
    print(f"  integrator            = {int(o.integrator)}  (0=EULER 1=RK4 2=IMPLICIT 3=IMPLICITFAST)")
    print(f"  solver                = {int(o.solver)}      (0=PGS 1=CG 2=NEWTON)")
    print(f"  iterations            = {int(o.iterations)}")
    print(f"  ls_iterations         = {int(o.ls_iterations)}")
    print(f"  noslip_iterations     = {int(o.noslip_iterations)}")
    print(f"  ccd_iterations        = {int(o.ccd_iterations)}")
    print(f"  tolerance             = {_fmt(o.tolerance)}")
    print(f"  ls_tolerance          = {_fmt(o.ls_tolerance)}")
    print(f"  noslip_tolerance      = {_fmt(o.noslip_tolerance)}")
    print(f"  ccd_tolerance         = {_fmt(o.ccd_tolerance)}")
    print(f"  impratio              = {_fmt(o.impratio)}")
    print(f"  jacobian              = {int(o.jacobian)}    (0=DENSE 1=SPARSE 2=AUTO)")
    print(f"  cone                  = {int(o.cone)}        (0=PYRAMIDAL 1=ELLIPTIC)")
    print(f"  disableflags          = 0x{int(o.disableflags):x}")
    print(f"  enableflags           = 0x{int(o.enableflags):x}")
    # Environment / defaults
    print(f"  gravity               = {_fmt_vec(o.gravity)}")
    print(f"  wind                  = {_fmt_vec(o.wind)}")
    print(f"  magnetic              = {_fmt_vec(o.magnetic)}")
    print(f"  density               = {_fmt(o.density)}")
    print(f"  viscosity             = {_fmt(o.viscosity)}")
    # Default contact override (what mjwarp falls back to when geom_* are 0)
    print(f"  o_margin              = {_fmt(o.o_margin)}")
    print(f"  o_solref              = {_fmt_vec(o.o_solref)}")
    print(f"  o_solimp              = {_fmt_vec(o.o_solimp)}")
    print(f"  o_friction            = {_fmt_vec(o.o_friction)}")


def dump_sizes(mj: mujoco.MjModel) -> None:
    print("\n[SIZES]")
    for k in (
        "nq",
        "nv",
        "nu",
        "na",
        "nbody",
        "njnt",
        "ngeom",
        "nsite",
        "ncam",
        "nlight",
        "nmesh",
        "neq",
        "ntendon",
        "nsensor",
        "nnumeric",
        "ntext",
        "ntuple",
        "nkey",
        "nmat",
        "nexclude",
        "npair",
    ):
        print(f"  {k:<10}= {getattr(mj, k)}")


# ── per-joint ──────────────────────────────────────────────────────────
def dump_joints(mj: mujoco.MjModel) -> None:
    print("\n[JOINTS — sorted by leaf name]")
    header = (
        f"{'leaf_name':<32} {'type':<6} {'bodyid':>6} {'dofadr':>6} {'qposadr':>7} "
        f"{'range_lo':>11} {'range_hi':>11} {'stiffness':>11} {'margin':>11} "
        f"{'solref':>26} {'solimp':>36}"
    )
    print(header)
    print("-" * len(header))

    names = _names(mj, mujoco.mjtObj.mjOBJ_JOINT)
    order = sorted(range(mj.njnt), key=lambda i: _leaf(names[i]))
    for j in order:
        leaf = _leaf(names[j])
        typ = int(mj.jnt_type[j])  # 0=FREE 1=BALL 2=SLIDE 3=HINGE
        print(
            f"{leaf:<32} {typ:<6} "
            f"{int(mj.jnt_bodyid[j]):>6} {int(mj.jnt_dofadr[j]):>6} "
            f"{int(mj.jnt_qposadr[j]):>7} "
            f"{_fmt(mj.jnt_range[j, 0])} {_fmt(mj.jnt_range[j, 1])} "
            f"{_fmt(mj.jnt_stiffness[j])} {_fmt(mj.jnt_margin[j])} "
            f"{_fmt_vec(mj.jnt_solref[j]):>26} {_fmt_vec(mj.jnt_solimp[j]):>36}"
        )

    # Also dump axis/pos so a mis-oriented joint jumps out.
    print("\n[JOINT AXIS + POS — sorted by leaf name]")
    header = f"{'leaf_name':<32} {'axis':<40} {'pos':<40}"
    print(header)
    print("-" * len(header))
    for j in order:
        leaf = _leaf(names[j])
        print(f"{leaf:<32} {_fmt_vec(mj.jnt_axis[j]):<40} {_fmt_vec(mj.jnt_pos[j]):<40}")


# ── per-dof ────────────────────────────────────────────────────────────
def dump_dofs(mj: mujoco.MjModel) -> None:
    print("\n[DOFS — MuJoCo index order (use joint_name for alignment)]")
    # Build dof -> joint_leaf_name map
    jnames = _names(mj, mujoco.mjtObj.mjOBJ_JOINT)
    dof2leaf = [""] * mj.nv
    for j in range(mj.njnt):
        start = int(mj.jnt_dofadr[j])
        typ = int(mj.jnt_type[j])
        # FREE=6dof, BALL=3dof, SLIDE=1dof, HINGE=1dof
        width = {0: 6, 1: 3, 2: 1, 3: 1}.get(typ, 1)
        for k in range(width):
            dof2leaf[start + k] = f"{_leaf(jnames[j])}[{k}]"

    header = (
        f"{'dof_leaf':<36} {'armature':>11} {'damping':>11} "
        f"{'frictionloss':>13} {'invweight0':>11} {'M0':>11} "
        f"{'solref':>26} {'solimp':>36}"
    )
    print(header)
    print("-" * len(header))

    # Sort dofs by leaf label to align across sims.
    order = sorted(range(mj.nv), key=lambda d: dof2leaf[d])
    for d in order:
        print(
            f"{dof2leaf[d]:<36} "
            f"{_fmt(mj.dof_armature[d])} {_fmt(mj.dof_damping[d])} "
            f"{_fmt(mj.dof_frictionloss[d])} "
            f"{_fmt(mj.dof_invweight0[d])} {_fmt(mj.dof_M0[d])} "
            f"{_fmt_vec(mj.dof_solref[d]):>26} "
            f"{_fmt_vec(mj.dof_solimp[d]):>36}"
        )


# ── per-body ───────────────────────────────────────────────────────────
def dump_bodies(mj: mujoco.MjModel) -> None:
    print("\n[BODIES — sorted by leaf name]")
    header = (
        f"{'leaf_name':<36} {'mass':>11} {'ixx':>11} {'iyy':>11} {'izz':>11} "
        f"{'ipos':<30} {'iquat':<36} {'pos':<30} {'quat':<36}"
    )
    print(header)
    print("-" * len(header))
    names = _names(mj, mujoco.mjtObj.mjOBJ_BODY)
    order = sorted(range(mj.nbody), key=lambda i: _leaf(names[i]))
    for b in order:
        leaf = _leaf(names[b])
        I = mj.body_inertia[b]
        print(
            f"{leaf:<36} {_fmt(mj.body_mass[b])} "
            f"{_fmt(I[0])} {_fmt(I[1])} {_fmt(I[2])} "
            f"{_fmt_vec(mj.body_ipos[b]):<30} "
            f"{_fmt_vec(mj.body_iquat[b]):<36} "
            f"{_fmt_vec(mj.body_pos[b]):<30} "
            f"{_fmt_vec(mj.body_quat[b]):<36}"
        )


# ── per-actuator ───────────────────────────────────────────────────────
def dump_actuators(mj: mujoco.MjModel) -> None:
    print("\n[ACTUATORS — sorted by leaf name]")
    header = (
        f"{'leaf_name':<32} {'trntype':>7} {'gaintype':>8} {'biastype':>8} "
        f"{'dyntype':>7} {'gear0':>11} {'gain0':>11} {'bias2':>11} "
        f"{'frclim':>6} {'frc_lo':>11} {'frc_hi':>11} "
        f"{'ctrllim':>7} {'ctrl_lo':>11} {'ctrl_hi':>11}"
    )
    print(header)
    print("-" * len(header))
    names = _names(mj, mujoco.mjtObj.mjOBJ_ACTUATOR)
    order = sorted(range(mj.nu), key=lambda i: _leaf(names[i]))
    for a in order:
        leaf = _leaf(names[a])
        print(
            f"{leaf:<32} {int(mj.actuator_trntype[a]):>7} "
            f"{int(mj.actuator_gaintype[a]):>8} "
            f"{int(mj.actuator_biastype[a]):>8} "
            f"{int(mj.actuator_dyntype[a]):>7} "
            f"{_fmt(mj.actuator_gear[a, 0])} "
            f"{_fmt(mj.actuator_gainprm[a, 0])} "
            f"{_fmt(mj.actuator_biasprm[a, 2])} "
            f"{int(mj.actuator_forcelimited[a]):>6} "
            f"{_fmt(mj.actuator_forcerange[a, 0])} "
            f"{_fmt(mj.actuator_forcerange[a, 1])} "
            f"{int(mj.actuator_ctrllimited[a]):>7} "
            f"{_fmt(mj.actuator_ctrlrange[a, 0])} "
            f"{_fmt(mj.actuator_ctrlrange[a, 1])}"
        )

    # Full gainprm/biasprm/dynprm per actuator (10 values each).
    print("\n[ACTUATOR FULL PRM — sorted by leaf name]")
    for a in order:
        leaf = _leaf(names[a])
        print(
            f"{leaf:<32} gain={_fmt_vec(mj.actuator_gainprm[a])} "
            f"bias={_fmt_vec(mj.actuator_biasprm[a])} "
            f"dyn={_fmt_vec(mj.actuator_dynprm[a])}"
        )


# ── per-geom (collision-relevant) ──────────────────────────────────────
def dump_geoms(mj: mujoco.MjModel) -> None:
    print("\n[GEOMS — sorted by leaf name]")
    header = (
        f"{'leaf_name':<36} {'type':>5} {'bodyid':>6} {'condim':>6} "
        f"{'priority':>8} {'contype':>7} {'conaff':>6} "
        f"{'friction':<28} {'solref':>26} {'solimp':>36} "
        f"{'margin':>11} {'gap':>11}"
    )
    print(header)
    print("-" * len(header))
    names = _names(mj, mujoco.mjtObj.mjOBJ_GEOM)
    order = sorted(range(mj.ngeom), key=lambda i: _leaf(names[i]))
    for g in order:
        leaf = _leaf(names[g])
        print(
            f"{leaf:<36} {int(mj.geom_type[g]):>5} {int(mj.geom_bodyid[g]):>6} "
            f"{int(mj.geom_condim[g]):>6} {int(mj.geom_priority[g]):>8} "
            f"{int(mj.geom_contype[g]):>7} {int(mj.geom_conaffinity[g]):>6} "
            f"{_fmt_vec(mj.geom_friction[g]):<28} "
            f"{_fmt_vec(mj.geom_solref[g]):>26} "
            f"{_fmt_vec(mj.geom_solimp[g]):>36} "
            f"{_fmt(mj.geom_margin[g])} {_fmt(mj.geom_gap[g])}"
        )


# ── main ───────────────────────────────────────────────────────────────
def _resolve_mj_model(env, eval_sim: str) -> mujoco.MjModel:
    if eval_sim == "mujoco":
        return env.scene_manager.mj_model
    if eval_sim == "newton":
        solver = env.scene_manager.solver
        if solver is None:
            raise RuntimeError("Newton scene_manager.solver is None")
        if not hasattr(solver, "mj_model") or solver.mj_model is None:
            raise RuntimeError(
                "Newton solver does not expose mj_model — expected SolverMuJoCo with use_mujoco_contacts=True"
            )
        return solver.mj_model
    raise ValueError(f"unknown eval_sim: {eval_sim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--eval_sim", required=True, choices=("newton", "mujoco"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or f"g1_mjm_{args.eval_sim}.txt"
    log_fh = open(out_path, "w")
    orig_stdout = sys.stdout
    sys.stdout = log_fh

    print(f"# g1_mjmodel_dump sim={args.eval_sim} policy={args.policy_path}")

    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=args.eval_sim,
        extra_overrides={"env": {"num_envs": 1}},
        record_video=False,
    )
    mj = _resolve_mj_model(evaluator.env, args.eval_sim)

    dump_opt(mj)
    dump_sizes(mj)
    dump_joints(mj)
    dump_dofs(mj)
    dump_bodies(mj)
    dump_actuators(mj)
    dump_geoms(mj)

    sys.stdout = orig_stdout
    log_fh.close()
    print(f"[g1_mjmodel_dump] wrote {out_path}")


if __name__ == "__main__":
    main()
