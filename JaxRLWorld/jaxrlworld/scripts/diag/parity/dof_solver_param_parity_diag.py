"""Do the three backends run the same solver and give each joint the
same DOF parameters?

The fourth question in the series. ``contact_pair_parity_diag`` and
``contact_param_parity_diag`` establish that the same geometry touches
with the same contact parameters. Two robots can pass both and still
move differently, because joint-space dynamics are set by quantities no
contact table shows: the solver configuration (timestep, iterations,
friction cone, impratio, integrator) and the per-DOF passive parameters
(damping, armature, frictionloss) plus the PD gains the action manager
applies.

Those travel by different routes per backend — solver options live in
three unrelated config objects (mjlab ``MujocoCfg``, Newton
``SolverMuJoCoCfg``, Genesis ``RigidOptions``), and DOF parameters can
be overridden by the actuator config on some paths and inherited from
the MJCF on others — so a preset that *looks* symmetric can hand each
engine different numbers. A ``dof_vel`` / joint-tracking reward that
diverges across sims with everything contact-side equal points here.

Reads every value from the LIVE objects after build (compiled mj_model
for mujoco/newton, solver options + entity getters for genesis, and the
shared action manager), never from the preset — the point is what the
simulator was actually handed.

Values are read once right after build, before any explicit reset, so
startup DR has not fired; a nonzero per-env std in the PD table means it
did and the affected row is reported rather than failed.

Usage:
    python -m jaxrlworld.scripts.diag.parity.dof_solver_param_parity_diag
    python -m jaxrlworld.scripts.diag.parity.dof_solver_param_parity_diag --robots go2
    python -m jaxrlworld.scripts.diag.parity.dof_solver_param_parity_diag --robots module.path:ClassName
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

from jaxrlworld.scripts.diag.parity.contact_pair_parity_diag import (  # noqa: E402
    ROBOTS,
    build,
)

SIMS = ("mujoco", "newton", "genesis")

# mjtIntegrator enum → name (mujoco/newton mj_model.opt.integrator).
_MJ_INTEGRATOR = {0: "euler", 1: "rk4", 2: "implicit", 3: "implicitfast"}
_MJ_CONE = {0: "pyramidal", 1: "elliptic"}

_DOF_FIELDS = ("damping", "armature", "frictionloss")
_PD_FIELDS = ("stiffness", "damping", "effort_limit")

_ATOL = 1e-8


def leaf(name: str) -> str:
    return name.split("/")[-1]


def canonical_joint(raw: str, vocab: list[str]) -> str:
    """Map a backend's joint label to the action-manager joint name.

    Backends decorate the XML joint name differently (mjlab attaches
    under an entity prefix, Newton keeps a body-path label, and either
    may flatten separators), so cross-sim matching goes by suffix
    against the canonical actuated-joint vocabulary instead of by a
    separator convention.
    """
    hits = [n for n in vocab if raw == n or raw.endswith("_" + n) or raw.endswith("/" + n)]
    if len(hits) == 1:
        return hits[0]
    return leaf(raw)


def decode_disableflags(flags: int) -> str:
    import mujoco

    names = [
        n.removeprefix("mjDSBL_").lower()
        for n in dir(mujoco.mjtDisableBit)
        if n.startswith("mjDSBL_") and flags & getattr(mujoco.mjtDisableBit, n).value
    ]
    return "+".join(sorted(names)) if names else "none"


def read_solver(env, sim: str) -> dict[str, object]:
    """Comparable solver/timing facts, plus backend-only extras under ``x_``."""
    out: dict[str, object] = {
        "control_dt": round(env.physics_dt * env.decimation, 9),
        "physics_dt": round(env.physics_dt, 9),
        "decimation": env.decimation,
    }
    if sim in ("mujoco", "newton"):
        m = env.scene_manager.mj_model if sim == "mujoco" else env.scene_manager.solver.mj_model
        out.update(
            iterations=int(m.opt.iterations),
            ls_iterations=int(m.opt.ls_iterations),
            impratio=float(m.opt.impratio),
            cone=_MJ_CONE.get(int(m.opt.cone), str(m.opt.cone)),
            integrator=_MJ_INTEGRATOR.get(int(m.opt.integrator), str(m.opt.integrator)),
            gravity_z=round(float(m.opt.gravity[2]), 6),
            disableflags=decode_disableflags(int(m.opt.disableflags)),
        )
        if sim == "mujoco":
            out["solver_dt"] = round(float(m.opt.timestep), 9)
        else:
            # Newton's SolverMuJoCo writes the per-call dt into the live
            # warp model at every step (mjw_model.opt.timestep.fill_(dt));
            # the CPU-side mj_model keeps a stale build-time value, so
            # the integration dt IS the scene dt the manager passes.
            out["solver_dt"] = out["physics_dt"]
            out["x_mj_model_template_dt"] = round(float(m.opt.timestep), 9)
    else:
        # Per-sim lazy import: genesis only loads in the process that
        # actually builds a genesis env.
        import genesis as gs

        options = env.scene_manager.scene.sim.rigid_solver._options
        out.update(
            solver_dt=round(float(options.dt), 9),
            iterations=int(options.iterations),
            ls_iterations=int(options.ls_iterations),
            impratio=float(options.impratio),
            cone=gs.friction_cone(int(options.friction_cone)).name,
            integrator=gs.integrator(int(options.integrator)).name,
            gravity_z=round(float(env.scene_manager.scene.sim.gravity[2]), 6),
            x_constraint_solver=gs.constraint_solver(int(options.constraint_solver)).name,
            x_tolerance=float(options.tolerance),
        )
    return out


def read_dofs(env, sim: str, vocab: list[str]) -> dict[str, dict[str, float]]:
    """Per-actuated-joint passive DOF parameters, keyed by canonical joint name."""
    out: dict[str, dict[str, float]] = {}
    if sim in ("mujoco", "newton"):
        m = env.scene_manager.mj_model if sim == "mujoco" else env.scene_manager.solver.mj_model
        for j in range(m.njnt):
            if m.jnt_type[j] == 0:  # mjJNT_FREE
                continue
            dof = int(m.jnt_dofadr[j])
            out[canonical_joint(m.joint(j).name, vocab)] = {
                "damping": float(m.dof_damping[dof]),
                "armature": float(m.dof_armature[dof]),
                "frictionloss": float(m.dof_frictionloss[dof]),
            }
    else:
        entity = env.scene_manager["robot"]
        damping = entity.get_dofs_damping()
        armature = entity.get_dofs_armature()
        frictionloss = entity.get_dofs_frictionloss()
        # batch_dofs_info gives (num_envs, n_dofs); otherwise (n_dofs,).
        if damping.dim() == 2:
            damping, armature, frictionloss = damping[0], armature[0], frictionloss[0]
        for joint in entity.joints:
            if joint.n_dofs != 1:  # skip the free joint
                continue
            dof = int(joint.dofs_idx_local[0])
            out[canonical_joint(joint.name, vocab)] = {
                "damping": float(damping[dof]),
                "armature": float(armature[dof]),
                "frictionloss": float(frictionloss[dof]),
            }
    return out


def read_pd(env) -> dict[str, dict[str, tuple[float, float]]]:
    """Per-joint (env0 value, per-env std) for the action-manager PD path.

    Uniform across backends: the explicit PD actuators live in the shared
    action manager. A nonzero std means startup DR fired for that field.
    """
    mgr = env.act_manager
    names = list(mgr._actuated_joint_names)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for act, jidx in mgr._actuators:
        for field in _PD_FIELDS:
            t = getattr(act, field)
            if t.dim() == 2:
                vals, stds = t[0], t.std(dim=0)
            else:
                vals, stds = t, t * 0.0
            for col, j in enumerate(jidx):
                out.setdefault(names[int(j)], {})[field] = (float(vals[col]), float(stds[col]))
    return out


def measure(robot: str, sim: str) -> tuple[dict, dict, dict]:
    env = build(robot, sim)
    vocab = list(env.act_manager._actuated_joint_names)
    solver, dofs, pd = read_solver(env, sim), read_dofs(env, sim, vocab), read_pd(env)
    del env
    return solver, dofs, pd


def run(robot: str, sims: list[str]) -> list[str]:
    print("=" * 100)
    print(f"  {robot.upper()}")
    print("=" * 100)

    solver, dofs, pd = {}, {}, {}
    for sim in sims:
        solver[sim], dofs[sim], pd[sim] = measure(robot, sim)

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(f"{robot}: {label}: {detail}")

    # ── 1. Solver / timing ────────────────────────────────────────────
    print("\n    SOLVER / TIMING — what each engine integrates with")
    keys = sorted({k for s in solver.values() for k in s}, key=lambda k: (k.startswith("x_"), k))
    print(f"      {'field':<22}" + "".join(f"{sim:>18}" for sim in sims))
    for key in keys:
        cells = {sim: solver[sim].get(key) for sim in sims}
        print(f"      {key:<22}" + "".join(f"{str(cells[sim]) if cells[sim] is not None else '—':>18}" for sim in sims))
        speak = [sim for sim in sims if cells[sim] is not None]
        if key.startswith("x_") or len(speak) < 2:
            continue
        vals = {str(cells[sim]) for sim in speak}
        check(f"solver.{key} agrees across {'/'.join(speak)}", len(vals) == 1, f"{cells}" if len(vals) > 1 else "")

    # ── 2. Passive DOF parameters ─────────────────────────────────────
    print("\n    PER-DOF PASSIVE PARAMETERS — damping / armature / frictionloss")
    name_sets = {sim: set(dofs[sim]) for sim in sims}
    all_names = sorted(set.union(*name_sets.values()))
    check(
        "same actuated joint set on every backend",
        all(name_sets[sim] == set(all_names) for sim in sims),
        f"{ {sim: sorted(set(all_names) - name_sets[sim]) for sim in sims} }",
    )
    for field in _DOF_FIELDS:
        print(f"\n      {field}")
        print(f"      {'joint':<28}" + "".join(f"{sim:>16}" for sim in sims))
        for name in all_names:
            row = {sim: dofs[sim].get(name, {}).get(field) for sim in sims}
            print(
                f"      {name:<28}"
                + "".join(f"{row[sim]:>16.6g}" if row[sim] is not None else f"{'—':>16}" for sim in sims)
            )
            present = [row[sim] for sim in sims if row[sim] is not None]
            if len(present) == len(sims) and (max(present) - min(present)) > _ATOL:
                check(f"{name} {field}", False, f"{row}")
    print("    (unlisted rows agree)")

    # ── 3. Action-manager PD path ─────────────────────────────────────
    print("\n    ACTION-MANAGER PD — stiffness / damping / effort_limit (env0, ±std across envs)")
    pd_names = sorted(set.union(*(set(pd[sim]) for sim in sims)))
    for field in _PD_FIELDS:
        print(f"\n      {field}")
        print(f"      {'joint':<28}" + "".join(f"{sim:>22}" for sim in sims))
        for name in pd_names:
            row = {sim: pd[sim].get(name, {}).get(field) for sim in sims}
            cells = []
            for sim in sims:
                if row[sim] is None:
                    cells.append(f"{'—':>22}")
                else:
                    v, s = row[sim]
                    cells.append(f"{v:>14.6g} ±{s:<6.2g}")
            print(f"      {name:<28}" + "".join(cells))
            vals = [row[sim] for sim in sims if row[sim] is not None]
            if len(vals) < len(sims):
                check(f"{name} pd.{field} present on all backends", False, f"{row}")
                continue
            if any(s > _ATOL for _, s in vals):
                # Startup DR landed: per-env values, cross-sim equality of
                # env0 is meaningless. Report, don't fail.
                print(f"        note: per-env spread on {name} {field} (startup DR fired) — parity not judged")
                continue
            if max(v for v, _ in vals) - min(v for v, _ in vals) > _ATOL:
                check(f"{name} pd.{field}", False, f"{ {sim: row[sim][0] for sim in sims} }")
    print("    (unlisted rows agree)")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--robots",
        nargs="+",
        default=["go2"],
        metavar="NAME|module:Class",
        help=f"roster names {sorted(ROBOTS)}, or an explicit module:Class",
    )
    ap.add_argument("--sims", nargs="+", default=list(SIMS), choices=list(SIMS))
    args = ap.parse_args()

    print("=" * 100)
    print("  DO THE THREE BACKENDS RUN THE SAME SOLVER AND THE SAME DOF PARAMETERS")
    print("=" * 100)

    failures: list[str] = []
    for robot in args.robots:
        failures += run(robot, list(args.sims))

    print("\n" + "=" * 100)
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"   - {f}")
        print("=" * 100)
        return 1
    print("  same solver configuration, same DOF parameters, same PD gains")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
