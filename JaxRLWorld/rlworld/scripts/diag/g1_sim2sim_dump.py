"""Dump every reachable physics parameter + first-N-step eval trace for
G1 sim2sim audit.

Loads the trained checkpoint through ``PolicyEvaluator`` (same code path
``eval_cross_sim.py`` uses), rolls out ``--steps`` policy steps on
``num_envs=1``, and writes a wide plaintext dump:

  Static:  sim meta, canonical joint order, sim_index, action scale /
           offset, joint limits, per-actuator PD gains / armature /
           effort / delay, body names + masses (when sim exposes them),
           gravity.

  Per step: obs vector, policy output, processed action, applied
           torque, joint_pos, joint_vel, root pos/quat/lin_vel/ang_vel
           (world + body frame), projected gravity.

Usage (run for both sims with the SAME checkpoint, then diff):

    jaxpy JaxRLWorld/rlworld/scripts/diag/g1_sim2sim_dump.py \\
        --policy_path ./outputs/models/<run>/checkpoint_latest \\
        --eval_sim newton --out /tmp/g1_newton.txt

    jaxpy JaxRLWorld/rlworld/scripts/diag/g1_sim2sim_dump.py \\
        --policy_path ./outputs/models/<run>/checkpoint_latest \\
        --eval_sim mujoco --out /tmp/g1_mujoco.txt

    diff -u /tmp/g1_newton.txt /tmp/g1_mujoco.txt | head -400

Output is intentionally fixed-precision so line-by-line diff is the
intended comparison method.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from rlworld.rl.evals.evaluator import PolicyEvaluator


# ── formatting helpers ──────────────────────────────────────────────
def _tensor_to_list(t):
    if isinstance(t, torch.Tensor):
        return t.detach().flatten().cpu().tolist()
    return list(t)


def _fmt_row(values, prec: int = 5) -> str:
    out = []
    for v in values:
        try:
            out.append(f"{float(v):+.{prec}f}")
        except Exception:
            out.append(f"{str(v)[:10]:>10}")
    return " ".join(out)


def _fmt_maybe_tensor(v, prec: int = 4) -> str:
    if v is None:
        return "None"
    if torch.is_tensor(v):
        flat = v.detach().flatten().cpu().tolist()
        return "[" + ", ".join(f"{float(x):+.{prec}f}" for x in flat) + "]"
    if hasattr(v, "__len__") and not isinstance(v, str):
        return "[" + ", ".join(f"{float(x):+.{prec}f}" for x in v) + "]"
    return str(v)


def _first_attr(obj, *names):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _g0(x):
    """Index env 0 of a [num_envs, ...] tensor. Dim-safe, list-safe."""
    if isinstance(x, torch.Tensor):
        if x.dim() > 1:
            return x[0].detach().cpu().tolist()
        return x.detach().cpu().tolist()
    return list(x)


def _to_numpy(x):
    """Best-effort convert to a numpy array for reflection-based state dump.

    Returns ``None`` for non-numeric objects (methods, strings, modules,
    builders) so the caller can skip them instead of formatting garbage.
    """
    if x is None:
        return None
    # Skip callables (bound methods, functions) and obvious non-numeric types.
    if callable(x) and not hasattr(x, "__array_interface__") and not hasattr(x, "numpy"):
        return None
    if isinstance(x, (str, bytes, type, dict, set)):
        return None
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        try:
            arr = x.numpy()
            if arr is None or callable(arr):
                return None
            arr = np.asarray(arr)
            if arr.dtype == object:
                return None
            return arr
        except Exception:
            pass
    if hasattr(x, "detach"):
        try:
            arr = x.detach().cpu().numpy()
            if arr.dtype == object:
                return None
            return arr
        except Exception:
            pass
    try:
        arr = np.asarray(x)
        # Reject object arrays (typically from np.asarray(method_or_builder)).
        if arr.dtype == object:
            return None
        # Reject 0-d arrays wrapping non-numeric scalars.
        if arr.ndim == 0:
            try:
                float(arr.item())
            except Exception:
                return None
        return arr
    except Exception:
        return None


def dump_raw_sim_state(env, label: str, max_preview: int = 40) -> None:
    """Reflection-based dump of the sim backend's raw qpos-like arrays.

    We need to see what's actually in ``mjData.qpos`` (MuJoCo) or the
    Newton ``state_0.body_q / joint_q`` tensors at a specific moment
    (reset-immediate vs post-first-step), because the per-env
    ``root_link_pos_w`` reader goes through FK that can mask a raw qpos
    that never got overwritten by ``init_state``.

    Tries a list of candidate attribute paths and dumps whichever exist.
    """
    print(f"\n[RAW SIM STATE — {label}]")
    sm = getattr(env, "scene_manager", None)
    if sm is None:
        print("  (env has no scene_manager)")
        return

    probes = [
        # mjlab / MuJoCo backend candidate paths.
        ("scene_manager._scene.data.qpos", lambda: sm._scene.data.qpos),
        ("scene_manager._scene.data.qvel", lambda: sm._scene.data.qvel),
        ("scene_manager._scene.mj_data.qpos", lambda: sm._scene.mj_data.qpos),
        ("scene_manager.sim.data.qpos", lambda: sm.sim.data.qpos),
        ("scene_manager.sim._data.qpos", lambda: sm.sim._data.qpos),
        ("scene_manager._scene.sim.data.qpos", lambda: sm._scene.sim.data.qpos),
        ("scene_manager.mj_data.qpos", lambda: sm.mj_data.qpos),
        ("scene_manager.model.qpos0", lambda: sm.model.qpos0),
        # Newton backend candidate paths.
        ("scene_manager.state_0.body_q", lambda: sm.state_0.body_q),
        ("scene_manager.state_0.joint_q", lambda: sm.state_0.joint_q),
        ("scene_manager.state_0.joint_qd", lambda: sm.state_0.joint_qd),
        ("scene_manager.state_0.body_qd", lambda: sm.state_0.body_qd),
        ("scene_manager.robot_view.get_root_pose()", lambda: sm.robot_view.get_root_pose()),
        ("scene_manager.robot_view.get_root_velocity()", lambda: sm.robot_view.get_root_velocity()),
    ]

    hit = 0
    for name, getter in probes:
        try:
            val = getter()
        except Exception:
            continue
        arr = _to_numpy(val)
        if arr is None:
            continue
        hit += 1
        flat = np.asarray(arr).flatten()
        shape = getattr(arr, "shape", "?")
        dtype = getattr(arr, "dtype", "?")
        preview_n = min(max_preview, flat.size)
        preview = flat[:preview_n]
        preview_str = _fmt_row(preview) if preview.size > 0 else "(empty)"
        print(f"  {name}: shape={shape} dtype={dtype} size={flat.size}")
        print(f"    first{preview_n:2d} = {preview_str}")

    if hit == 0:
        attrs = [a for a in dir(sm) if not a.startswith("_")][:30]
        print(f"  (no probes matched; top-level scene_manager attrs: {attrs})")


# ── contact + physics full dump ─────────────────────────────────────
def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _fmt_arr(arr, max_n: int = 40, prec: int = 5) -> str:
    try:
        flat = np.asarray(arr).flatten()
    except Exception:
        return f"<unformattable {type(arr).__name__}>"
    if flat.dtype == object:
        return f"<object-dtype size={flat.size}>"
    n = min(max_n, flat.size)
    if n == 0:
        return "(empty)"
    return _fmt_row(flat[:n].tolist(), prec=prec) + (f" [+{flat.size - n} more]" if flat.size > n else "")


def dump_contact_and_physics(env, label: str) -> None:
    """Exhaustive dump: torque routing, contacts, per-geom friction/solref/
    solimp/condim/priority, per-body mass/inertia, solver/integrator options,
    and every plausible Newton ``model`` / ``state`` / ``contacts`` array.

    Intended to be diffed line-by-line between Newton and MuJoCo runs.
    """
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"[CONTACT+PHYSICS FULL — {label}]")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    sm = getattr(env, "scene_manager", None)
    if sm is None:
        print("  (no scene_manager)")
        return
    sim_type = getattr(env, "sim_type", "?")
    print(f"  sim_type = {sim_type}")

    # ==============================================================
    # MuJoCo / mjlab branch
    # ==============================================================
    if sim_type == "mujoco":
        try:
            import mujoco
        except Exception as e:
            print(f"  [cannot import mujoco: {e}]")
            return
        mj = getattr(sm, "mj_model", None) or _try(lambda: sm.sim.mj_model)
        data = _try(lambda: sm.sim.data) or _try(lambda: sm._sim.data)
        wp_data = _try(lambda: sm.sim.wp_data) or _try(lambda: sm._sim.wp_data)
        if mj is None:
            print("  [mj_model not reachable]")
            return

        # Sim options
        print("\n[MJ SIM OPTIONS]")
        print(
            f"  nv={int(mj.nv)} nq={int(mj.nq)} nbody={int(mj.nbody)} ngeom={int(mj.ngeom)} "
            f"npair={int(mj.npair)} njnt={int(mj.njnt)} nu={int(mj.nu)}"
        )
        print(f"  opt.timestep={float(mj.opt.timestep):.6f}")
        print(f"  opt.integrator={int(mj.opt.integrator)} (0=EULER 1=RK4 2=IMPLICIT 3=IMPLICITFAST)")
        print(f"  opt.solver={int(mj.opt.solver)} (0=PGS 1=CG 2=NEWTON)")
        print(f"  opt.cone={int(mj.opt.cone)} (0=PYRAMIDAL 1=ELLIPTIC)")
        print(f"  opt.jacobian={int(mj.opt.jacobian)}")
        print(
            f"  opt.iterations={int(mj.opt.iterations)} ls_iterations={int(mj.opt.ls_iterations)} "
            f"ccd_iterations={int(mj.opt.ccd_iterations)}"
        )
        print(f"  opt.tolerance={float(mj.opt.tolerance):.2e} ls_tolerance={float(mj.opt.ls_tolerance):.2e}")
        print(f"  opt.impratio={float(mj.opt.impratio):.4f}")
        print(f"  opt.gravity={list(mj.opt.gravity)}")
        print(f"  opt.o_friction={list(mj.opt.o_friction)}")
        print(f"  opt.o_solref={list(mj.opt.o_solref)}")
        print(f"  opt.o_solimp={list(mj.opt.o_solimp)}")
        print(f"  opt.o_margin={float(mj.opt.o_margin):.6f}")
        print(f"  opt.noslip_iterations={int(getattr(mj.opt, 'noslip_iterations', -1))}")
        print(f"  opt.disableflags={int(mj.opt.disableflags)} enableflags={int(mj.opt.enableflags)}")

        # Per-DoF forces — prefer wp_data (what mjwarp.step actually reads)
        src, src_name = (wp_data, "wp_data") if wp_data is not None else (data, "data")
        if src is not None:
            print(f"\n[PER-DOF FORCES (from {src_name})]")
            for field in (
                "ctrl",
                "qfrc_applied",
                "qfrc_actuator",
                "qfrc_bias",
                "qfrc_passive",
                "qfrc_smooth",
                "qfrc_constraint",
                "qacc",
                "qvel",
                "qpos",
            ):
                v = getattr(src, field, None)
                arr = _to_numpy(v)
                if arr is None:
                    print(f"  {field}: <missing>")
                    continue
                print(f"  {field}: shape={arr.shape} {_fmt_arr(arr, max_n=40, prec=5)}")

        # Contacts
        print("\n[ACTIVE CONTACTS]")
        try:
            ncon = int(data.ncon) if data is not None else 0
        except Exception:
            ncon = 0
        try:
            if wp_data is not None and ncon == 0:
                ncon_wp = _to_numpy(getattr(wp_data, "ncon", None))
                if ncon_wp is not None:
                    ncon = int(np.asarray(ncon_wp).flatten()[0])
        except Exception:
            pass
        print(f"  ncon={ncon}")
        if ncon > 0 and data is not None:
            for c in range(min(ncon, 30)):
                try:
                    con = data.contact[c]
                    g0 = int(con.geom[0]) if hasattr(con, "geom") else int(con.geom1)
                    g1 = int(con.geom[1]) if hasattr(con, "geom") else int(con.geom2)
                    n0 = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g0) or f"<{g0}>"
                    n1 = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"<{g1}>"
                    pos = con.pos
                    dist = float(con.dist)
                    force = np.zeros(6)
                    try:
                        mujoco.mj_contactForce(mj, data, c, force)
                    except Exception:
                        pass
                    fr = con.friction if hasattr(con, "friction") else None
                    print(
                        f"  c{c}: {n0} <-> {n1} dist={dist:+.6f} "
                        f"pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
                        f"F_n={force[0]:+.3f} F_t1={force[1]:+.3f} F_t2={force[2]:+.3f} "
                        f"T=({force[3]:+.3f},{force[4]:+.3f},{force[5]:+.3f})"
                        + (f" fric={list(fr)}" if fr is not None else "")
                    )
                except Exception as e:
                    print(f"  c{c}: [unreadable: {e}]")

        # Per-geom collision params
        print("\n[PER-GEOM COLLISION PARAMS (contype|conaffinity != 0 only)]")
        hdr = (
            f"  {'gid':>4} {'geom_name':<44} {'body':<28} "
            f"{'ctype':>5} {'cafty':>5} {'condim':>6} {'pri':>3} "
            f"{'size':>24} {'mu':>7} {'tors':>7} {'roll':>7} "
            f"{'solref':>18} {'solimp':>32} {'margin':>9} {'gap':>8}"
        )
        print(hdr)
        for g in range(mj.ngeom):
            ctype = int(mj.geom_contype[g])
            cafty = int(mj.geom_conaffinity[g])
            if ctype == 0 and cafty == 0:
                continue
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g) or "?"
            bid = int(mj.geom_bodyid[g])
            bname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, bid) or "?"
            sz = mj.geom_size[g]
            sz_s = f"[{float(sz[0]):+.3f},{float(sz[1]):+.3f},{float(sz[2]):+.3f}]"
            mu = float(mj.geom_friction[g, 0])
            tors = float(mj.geom_friction[g, 1])
            roll = float(mj.geom_friction[g, 2])
            sr = mj.geom_solref[g]
            si = mj.geom_solimp[g]
            sr_s = "[" + ",".join(f"{float(v):+.4f}" for v in sr) + "]"
            si_s = "[" + ",".join(f"{float(v):+.3f}" for v in si) + "]"
            margin = float(mj.geom_margin[g])
            gap = float(mj.geom_gap[g])
            print(
                f"  {g:>4} {name[:44]:<44} {bname[:28]:<28} "
                f"{ctype:>5} {cafty:>5} {int(mj.geom_condim[g]):>6} {int(mj.geom_priority[g]):>3} "
                f"{sz_s:>24} {mu:>+7.3f} {tors:>+7.3f} {roll:>+7.3f} "
                f"{sr_s:>18} {si_s:>32} {margin:>+9.5f} {gap:>+8.5f}"
            )

        # Per-body mass + inertia
        print("\n[PER-BODY MASS + INERTIA + IPOS]")
        for b in range(mj.nbody):
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b) or "?"
            m = float(mj.body_mass[b])
            i = mj.body_inertia[b]
            ipos = mj.body_ipos[b]
            print(
                f"  b{b:>3} {name[:36]:<36} mass={m:+.6f} "
                f"I=({float(i[0]):+.5f},{float(i[1]):+.5f},{float(i[2]):+.5f}) "
                f"ipos=({float(ipos[0]):+.4f},{float(ipos[1]):+.4f},{float(ipos[2]):+.4f})"
            )

        # Per-DoF params
        print("\n[PER-DOF (armature / damping / frictionloss / invweight)]")
        for d in range(int(mj.nv)):
            a = float(mj.dof_armature[d])
            dmp = float(mj.dof_damping[d])
            fr = float(mj.dof_frictionloss[d])
            iw = float(mj.dof_invweight0[d])
            print(f"  d{d:>3} armature={a:+.6f} damping={dmp:+.5f} frictionloss={fr:+.4f} invweight0={iw:+.6f}")

        # Per-actuator params
        print("\n[PER-ACTUATOR]")
        for a in range(int(mj.nu)):
            aname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "?"
            jid = int(mj.actuator_trnid[a, 0])
            jname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, jid) or "?"
            gp = mj.actuator_gainprm[a]
            bp = mj.actuator_biasprm[a]
            fr = mj.actuator_forcerange[a]
            cr = mj.actuator_ctrlrange[a]
            print(
                f"  a{a:>3} {aname[:30]:<30} joint={jname[:30]:<30} "
                f"gainprm={list(float(x) for x in gp[:3])} biasprm={list(float(x) for x in bp[:3])} "
                f"forcerange=[{float(fr[0]):+.1f},{float(fr[1]):+.1f}] "
                f"ctrlrange=[{float(cr[0]):+.3f},{float(cr[1]):+.3f}]"
            )

        # Explicit geom pairs (if any)
        if int(mj.npair) > 0:
            print(f"\n[EXPLICIT GEOM PAIRS — npair={int(mj.npair)}]")
            for p in range(int(mj.npair)):
                g0, g1 = int(mj.pair_geom1[p]), int(mj.pair_geom2[p])
                n0 = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g0) or f"<{g0}>"
                n1 = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"<{g1}>"
                fr = mj.pair_friction[p]
                sr = mj.pair_solref[p]
                si = mj.pair_solimp[p]
                print(
                    f"  p{p}: {n0} <-> {n1} condim={int(mj.pair_dim[p])} "
                    f"fric={list(float(x) for x in fr)} "
                    f"solref={list(float(x) for x in sr)} "
                    f"solimp={list(float(x) for x in si)}"
                )

    # ==============================================================
    # Newton branch
    # ==============================================================
    elif sim_type == "newton":
        try:
            import warp as wp  # noqa: F401
        except Exception as e:
            print(f"  [cannot import warp: {e}]")
            return
        model = getattr(sm, "model", None)
        control = getattr(sm, "control", None)
        state_0 = getattr(sm, "state_0", None)
        state_1 = getattr(sm, "state_1", None)
        contacts = getattr(sm, "contacts", None)
        solver = getattr(sm, "solver", None)

        # Core scalars
        print("\n[NEWTON SIM OPTIONS]")
        for a in (
            "world_count",
            "joint_count",
            "joint_dof_count",
            "body_count",
            "shape_count",
            "rigid_contact_max",
            "gravity",
            "rigid_contact_margin",
            "rigid_contact_torsional_friction",
            "rigid_contact_rolling_friction",
            "rigid_contact_con_ratio",
            "soft_contact_ke",
            "soft_contact_kd",
            "soft_contact_mu",
            "soft_contact_kf",
        ):
            if model is None:
                break
            v = getattr(model, a, "<missing>")
            arr = _to_numpy(v) if v != "<missing>" else None
            if arr is not None and arr.size <= 6:
                print(f"  {a} = {arr.tolist()}")
            elif arr is not None:
                print(f"  {a} shape={arr.shape}")
            else:
                print(f"  {a} = {v}")

        # Control (applied torque routing)
        if control is not None:
            print("\n[NEWTON CONTROL]")
            for a in dir(control):
                if a.startswith("_"):
                    continue
                try:
                    v = getattr(control, a)
                except Exception:
                    continue
                arr = _to_numpy(v)
                if arr is None:
                    continue
                print(f"  control.{a}: shape={arr.shape} {_fmt_arr(arr, max_n=40, prec=5)}")

        # State arrays (both state_0 and state_1)
        for st_name, st in (("state_0", state_0), ("state_1", state_1)):
            if st is None:
                continue
            print(f"\n[NEWTON {st_name} — top-level wp.array attrs]")
            for a in sorted(dir(st)):
                if a.startswith("_") or a in ("assign", "clear_forces", "requires_grad"):
                    continue
                try:
                    v = getattr(st, a)
                except Exception:
                    continue
                arr = _to_numpy(v)
                if arr is None:
                    continue
                print(f"  {st_name}.{a}: shape={arr.shape} {_fmt_arr(arr, max_n=40, prec=5)}")
            # mujoco namespace
            mjns = getattr(st, "mujoco", None)
            if mjns is not None:
                print(f"\n[NEWTON {st_name}.mujoco namespace — mjwarp Data mirror]")
                for a in sorted(dir(mjns)):
                    if a.startswith("_"):
                        continue
                    try:
                        v = getattr(mjns, a)
                    except Exception:
                        continue
                    arr = _to_numpy(v)
                    if arr is None:
                        continue
                    print(f"  {st_name}.mujoco.{a}: shape={arr.shape} {_fmt_arr(arr, max_n=40, prec=5)}")

        # Contacts
        if contacts is not None:
            print("\n[NEWTON CONTACTS]")
            for a in sorted(dir(contacts)):
                if a.startswith("_"):
                    continue
                try:
                    v = getattr(contacts, a)
                except Exception:
                    continue
                arr = _to_numpy(v)
                if arr is None:
                    if not callable(v) and not isinstance(v, (type, dict)):
                        try:
                            s = repr(v)[:120]
                        except Exception:
                            s = f"<{type(v).__name__}>"
                        print(f"  contacts.{a} = {s}")
                    continue
                print(f"  contacts.{a}: shape={arr.shape} {_fmt_arr(arr, max_n=60, prec=5)}")

        # Model: every "physically interesting" attribute
        if model is not None:
            print("\n[NEWTON MODEL — shape/body/joint/contact params]")
            # Explicit list of attrs we care about, in groups.
            groups = {
                "joint": [
                    "joint_label",
                    "joint_type",
                    "joint_qd_start",
                    "joint_q_start",
                    "joint_dof_count",
                    "joint_target_ke",
                    "joint_target_kd",
                    "joint_armature",
                    "joint_effort_limit",
                    "joint_velocity_limit",
                    "joint_friction",
                    "joint_damping",
                    "joint_limit_lower",
                    "joint_limit_upper",
                    "joint_limit_ke",
                    "joint_limit_kd",
                ],
                "body": ["body_label", "body_mass", "body_inertia", "body_com", "body_inv_mass", "body_inv_inertia"],
                "shape": [
                    "shape_body",
                    "shape_transform",
                    "shape_geo",
                    "shape_count",
                    "shape_materials",
                    "shape_mu",
                    "shape_ke",
                    "shape_kd",
                    "shape_kf",
                    "shape_contact_thickness",
                    "shape_flags",
                    "shape_priority",
                    "shape_filter",
                    "shape_collision_group",
                    "shape_radius",
                    "shape_scale",
                ],
            }
            for gname, attrs in groups.items():
                print(f"  -- group [{gname}] --")
                for a in attrs:
                    v = getattr(model, a, None)
                    if v is None:
                        continue
                    arr = _to_numpy(v)
                    if arr is None:
                        print(f"    model.{a} = <non-array {type(v).__name__}> {v if not callable(v) else ''}")
                        continue
                    print(f"    model.{a}: shape={arr.shape} dtype={arr.dtype} {_fmt_arr(arr, max_n=60, prec=5)}")
            # Shape materials submembers
            sm_attr = getattr(model, "shape_materials", None)
            if sm_attr is not None:
                print("  -- group [shape_materials sub-fields] --")
                for a in sorted(dir(sm_attr)):
                    if a.startswith("_"):
                        continue
                    try:
                        v = getattr(sm_attr, a)
                    except Exception:
                        continue
                    arr = _to_numpy(v)
                    if arr is None:
                        continue
                    print(f"    shape_materials.{a}: shape={arr.shape} {_fmt_arr(arr, max_n=60, prec=5)}")

        # SolverMuJoCo internal mjwarp model / data
        if solver is not None:
            print(f"\n[NEWTON SOLVER — {type(solver).__name__}]")
            print(f"  use_mujoco_cpu = {getattr(solver, 'use_mujoco_cpu', '?')}")
            for inner_name in ("mjw_model", "mjw_data", "mj_model", "mj_data"):
                inner = getattr(solver, inner_name, None)
                if inner is None:
                    continue
                print(f"\n  [solver.{inner_name}] -- interesting attrs")
                interesting_kw = (
                    "friction",
                    "solref",
                    "solimp",
                    "condim",
                    "priority",
                    "margin",
                    "gap",
                    "geom",
                    "dof_",
                    "qfrc_",
                    "ctrl",
                    "pair_",
                    "contact",
                    "body_mass",
                    "body_inertia",
                    "armature",
                    "damping",
                    "frictionloss",
                    "actuator_",
                    "jnt_",
                    "opt_",
                )
                for a in sorted(dir(inner)):
                    if a.startswith("_"):
                        continue
                    if not any(k in a.lower() for k in interesting_kw):
                        continue
                    try:
                        v = getattr(inner, a)
                    except Exception:
                        continue
                    arr = _to_numpy(v)
                    if arr is None:
                        if not callable(v):
                            print(f"    {a} = {v!r}")
                        continue
                    print(f"    {a}: shape={arr.shape} {_fmt_arr(arr, max_n=40, prec=5)}")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


# ── static dump ─────────────────────────────────────────────────────
def print_static(env) -> None:
    am = env.act_manager
    rd = env.robot_data

    print("=" * 100)
    print("[SIM META]")
    print(f"  sim_type              = {getattr(env, 'sim_type', '?')}")
    print(f"  device                = {env.device}")
    print(f"  num_envs              = {env.num_envs}")
    print(f"  num_actions           = {env.num_actions}")
    print(f"  physics_dt            = {getattr(env, 'physics_dt', '?')}")
    print(f"  control_dt            = {getattr(env, 'control_dt', '?')}")
    print(f"  decimation            = {getattr(env, 'decimation', '?')}")
    print(f"  total_action_dim      = {am.total_action_dim}")

    # Canonical joint order — this is the single source of truth used
    # everywhere for obs / action indexing.
    print("\n[CANONICAL JOINT ORDER]")
    print(f"{'idx':<4} {'joint_name':<36} {'sim_idx':<8} {'scale':>12} {'offset':>12}")
    scales = _tensor_to_list(am._scale) if hasattr(am, "_scale") else [None] * am.total_action_dim
    offsets = (
        _tensor_to_list(am._offset[0])
        if hasattr(am, "_offset") and am._offset is not None
        else [None] * am.total_action_dim
    )
    sim_ids = _tensor_to_list(am._indexing.sim_indices) if hasattr(am, "_indexing") else [None] * am.total_action_dim
    for i, name in enumerate(am.actuated_joint_names):
        s = f"{scales[i]:+.6f}" if scales[i] is not None else "?"
        o = f"{offsets[i]:+.6f}" if offsets[i] is not None else "?"
        si = str(int(sim_ids[i])) if sim_ids[i] is not None else "?"
        print(f"{i:<4} {name:<36} {si:<8} {s:>12} {o:>12}")

    # Joint limits in canonical order.
    try:
        lo = _tensor_to_list(am._indexing.joint_limits_lower)
        hi = _tensor_to_list(am._indexing.joint_limits_upper)
        print("\n[JOINT LIMITS (canonical order)]")
        print(f"{'idx':<4} {'joint_name':<36} {'lower':>11} {'upper':>11}")
        for i, name in enumerate(am.actuated_joint_names):
            print(f"{i:<4} {name:<36} {lo[i]:+.5f}   {hi[i]:+.5f}")
    except Exception as e:
        print(f"[joint limits dump skipped: {e}]")

    # Per-actuator PD / armature / effort.
    print("\n[ACTUATOR PD GAINS]")
    if not getattr(am, "_actuators", None):
        print("  (no explicit actuators — Implicit mode, PD is sim-side)")
    else:
        for idx, (inst, joint_ids) in enumerate(am._actuators):
            kp = _first_attr(inst, "stiffness", "_stiffness", "kp", "_kp")
            kd = _first_attr(inst, "damping", "_damping", "kd", "_kd")
            armature = _first_attr(inst, "armature", "_armature")
            effort = _first_attr(inst, "effort_limit", "_effort_limit", "effort")
            delay_min = _first_attr(inst, "min_delay", "_min_delay")
            delay_max = _first_attr(inst, "max_delay", "_max_delay")
            ids = _tensor_to_list(joint_ids) if torch.is_tensor(joint_ids) else list(joint_ids)
            print(f"  actuator[{idx}] type={type(inst).__name__}  joint_ids={ids}")
            print(f"    kp           = {_fmt_maybe_tensor(kp)}")
            print(f"    kd           = {_fmt_maybe_tensor(kd)}")
            print(f"    armature     = {_fmt_maybe_tensor(armature)}")
            print(f"    effort_limit = {_fmt_maybe_tensor(effort)}")
            print(f"    min_delay    = {delay_min}")
            print(f"    max_delay    = {delay_max}")

    # Bodies + masses (sim-dependent accessors).
    print("\n[BODIES]")
    body_names = getattr(rd, "body_names_all", None) or getattr(rd, "body_names", None)
    if body_names is None:
        print("  (body_names unavailable)")
    else:
        print(f"  count = {len(body_names)}")
        masses = None
        for getter in (
            lambda: rd.body_mass_all,
            lambda: rd.body_masses,
            lambda: rd.body_mass,
            lambda: getattr(getattr(env, "scene_manager", None), "model", None).body_mass,
        ):
            try:
                m = getter()
                if m is not None:
                    masses = _tensor_to_list(m) if torch.is_tensor(m) else list(m)
                    break
            except Exception:
                continue
        if masses is not None:
            print(f"  {'idx':<4} {'body_name':<36} {'mass[kg]':>12}")
            for i, nm in enumerate(body_names):
                mm = masses[i] if i < len(masses) else None
                mm_s = f"{float(mm):.6f}" if mm is not None else "?"
                print(f"  {i:<4} {str(nm):<36} {mm_s:>12}")
        else:
            print(f"  body_names: {[str(n) for n in body_names]}")

    # Gravity (sim-dependent).
    try:
        g = env.scene_manager.gravity
        print(f"\n[GRAVITY] {_fmt_maybe_tensor(g)}")
    except Exception:
        pass

    print("=" * 100)


# ── per-step dump ───────────────────────────────────────────────────
def dump_step(env, step_idx: int, obs, raw_action) -> None:
    am = env.act_manager
    rd = env.robot_data

    print(f"\n── step {step_idx} ──")

    # Observation. Split by group if the obs_manager returned a dict.
    if isinstance(obs, dict):
        for gname, gvec in obs.items():
            vec = _g0(gvec)
            print(f"obs[{gname}]      [{len(vec):3d}] = {_fmt_row(vec)}")
    else:
        vec = _g0(obs)
        print(f"obs              [{len(vec):3d}] = {_fmt_row(vec)}")

    # Policy output (after joint_perm → sim-native order).
    vec = _g0(raw_action)
    print(f"policy_action    [{len(vec):2d}] = {_fmt_row(vec)}")

    # Processed action (after scale/offset) if the act_manager exposes it.
    pa = _first_attr(am, "_processed_action", "processed_action")
    if pa is None and getattr(am, "_processed_action_history", None):
        pa = am._processed_action_history[0]
    if pa is not None:
        vec = _g0(pa)
        print(f"processed_action [{len(vec):2d}] = {_fmt_row(vec)}")

    # Applied torque (the PD output that actually goes into the sim).
    tq = _first_attr(am, "applied_torque", "_applied_torque")
    if tq is not None:
        vec = _g0(tq)
        print(f"applied_torque   [{len(vec):2d}] = {_fmt_row(vec)}")

    # Joint state.
    print(f"joint_pos        [{rd.joint_pos.shape[-1]:2d}] = {_fmt_row(_g0(rd.joint_pos))}")
    print(f"joint_vel        [{rd.joint_vel.shape[-1]:2d}] = {_fmt_row(_g0(rd.joint_vel))}")

    # Root state — world and body frame when available.
    print(f"root_pos_w       [ 3] = {_fmt_row(_g0(rd.root_link_pos_w))}")
    print(f"root_quat_w wxyz [ 4] = {_fmt_row(_g0(rd.root_link_quat_w))}")
    print(f"root_lin_vel_w   [ 3] = {_fmt_row(_g0(rd.root_link_lin_vel_w))}")
    print(f"root_ang_vel_w   [ 3] = {_fmt_row(_g0(rd.root_link_ang_vel_w))}")
    if hasattr(rd, "root_link_lin_vel_b"):
        print(f"root_lin_vel_b   [ 3] = {_fmt_row(_g0(rd.root_link_lin_vel_b))}")
    if hasattr(rd, "root_link_ang_vel_b"):
        print(f"root_ang_vel_b   [ 3] = {_fmt_row(_g0(rd.root_link_ang_vel_b))}")
    if hasattr(rd, "projected_gravity_b"):
        print(f"proj_gravity_b   [ 3] = {_fmt_row(_g0(rd.projected_gravity_b))}")


# ── main ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument(
        "--eval_sim",
        required=True,
        choices=("newton", "mujoco", "genesis"),
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file. Defaults to ./g1_dump_<sim>.txt.",
    )
    args = parser.parse_args()

    out_path = args.out or f"g1_dump_{args.eval_sim}.txt"
    log_fh = open(out_path, "w")
    orig_stdout = sys.stdout
    sys.stdout = log_fh

    print(f"# g1_sim2sim_dump sim={args.eval_sim} steps={args.steps} policy_path={args.policy_path}")

    # num_envs=1 → one row per print.
    extra = {"env": {"num_envs": 1}}

    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=args.eval_sim,
        extra_overrides=extra,
        record_video=False,
    )
    env = evaluator.env
    policy = evaluator.policy

    print_static(env)

    # Critical: dump the raw sim-native state AT reset-immediate, before any
    # policy step runs. This is the key moment where the two sims may
    # disagree on how init_state was applied.
    dump_raw_sim_state(env, "post-reset, pre-step-0")
    dump_contact_and_physics(env, "post-reset, pre-step-0")

    # Initial obs — PolicyEvaluator.__init__ already called env.reset().
    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state()

    action = None
    with torch.no_grad():
        for step_idx in range(args.steps):
            action = policy.get_action(obs, robot_states)
            # Dump pre-step — the state that produced this action.
            dump_step(env, step_idx, obs, action)
            # Raw sim state at the same moment — lets us see what mjData /
            # Newton state_0 hold vs what the high-level readers report.
            dump_raw_sim_state(env, f"pre-step-{step_idx}")
            dump_contact_and_physics(env, f"pre-step-{step_idx}")

            obs, _rewards, _terminated, _truncated, _infos = env.step(action)
            robot_states = env.get_robot_state()

    # Final post-step dump so the last action's effect is visible.
    if action is not None:
        print("\n── post-final-step state ──")
        dump_step(env, args.steps, obs, action)
        dump_raw_sim_state(env, "post-final-step")
        dump_contact_and_physics(env, "post-final-step")

    sys.stdout = orig_stdout
    log_fh.close()
    print(f"[g1_sim2sim_dump] wrote {out_path}")


if __name__ == "__main__":
    main()
