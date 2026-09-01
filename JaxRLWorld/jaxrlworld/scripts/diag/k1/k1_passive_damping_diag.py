"""K1 passive joint damping: is the XML ``dof_damping`` (leg 3 / arm 2) ACTUALLY
present in the running simulator's model, and is it DISTINCT from the external
PD kd?

Background. The training MJCF (``k1_mjx_feetonly.xml``) sets, in its default
class, ``damping=3`` on the legs (Hip/Knee/Ankle) and ``damping=2`` on the
arms/head (Shoulder/Elbow/Head). Booster's deploy model (``K1_22dof.xml``) has
``damping=0``. The RL policy's PD kd is applied EXTERNALLY (``DelayedPD``, motor
mode; ``K1SpecFn`` strips the XML actuators) — so two independent
velocity-opposing layers exist:

  * the model's PASSIVE ``dof_damping`` (physical joint viscous damping)   ← XML
  * the controller's PD kd (``kp*err - kd*vel``, action pipeline)          ← config

This diag reads the LIVE, compiled model that the solver actually steps and
answers, per sim, after build + reset:

  1. model ``dof_damping`` per joint (legs vs arms) vs XML-expected (3 / 2)
  2. model ``dof_frictionloss`` per joint vs XML-expected (0.1)
  3. model ``dof_armature`` per joint vs the Booster reference (per-joint,
     NOT the XML's flat 0.005 — a value near 0.005 means the override did NOT
     land)
  4. external PD kp/kd from the action manager (the active profile gains),
     printed to CONTRAST with (1): ``dof_damping`` and PD kd are different
     quantities living in different layers
  5. host-model actuator count / biastype — confirm the PD is NOT baked into
     the model as a position servo (XML actuators stripped ⇒ motor / external)

Values come from the BATCHED model the solver reads (env 0), not just the host
compile; the host model is used only for the joint name -> dof-address map.
Per-sim access mirrors ``k1_friction_dr_diag`` / ``k1_dr_landing_diag`` (sim
backends imported lazily on purpose — importing genesis/warp has device side
effects).

Run::

    jaxpy -m jaxrlworld.scripts.diag.k1.k1_passive_damping_diag --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.k1.k1_passive_damping_diag              # all three
"""

from __future__ import annotations

import argparse

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}

# XML default-class passive joint damping (k1_mjx_feetonly.xml).
_EXP_DAMP_LEG = 3.0
_EXP_DAMP_ARM = 2.0
_EXP_FRICTIONLOSS = 0.1
_LEG_KEYS = ("Hip", "Knee", "Ankle")
_ARM_KEYS = ("Shoulder", "Elbow", "Head")

# Reference armature (Booster K1_22dof.xml). Training overrides the XML's flat
# 0.005 with these per-joint values; anything near 0.005 means it did NOT land.
_EXP_ARM = {
    "Hip_Pitch": 0.0478125,
    "Hip_Roll": 0.0339552,
    "Hip_Yaw": 0.0282528,
    "Knee": 0.095625,
    "Ankle": 0.0565,
    "Head": 0.002,
    "Shoulder": 0.001,
    "Elbow": 0.001,
}
_XML_FLAT_ARM = 0.005
_TOL = 1e-4
# frictionloss (0.1) and armature are DR-randomized per env, so env-0 lands a
# draw, not the base. Verdict uses DR-tolerant bands; damping is NOT randomized
# so it stays exact.
_FRIC_DR_TOL = 0.05  # |v - 0.1| within this ⇒ base 0.1 (DR jitter)
_ARM_DR_FRAC = 0.25  # within 25% of the Booster ref ⇒ override landed


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _group(name: str) -> str:
    if any(k in name for k in _LEG_KEYS):
        return "leg"
    if any(k in name for k in _ARM_KEYS):
        return "arm"
    return "other"


def _exp_damp(name: str) -> float:
    g = _group(name)
    return _EXP_DAMP_LEG if g == "leg" else (_EXP_DAMP_ARM if g == "arm" else 0.0)


def _exp_arm(name: str) -> float | None:
    """Reference armature for a joint. Matches by the LAST-occurring key so the
    full kinematic-chain names some backends use (e.g. newton's
    ``..._Hip_Pitch_..._Hip_Roll``) resolve to the LEAF joint, not a parent
    whose name is a substring."""
    best, best_pos = None, -1
    for key, val in _EXP_ARM.items():
        pos = name.rfind(key)
        if pos > best_pos:
            best_pos, best = pos, val
    return best


def _env0(x, nv: int):
    """Batched dof array -> env-0 1-D tensor of length nv. Crashes on mismatch."""
    import torch

    t = torch.as_tensor(x).detach().float().cpu().reshape(-1)
    if t.numel() == nv:
        return t
    if t.numel() % nv == 0:
        return t.reshape(-1, nv)[0]
    raise RuntimeError(f"dof array size {t.numel()} not a multiple of nv={nv}")


def _pd_gains(env):
    """External PD kp/kd per actuated joint (env 0) from the action manager."""
    import torch

    am = env.act_manager
    names = list(am.actuated_joint_names)
    dev = env.device
    kp = torch.zeros(len(names), device=dev)
    kd = torch.zeros(len(names), device=dev)
    act_type = None
    for actuator, joint_idx in am.actuators:
        s = torch.as_tensor(actuator.stiffness, device=dev).float()
        d = torch.as_tensor(actuator.damping, device=dev).float()
        if s.dim() == 2:
            s = s[0]
        if d.dim() == 2:
            d = d[0]
        kp[joint_idx] = s.reshape(-1)
        kd[joint_idx] = d.reshape(-1)
        act_type = type(actuator).__name__
    return names, kp.detach().cpu(), kd.detach().cpu(), act_type


def _read_host(sim: str, env):
    """(host mj_model, batched dof_damping/frictionloss/armature env-0) for the
    mjwarp-backed sims. Returns None for genesis (no host MjModel)."""
    sm = env.scene_manager
    if sim == "mujoco":
        mj = sm.mj_model
        nv = int(mj.nv)
        damp = _env0(sm.model.dof_damping, nv)
        fric = _env0(sm.model.dof_frictionloss, nv)
        arm = _env0(sm.model.dof_armature, nv)
        return mj, damp, fric, arm
    if sim == "newton":
        import warp as wp

        mj = sm.solver.mj_model
        nv = int(mj.nv)
        mjw = sm.solver.mjw_model
        damp = _env0(wp.to_torch(mjw.dof_damping), nv)
        fric = _env0(wp.to_torch(mjw.dof_frictionloss), nv)
        arm = _env0(wp.to_torch(mjw.dof_armature), nv)
        return mj, damp, fric, arm
    return None


def _per_joint_rows(mj, damp, fric, arm):
    """Per-hinge-joint rows via host mj_model name<->dofadr map."""
    import mujoco

    rows = []
    for j in range(mj.njnt):
        if int(mj.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
        d = int(mj.jnt_dofadr[j])
        rows.append(
            {
                "name": name,
                "damping": float(damp[d]),
                "frictionloss": float(fric[d]),
                "armature": float(arm[d]),
            }
        )
    return rows


def _actuator_mode(mj):
    """Host-model actuator summary. Distinguishes an INERT position servo
    (kp=0 leftover from the XML <actuator> block, zeroed for explicit
    actuators) from an ACTIVE one (kp>0 ⇒ the model PD runs on TOP of the
    external DelayedPD torque = double control)."""
    import mujoco

    nu = int(mj.nu)
    if nu == 0:
        return {
            "nu": 0,
            "n_position_servo": 0,
            "servo_kp_max": 0.0,
            "servo_kv_max": 0.0,
            "note": "no model actuators (control fully external)",
        }
    affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    bias = [int(mj.actuator_biastype[a]) for a in range(nu)]
    gp = mj.actuator_gainprm.reshape(nu, -1)  # position servo: gainprm[0] = kp
    bp = mj.actuator_biasprm.reshape(nu, -1)  # position servo: biasprm[2] = -kv
    servo_kp = [float(gp[a, 0]) for a in range(nu) if bias[a] == affine]
    servo_kv = [float(-bp[a, 2]) for a in range(nu) if bias[a] == affine and bp.shape[1] > 2]
    n_servo = len(servo_kp)
    kp_max = max(servo_kp) if servo_kp else 0.0
    kv_max = max(servo_kv) if servo_kv else 0.0
    if n_servo == 0:
        note = "all motor (no PD in model)"
    elif kp_max < 1e-6 and kv_max < 1e-6:
        note = f"{n_servo} position servos INERT (kp=kv=0) — external torque drives, no double control"
    else:
        note = (
            f"{n_servo} position servos ACTIVE (kp<= {kp_max:.3g}, kv<= {kv_max:.3g}) — DOUBLE control with DelayedPD"
        )
    return {"nu": nu, "n_position_servo": n_servo, "servo_kp_max": kp_max, "servo_kv_max": kv_max, "note": note}


def _genesis_dist(env):
    """Genesis has no host MjModel; report the dof_damping/armature value
    distribution (how many dofs ~3 / ~2 / ~0 / other) — enough to confirm the
    passive damping is present without a name<->dof map."""
    import torch

    robot = env.scene_manager["robot"]
    damp = torch.as_tensor(robot.get_dofs_damping()).detach().float().cpu().reshape(-1)
    arm = torch.as_tensor(robot.get_dofs_armature()).detach().float().cpu().reshape(-1)

    def _count(t, val, tol=1e-3):
        return int(((t - val).abs() <= tol).sum())

    n = damp.numel()
    return {
        "n_dofs": n,
        "damp_leg3": _count(damp, _EXP_DAMP_LEG),
        "damp_arm2": _count(damp, _EXP_DAMP_ARM),
        "damp_zero": _count(damp, 0.0),
        "damp_min": float(damp.min()),
        "damp_max": float(damp.max()),
        "arm_flat005": _count(arm, _XML_FLAT_ARM),
        "arm_min": float(arm.min()),
        "arm_max": float(arm.max()),
    }


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs} seed={seed}")

    from jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    preset = K1G1RecipeConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    cfgs = preset.build()
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()
    _stage("env built + first reset")

    out: dict = {"sim": sim, "num_envs": num_envs}

    pd_names, kp, kd, act_type = _pd_gains(env)
    out["pd_act_type"] = act_type
    out["pd_kp"] = {
        "leg": [float(kp[i]) for i, n in enumerate(pd_names) if _group(n) == "leg"],
        "arm": [float(kp[i]) for i, n in enumerate(pd_names) if _group(n) == "arm"],
    }
    out["pd_kd"] = {
        "leg": [float(kd[i]) for i, n in enumerate(pd_names) if _group(n) == "leg"],
        "arm": [float(kd[i]) for i, n in enumerate(pd_names) if _group(n) == "arm"],
    }

    host = _read_host(sim, env)
    if host is not None:
        mj, damp, fric, arm = host
        out["rows"] = _per_joint_rows(mj, damp, fric, arm)
        out["act_mode"] = _actuator_mode(mj)
    else:
        out["genesis_dist"] = _genesis_dist(env)

    _stage(f"cell done: {sim}")
    return out


def _uniq(vals: list[float]) -> str:
    seen = sorted({round(v, 4) for v in vals})
    return ", ".join(f"{v:g}" for v in seen) if seen else "-"


def _print_cell(r: dict) -> None:
    sim = r["sim"]
    print(f"\n===== {sim.upper()} (num_envs={r['num_envs']}) =====")

    # External PD (contrast layer).
    print(f"  external PD actuator: {r['pd_act_type']}")
    print(f"    PD kp   leg=[{_uniq(r['pd_kp']['leg'])}]  arm=[{_uniq(r['pd_kp']['arm'])}]")
    print(f"    PD kd   leg=[{_uniq(r['pd_kd']['leg'])}]  arm=[{_uniq(r['pd_kd']['arm'])}]")
    print("    (^ controller layer — SEPARATE from the model dof_damping below)")

    if "genesis_dist" in r:
        d = r["genesis_dist"]
        print(f"  model dof_damping distribution ({d['n_dofs']} dofs, no host name map):")
        print(
            f"    ~{_EXP_DAMP_LEG:g} (leg): {d['damp_leg3']}   ~{_EXP_DAMP_ARM:g} (arm): {d['damp_arm2']}"
            f"   ~0: {d['damp_zero']}   range=[{d['damp_min']:.4f}, {d['damp_max']:.4f}]"
        )
        print(
            f"  model dof_armature: min={d['arm_min']:.5f} max={d['arm_max']:.5f}"
            f"   (dofs ~0.005 XML-flat: {d['arm_flat005']})"
        )
        damp_ok = d["damp_leg3"] > 0 and d["damp_arm2"] > 0
        arm_ok = d["arm_flat005"] == 0 and d["arm_max"] > 0.02
        print(f"  → passive damping present: {damp_ok} | armature override landed: {arm_ok}")
        print(f"  VERDICT: {'PASS' if (damp_ok and arm_ok) else 'CHECK'}")
        return

    # Per-joint table (mujoco / newton).
    rows = r["rows"]
    am = r["act_mode"]
    print(f"  host actuators: nu={am['nu']} position-servo={am['n_position_servo']} ({am['note']})")
    print("  model dof (env 0, what the solver steps):")
    print(f"    {'joint':<22}{'damping':>9}{'exp':>6} {'':<2}{'frictln':>9}{'armature':>11}{'exp_arm':>10} {''}")
    all_damp_ok = all_fric_ok = all_arm_ok = True
    for row in rows:
        n = row["name"]
        ed = _exp_damp(n)
        ea = _exp_arm(n)
        dok = abs(row["damping"] - ed) <= _TOL  # damping is NOT DR'd ⇒ exact
        fok = abs(row["frictionloss"] - _EXP_FRICTIONLOSS) <= _FRIC_DR_TOL
        flat = 0.004 <= row["armature"] <= 0.006  # stuck at the XML flat 0.005?
        aok = ea is not None and not flat and abs(row["armature"] - ea) <= _ARM_DR_FRAC * ea
        all_damp_ok &= dok
        all_fric_ok &= fok
        all_arm_ok &= aok
        print(
            f"    {n:<22}{row['damping']:>9.3f}{ed:>6.1f} {'OK' if dok else '!!':<2}"
            f"{row['frictionloss']:>9.3f}{row['armature']:>11.5f}{(ea if ea is not None else 0):>10.5f} "
            f"{'OK' if aok else '!!'}"
        )
    print(
        f"  → dof_damping == XML(leg {_EXP_DAMP_LEG:g}/arm {_EXP_DAMP_ARM:g}): {all_damp_ok}"
        f" | frictionloss≈0.1 (DR): {all_fric_ok} | armature override landed (DR): {all_arm_ok}"
    )
    print(
        f"    [info] model actuators: nu={am['nu']} position-servo={am['n_position_servo']}"
        f" servo_kp_max={am['servo_kp_max']:.4g} servo_kv_max={am['servo_kv_max']:.4g}"
    )
    print(f"           {am['note']}")
    print(f"  VERDICT: {'PASS' if (all_damp_ok and all_fric_ok and all_arm_ok) else 'CHECK'}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="K1 passive joint damping (dof_damping) landing / separation-from-PD diag."
    )
    ap.add_argument("--sim", choices=_SIMS, help="Single backend (default: all three).")
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sims = [args.sim] if args.sim else list(_SIMS)
    results = []
    for sim in sims:
        try:
            results.append(run_cell(sim, args.num_envs, args.seed))
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"\n[{sim}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    for r in results:
        _print_cell(r)
    print()
    return 0 if len(results) == len(sims) else 1


if __name__ == "__main__":
    raise SystemExit(main())
