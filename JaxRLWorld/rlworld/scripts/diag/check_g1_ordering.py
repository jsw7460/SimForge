"""Cross-sim ordering verification for g1_29dof.

Builds the g1_29dof preset on each simulator (genesis / newton / mjlab),
resets each env, then dumps every ordering-sensitive piece of state that
could break sim-to-sim transfer:

  * num_actions / num_envs
  * actuated_joint_names (the *policy action order*)
  * joint_pos at reset (per action index)
  * body names + indices (full body list per sim)
  * key body world positions at reset (pelvis, torso, foot frames)
  * resolved PD gains / armature per actuated joint

Writes everything to ``g1_29dof_ordering_check.txt`` in the current
working directory and prints a short summary to stdout (PASS/FAIL per
section).

Use this when sim2sim transfer fails — e.g. newton-mujoco works but
genesis-newton / genesis-mujoco doesn't — to localize the divergence
to action ordering, body presence (the welded foot-frame dummies), or
PD-gain assignment.

Usage:
    python -m rlworld.scripts.diag.check_g1_ordering
    python -m rlworld.scripts.diag.check_g1_ordering --out my_dump.txt
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

# Multi-sim run in one process — bypass the single-backend guard.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np

_PRESETS: dict[str, tuple[str, str]] = {
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
}
_SIMS = ("genesis", "newton", "mujoco")
_KEY_BODIES = (
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_foot_frame",
    "right_foot_frame",
)


def _build_env(preset: str, sim: str, num_envs: int = 1):
    import importlib

    from rlworld.rl.runners import BaseRunner

    mod_path, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _safe(fn, default=None):
    """Run fn(); return its result or stringified error."""
    try:
        return fn()
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"


def _genesis_body_names(env) -> list[str]:
    entity = env.scene_manager["robot"]
    return [link.name for link in entity.links]


def _newton_body_names(env) -> list[str]:
    model = env.scene_manager.solver.model
    labels = list(getattr(model, "body_label", []))
    return [bl.rsplit("/", 1)[-1] if "/" in bl else bl for bl in labels]


def _mujoco_body_names(env) -> list[str]:
    import mujoco

    mj_model = env.scene_manager.mj_model
    out = []
    for i in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        bare = name.rsplit("/", 1)[-1] if "/" in name else name
        out.append(bare)
    return out


def _body_world_pos(env, sim: str, body_name: str):
    """Return (x, y, z) world position of body_name at the current state, or None."""
    try:
        if sim == "genesis":
            entity = env.scene_manager["robot"]
            try:
                link = entity.get_link(body_name)
            except Exception:
                return None
            pos = link.get_pos()
            # Genesis returns either (B, 3) torch tensor or numpy
            arr = pos[0] if hasattr(pos, "shape") and len(pos.shape) == 2 else pos
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            return tuple(float(x) for x in arr)
        elif sim == "newton":
            names = _newton_body_names(env)
            if body_name not in names:
                return None
            idx = names.index(body_name)
            rd = env.get_robot_data()
            pos = rd.body_link_pos_w[0, idx].detach().cpu().numpy()
            return tuple(float(x) for x in pos)
        elif sim == "mujoco":
            names = _mujoco_body_names(env)
            if body_name not in names:
                return None
            idx = names.index(body_name)
            rd = env.get_robot_data()
            pos = rd.body_link_pos_w[0, idx].detach().cpu().numpy()
            return tuple(float(x) for x in pos)
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"
    return None


def _resolved_gains(env, sim: str, joint_names: list[str]) -> dict[str, dict[str, float]]:
    """Resolve per-joint stiffness/damping/armature actually set on the sim model.

    Each sim stores PD gains differently:
      * Genesis: entity.get_dofs_kp / get_dofs_kv / get_dofs_armature
      * Newton: model.joint_target_ke / joint_target_kd / joint_armature
      * mjlab: actuator gainprm / biasprm — best-effort via MjModel
    Best-effort; sims that fail to expose values get an empty dict.
    """
    out: dict[str, dict[str, float]] = {n: {} for n in joint_names}
    try:
        if sim == "genesis":
            entity = env.scene_manager["robot"]
            # Use bare name matching — Genesis joint dofs by name
            from rlworld.rl.envs.managers.genesis import entity_utils  # type: ignore

            for n in joint_names:
                try:
                    dof_ids, _ = entity_utils.find_dofs(entity=entity, name_keys=[n])
                    if not dof_ids:
                        continue
                    kp = (
                        entity.get_dofs_kp(dof_ids).cpu().numpy()
                        if hasattr(entity.get_dofs_kp(dof_ids), "cpu")
                        else np.asarray(entity.get_dofs_kp(dof_ids))
                    )
                    kv = (
                        entity.get_dofs_kv(dof_ids).cpu().numpy()
                        if hasattr(entity.get_dofs_kv(dof_ids), "cpu")
                        else np.asarray(entity.get_dofs_kv(dof_ids))
                    )
                    arm = (
                        entity.get_dofs_armature(dof_ids).cpu().numpy()
                        if hasattr(entity.get_dofs_armature(dof_ids), "cpu")
                        else np.asarray(entity.get_dofs_armature(dof_ids))
                    )
                    out[n] = {"kp": float(kp.flat[0]), "kd": float(kv.flat[0]), "armature": float(arm.flat[0])}
                except Exception:
                    pass
        elif sim == "newton":
            import warp as wp

            model = env.scene_manager.solver.model
            jl = list(getattr(model, "joint_label", getattr(model, "joint_name", [])))
            ke = (
                wp.to_torch(model.joint_target_ke).cpu().numpy()
                if hasattr(model.joint_target_ke, "shape")
                else np.asarray(model.joint_target_ke)
            )
            kd = (
                wp.to_torch(model.joint_target_kd).cpu().numpy()
                if hasattr(model.joint_target_kd, "shape")
                else np.asarray(model.joint_target_kd)
            )
            arm = (
                wp.to_torch(model.joint_armature).cpu().numpy()
                if hasattr(model.joint_armature, "shape")
                else np.asarray(model.joint_armature)
            )
            jl_bare = [n.rsplit("/", 1)[-1] if "/" in n else n for n in jl]
            for n in joint_names:
                if n in jl_bare:
                    i = jl_bare.index(n)
                    out[n] = {
                        "kp": float(ke[i]) if i < len(ke) else float("nan"),
                        "kd": float(kd[i]) if i < len(kd) else float("nan"),
                        "armature": float(arm[i]) if i < len(arm) else float("nan"),
                    }
        elif sim == "mujoco":
            import mujoco

            mj_model = env.scene_manager.mj_model
            # Actuator gainprm[0] = kp for "position" actuators; biasprm[1] = -kp; biasprm[2] = -kv typically.
            # Joint armature is on dof_armature.
            for ai in range(mj_model.nu):
                aname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
                bare = aname.rsplit("/", 1)[-1] if "/" in aname else aname
                if bare in out:
                    kp = float(mj_model.actuator_gainprm[ai, 0])
                    # kv typically in biasprm[2] (negated); fallback to actuator_biasprm
                    bp = mj_model.actuator_biasprm[ai]
                    kv = float(-bp[2]) if bp.shape[0] > 2 else float("nan")
                    out[bare] = {"kp": kp, "kd": kv, "armature": float("nan")}
            # Pull armature from joint_armature by joint name.
            for ji in range(mj_model.njnt):
                jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, ji) or ""
                bare = jname.rsplit("/", 1)[-1] if "/" in jname else jname
                if bare in out and out[bare]:
                    dof_adr = int(mj_model.jnt_dofadr[ji])
                    if dof_adr < len(mj_model.dof_armature):
                        out[bare]["armature"] = float(mj_model.dof_armature[dof_adr])
    except Exception as e:
        out["__error__"] = f"{type(e).__name__}: {e}"  # type: ignore
    return out


def _collect(env, sim: str) -> dict:
    import torch

    torch.manual_seed(0)
    info: dict = {"sim": sim}

    info["num_envs"] = int(env.num_envs)
    info["num_actions"] = int(env.num_actions)

    info["actuated_joint_names"] = list(env.act_manager.actuated_joint_names)

    env.reset()
    rd = env.get_robot_data()

    # joint_pos in canonical (action) order. RobotData.joint_pos uses act-manager indexing.
    try:
        jp = rd.joint_pos[0].detach().cpu().float().numpy()
        info["joint_pos_at_reset"] = jp.tolist()
    except Exception as e:
        info["joint_pos_at_reset_error"] = f"{type(e).__name__}: {e}"

    # Base pose
    info["base_pos_at_reset"] = _safe(lambda: tuple(float(x) for x in rd.body_link_pos_w[0, 0].detach().cpu().numpy()))
    info["base_quat_at_reset"] = _safe(
        lambda: tuple(
            float(x)
            for x in (rd.body_link_quat_w[0, 0] if hasattr(rd, "body_link_quat_w") else rd.base_quat[0])
            .detach()
            .cpu()
            .numpy()
        )
    )

    # Body names enumeration
    body_namer = {"genesis": _genesis_body_names, "newton": _newton_body_names, "mujoco": _mujoco_body_names}[sim]
    try:
        info["body_names"] = body_namer(env)
    except Exception as e:
        info["body_names_error"] = f"{type(e).__name__}: {e}"
        info["body_names"] = []

    # Key body world positions
    info["key_body_pos"] = {bn: _body_world_pos(env, sim, bn) for bn in _KEY_BODIES}

    # Resolved PD gains
    info["resolved_gains"] = _resolved_gains(env, sim, info["actuated_joint_names"])

    return info


def _format_section(info: dict) -> str:
    sim = info.get("sim", "?")
    out: list[str] = []
    sep = "=" * 80
    out.append(sep)
    out.append(f"[{sim}]")
    out.append(sep)
    out.append("")

    if "build_error" in info:
        out.append(f"BUILD ERROR: {info['build_error']}")
        out.append(info.get("tb", ""))
        return "\n".join(out)

    out.append(f"num_envs   : {info.get('num_envs', '?')}")
    out.append(f"num_actions: {info.get('num_actions', '?')}")
    out.append("")

    names = info.get("actuated_joint_names", [])
    jp = info.get("joint_pos_at_reset", [])
    gains = info.get("resolved_gains", {})
    out.append("actuated_joint_names + joint_pos at reset + resolved gains (action order):")
    out.append(f"  {'idx':>3}  {'joint':35s}  {'q@reset':>9}  {'kp':>9}  {'kd':>9}  {'armature':>9}")
    out.append("  " + "-" * 78)
    for i, n in enumerate(names):
        q = jp[i] if i < len(jp) else None
        g = gains.get(n, {})
        kp = g.get("kp", float("nan"))
        kd = g.get("kd", float("nan"))
        arm = g.get("armature", float("nan"))
        q_s = f"{q:+9.4f}" if isinstance(q, int | float) else f"{'--':>9}"
        out.append(f"  {i:3d}  {n:35s}  {q_s}  {kp:9.3f}  {kd:9.3f}  {arm:9.4f}")
    out.append("")

    bp = info.get("base_pos_at_reset")
    bq = info.get("base_quat_at_reset")
    if isinstance(bp, tuple):
        out.append(f"base body world pos at reset : ({bp[0]:+.4f}, {bp[1]:+.4f}, {bp[2]:+.4f})")
    else:
        out.append(f"base body world pos at reset : {bp}")
    if isinstance(bq, tuple):
        out.append(f"base body world quat at reset: ({bq[0]:+.4f}, {bq[1]:+.4f}, {bq[2]:+.4f}, {bq[3]:+.4f})")
    else:
        out.append(f"base body world quat at reset: {bq}")
    out.append("")

    out.append("Key body world positions at reset:")
    for bn in _KEY_BODIES:
        p = info.get("key_body_pos", {}).get(bn)
        if isinstance(p, tuple):
            out.append(f"  {bn:30s}: ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
        else:
            out.append(f"  {bn:30s}: {p}")
    out.append("")

    bns = info.get("body_names", [])
    out.append(f"All body names ({len(bns)}):")
    for i, n in enumerate(bns):
        out.append(f"  {i:3d}: {n}")
    out.append("")

    return "\n".join(out)


def _format_comparison(results: dict[str, dict]) -> str:
    sims = list(results.keys())
    out: list[str] = []
    sep = "=" * 80
    out.append(sep)
    out.append("CROSS-SIM COMPARISON")
    out.append(sep)
    out.append("")

    # num_actions
    nas = {s: results[s].get("num_actions", -1) for s in sims}
    same = len(set(nas.values())) == 1
    out.append(f"num_actions: {nas}  → {'PASS' if same else 'FAIL'}")
    out.append("")

    # Joint name ordering per index
    names_by_sim = {s: results[s].get("actuated_joint_names", []) for s in sims}
    max_n = max((len(v) for v in names_by_sim.values()), default=0)
    out.append("Joint name ordering per action index:")
    hdr = f"  {'idx':>3} | " + " | ".join(f"{s:28s}" for s in sims) + " | match"
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    mismatch_indices = []
    for i in range(max_n):
        cells = []
        vals = []
        for s in sims:
            v = names_by_sim[s][i] if i < len(names_by_sim[s]) else "(missing)"
            cells.append(f"{v:28s}")
            vals.append(v)
        ok = len(set(vals)) == 1
        if not ok:
            mismatch_indices.append(i)
        out.append(f"  {i:3d} | " + " | ".join(cells) + f" | {'✓' if ok else '✗ MISMATCH'}")
    out.append("")
    out.append(f"Joint-order mismatches: {len(mismatch_indices)} / {max_n}")
    if mismatch_indices:
        out.append(f"  → mismatched indices: {mismatch_indices}")
    out.append("")

    # joint_pos cross-sim Δ per action index — uses each sim's own joint at idx i
    # (so if joint orders ALREADY differ, the values won't be comparable; we
    # report anyway with a sim-A name column for context).
    out.append("joint_pos at reset — per action index (sim-A=genesis name for context):")
    hdr = f"  {'idx':>3} | {'joint(genesis)':30s} | " + " | ".join(f"{s:>9s}" for s in sims) + " | maxΔ"
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    jp_by_sim = {s: results[s].get("joint_pos_at_reset", []) for s in sims}
    g_names = names_by_sim.get("genesis", names_by_sim.get(sims[0], []))
    for i in range(max_n):
        cells = []
        vals = []
        for s in sims:
            v = jp_by_sim[s][i] if i < len(jp_by_sim[s]) else None
            if isinstance(v, int | float):
                vals.append(v)
                cells.append(f"{v:+9.4f}")
            else:
                cells.append(f"{'--':>9}")
        d = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        n = g_names[i] if i < len(g_names) else ""
        out.append(f"  {i:3d} | {n:30s} | " + " | ".join(cells) + f" | {d:.4f}")
    out.append("")

    # Resolved gains per joint name — matches by NAME (not index), so even if
    # ordering is permuted across sims this still shows the right comparison.
    all_joint_names = sorted({n for v in names_by_sim.values() for n in v})
    out.append("Resolved PD gains by joint NAME (kp / kd / armature):")
    hdr = f"  {'joint':35s} | " + " | ".join(f"{s:>28s}" for s in sims)
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    for jn in all_joint_names:
        cells = []
        for s in sims:
            g = results[s].get("resolved_gains", {}).get(jn, {})
            if g:
                cells.append(
                    f"{g.get('kp', float('nan')):8.2f}/{g.get('kd', float('nan')):6.2f}/{g.get('armature', float('nan')):6.3f}"
                )
            else:
                cells.append(f"{'--':>28}")
        out.append(f"  {jn:35s} | " + " | ".join(f"{c:>28s}" for c in cells))
    out.append("")

    # Body presence
    bn_by_sim = {s: results[s].get("body_names", []) for s in sims}
    union = sorted(set().union(*[set(v) for v in bn_by_sim.values()]))
    out.append("Body presence across sims (✓ = present, ✗ = missing):")
    hdr = f"  {'body':35s} | " + " | ".join(f"{s:>10s}" for s in sims)
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    for bn in union:
        cells = []
        for s in sims:
            cells.append(" ✓ " if bn in bn_by_sim[s] else " ✗ ")
        out.append(f"  {bn:35s} | " + " | ".join(f"{c:>10s}" for c in cells))
    out.append("")

    # Key body positions cross-sim
    out.append("Key body world positions at reset (cross-sim):")
    hdr = f"  {'body':30s} | " + " | ".join(f"{s:>28s}" for s in sims)
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    for bn in _KEY_BODIES:
        cells = []
        for s in sims:
            p = results[s].get("key_body_pos", {}).get(bn)
            if isinstance(p, tuple):
                cells.append(f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
            else:
                cells.append(str(p) if p is not None else "None")
        out.append(f"  {bn:30s} | " + " | ".join(f"{c:>28s}" for c in cells))
    out.append("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="g1_29dof", choices=list(_PRESETS))
    ap.add_argument("--out", default="g1_29dof_ordering_check.txt")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    print(f"Writing report to: {out_path}")

    results: dict[str, dict] = {}
    for sim in _SIMS:
        print(f"\n{'=' * 60}\nBuilding [{sim}] ...")
        try:
            env = _build_env(args.preset, sim, num_envs=1)
            results[sim] = _collect(env, sim)
            del env
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print(
                f"  collected: {results[sim].get('num_actions', '?')} actions, "
                f"{len(results[sim].get('actuated_joint_names', []))} joints, "
                f"{len(results[sim].get('body_names', []))} bodies"
            )
        except Exception as e:
            print(f"  BUILD ERROR: {type(e).__name__}: {e}")
            results[sim] = {"sim": sim, "build_error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}

    sections: list[str] = []
    for sim in _SIMS:
        sections.append(_format_section(results[sim]))
    valid = {s: r for s, r in results.items() if "build_error" not in r}
    if len(valid) >= 2:
        sections.append(_format_comparison(valid))

    out_path.write_text("\n".join(sections), encoding="utf-8")

    # Short stdout summary.
    print(f"\n{'=' * 60}\nSUMMARY")
    if len(valid) >= 2:
        names_by_sim = {s: valid[s].get("actuated_joint_names", []) for s in valid}
        max_n = max(len(v) for v in names_by_sim.values())
        mismatches = []
        for i in range(max_n):
            vals = {names_by_sim[s][i] for s in valid if i < len(names_by_sim[s])}
            if len(vals) != 1:
                mismatches.append(i)
        nas = {s: valid[s].get("num_actions", -1) for s in valid}
        print(f"  num_actions per sim: {nas}")
        print(f"  joint-order mismatches: {len(mismatches)} / {max_n}")
        bn_union = set().union(*[set(valid[s].get("body_names", [])) for s in valid])
        bn_missing_anywhere = sorted(
            bn for bn in bn_union if not all(bn in valid[s].get("body_names", []) for s in valid)
        )
        print(f"  bodies present in some sims but missing in others: {len(bn_missing_anywhere)}")
        if bn_missing_anywhere:
            print(f"    {bn_missing_anywhere}")
    print(f"\nFull report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
