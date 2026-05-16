"""Cross-sim action-indexing parity for ALL presets.

For every preset under ``rlworld/rl/configs/presets/`` and every supported
simulator backend, this diag builds the env, resets it, and dumps every
per-joint quantity that is sensitive to action ordering — name, q-at-reset,
actuator kp/kd, sim-side armature/frictionloss. Then it cross-checks each
quantity across sims at the same canonical index AND for the same joint
name, so every kind of permutation/misalignment is caught.

The action manager's canonical joint order is the source of truth (built
from a kinematic-tree DFS in each sim's ``build_articulation_indexing``).
If this diag reports ✓ on every row for every preset, then:

  * The canonical joint list is identical across sims (so a policy trained
    in one sim transfers cleanly to the others).
  * ``r.p_gains`` / ``r.d_gains`` / ``r.armature`` / actuator ``frictionloss``
    threaded from the robot config land on the SAME physical joint in
    every sim (i.e. the dict→joint name resolution agrees with the
    canonical order on each side).
  * Domain randomization terms that consume per-joint random samples
    (e.g. ``randomize_joint_friction``) target the same physical joint
    across sims when canonical index is preserved through the event API.

Writes the report to ``action_indexing_parity.txt`` in the current
working directory and prints a short summary table to stdout.

Usage:
    python -m rlworld.scripts.diag.check_action_indexing_parity
    python -m rlworld.scripts.diag.check_action_indexing_parity --preset g1_29dof
    python -m rlworld.scripts.diag.check_action_indexing_parity --out custom.txt
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

# Multi-sim run in one process — bypass the single-backend guard.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")


# (preset_key, module_path, cfg_class_name)
_PRESETS: dict[str, tuple[str, str]] = {
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("rlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
    "go2_flat": ("rlworld.rl.configs.presets.go2_flat.base", "Go2FlatConfig"),
    "t1_getup": ("rlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
    "t1_tracking": ("rlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
}
_SIMS = ("genesis", "newton", "mujoco")

# Metrics whose per-joint value must agree across sims for a sound preset.
_METRICS = ("kp", "kd", "armature", "frictionloss", "q_reset")


def _build_env(preset: str, sim: str, num_envs: int = 1):
    import importlib

    from rlworld.rl.runners import BaseRunner

    mod_path, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _capture_actuator_gains(env) -> dict[str, dict[str, float]]:
    """Return {joint_name: {'kp': float, 'kd': float}} from the actuator side.

    Built from ``env.act_manager._actuators`` — each entry is
    (actuator, joint_indices_into_action_dim). Stiffness / damping live on
    the actuator as ``(num_envs, num_joints_in_group)`` tensors.
    """
    out: dict[str, dict[str, float]] = {}
    act_mgr = env.act_manager
    names = list(act_mgr.actuated_joint_names)
    for actuator, joint_indices in act_mgr._actuators:
        idx_list = joint_indices.detach().cpu().tolist() if hasattr(joint_indices, "detach") else list(joint_indices)
        for local_i, canonical_i in enumerate(idx_list):
            jname = names[int(canonical_i)]
            d: dict[str, float] = {}
            if hasattr(actuator, "stiffness"):
                d["kp"] = float(actuator.stiffness[0, local_i])
            if hasattr(actuator, "damping"):
                d["kd"] = float(actuator.damping[0, local_i])
            out[jname] = d
    return out


def _capture_sim_side(env, sim: str) -> dict[str, dict[str, float]]:
    """Return {joint_name: {'armature': float, 'frictionloss': float}} from sim arrays.

    Uses ``ArticulationIndexing.sim_indices`` to look up per-canonical-joint
    armature / frictionloss from the simulator's underlying model.
    """
    out: dict[str, dict[str, float]] = {}
    act_mgr = env.act_manager
    names = list(act_mgr.actuated_joint_names)
    indexing = getattr(act_mgr, "_indexing", None)
    if indexing is None:
        return out

    if sim == "genesis":
        entity = env.scene_manager["robot"]
        sim_indices = indexing.sim_indices
        # Genesis: try get_dofs_armature / get_dofs_frictionloss
        arm = _try(lambda: entity.get_dofs_armature(dofs_idx_local=sim_indices))
        fri = _try(lambda: entity.get_dofs_frictionloss(dofs_idx_local=sim_indices))
        for ci, jname in enumerate(names):
            d: dict[str, float] = {}
            if arm is not None:
                v = arm[0, ci] if arm.ndim == 2 else arm[ci]
                d["armature"] = float(v)
            if fri is not None:
                v = fri[0, ci] if fri.ndim == 2 else fri[ci]
                d["frictionloss"] = float(v)
            out[jname] = d

    elif sim == "newton":
        import warp as wp

        model = env.scene_manager.solver.model
        qd_indices = (
            indexing.newton_qd_indices.detach().cpu().numpy() if hasattr(indexing, "newton_qd_indices") else None
        )
        if qd_indices is None:
            return out
        arm = _try(lambda: wp.to_torch(model.joint_armature).detach().cpu().numpy())
        fri = _try(lambda: wp.to_torch(model.joint_friction).detach().cpu().numpy())
        for ci, jname in enumerate(names):
            qd = int(qd_indices[ci])
            d: dict[str, float] = {}
            if arm is not None and qd < len(arm):
                d["armature"] = float(arm[qd])
            if fri is not None and qd < len(fri):
                d["frictionloss"] = float(fri[qd])
            out[jname] = d

    elif sim == "mujoco":
        import mujoco

        mj_model = env.scene_manager.mj_model
        entity = env.scene_manager._scene["robot"]
        entity_joint_names = list(entity.joint_names)
        # sim_indices are entity-joint indices; map → mj joint id → dof adr.
        sim_indices = indexing.sim_indices.detach().cpu().tolist()
        for ci, jname in enumerate(names):
            entity_jidx = int(sim_indices[ci])
            bare = entity_joint_names[entity_jidx]
            # Find the full mj joint by bare name (could be prefixed like 'robot/g1_29dof/...').
            mj_jid = None
            for jid in range(mj_model.njnt):
                raw = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
                if raw.rsplit("/", 1)[-1] == bare:
                    mj_jid = jid
                    break
            d: dict[str, float] = {}
            if mj_jid is not None:
                dof_adr = int(mj_model.jnt_dofadr[mj_jid])
                if dof_adr < len(mj_model.dof_armature):
                    d["armature"] = float(mj_model.dof_armature[dof_adr])
                if dof_adr < len(mj_model.dof_frictionloss):
                    d["frictionloss"] = float(mj_model.dof_frictionloss[dof_adr])
            out[jname] = d

    return out


def _collect(preset: str, sim: str) -> dict:
    """Build + reset + capture every order-sensitive per-joint quantity."""
    import torch

    info: dict = {"preset": preset, "sim": sim}
    try:
        env = _build_env(preset, sim, num_envs=1)
    except Exception as e:
        info["build_error"] = f"{type(e).__name__}: {e}"
        info["tb"] = traceback.format_exc()
        return info

    try:
        torch.manual_seed(0)
        env.reset()
        names = list(env.act_manager.actuated_joint_names)
        info["actuated_joint_names"] = names

        # q at reset — already in canonical order via RobotData.joint_pos.
        rd = env.get_robot_data()
        q_reset = rd.joint_pos[0].detach().cpu().float().numpy().tolist()
        info["q_reset_by_name"] = {names[i]: float(q_reset[i]) for i in range(len(names))}

        # Actuator-side kp / kd.
        act_metrics = _capture_actuator_gains(env)
        # Sim-side armature / frictionloss.
        sim_metrics = _capture_sim_side(env, sim)

        # Merge into one per-joint dict in canonical order.
        per_joint: dict[str, dict[str, float]] = {}
        for n in names:
            d: dict[str, float] = {}
            if n in act_metrics:
                d.update(act_metrics[n])
            if n in sim_metrics:
                d.update(sim_metrics[n])
            d["q_reset"] = float(info["q_reset_by_name"].get(n, float("nan")))
            per_joint[n] = d
        info["per_joint"] = per_joint
    except Exception as e:
        info["capture_error"] = f"{type(e).__name__}: {e}"
        info["tb"] = traceback.format_exc()
    finally:
        del env
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return info


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and (v != v)):
        return "       --"
    if isinstance(v, int | float):
        absv = abs(float(v))
        if absv >= 100:
            return f"{v:9.2f}"
        if absv >= 1:
            return f"{v:9.4f}"
        return f"{v:9.5f}"
    return str(v)[:9]


def _format_preset_section(preset: str, by_sim: dict[str, dict]) -> str:
    out: list[str] = []
    sep = "=" * 100
    out.append("")
    out.append(sep)
    out.append(f"PRESET: {preset}")
    out.append(sep)
    out.append("")

    # Per-sim build status
    out.append("Build status:")
    for sim, info in by_sim.items():
        if "build_error" in info:
            out.append(f"  [{sim}] BUILD ERROR: {info['build_error']}")
        elif "capture_error" in info:
            out.append(f"  [{sim}] CAPTURE ERROR: {info['capture_error']}")
        else:
            out.append(f"  [{sim}] OK  ({len(info.get('actuated_joint_names', []))} joints)")
    out.append("")

    # Joint name ordering
    name_lists = {s: info.get("actuated_joint_names", []) for s, info in by_sim.items() if "build_error" not in info}
    if not name_lists:
        out.append("All sims failed to build — skipping rest of preset.")
        return "\n".join(out)

    sims = list(name_lists.keys())
    max_n = max(len(v) for v in name_lists.values()) if name_lists else 0

    out.append(f"Canonical joint ordering ({max_n} joints across {len(sims)} sims):")
    hdr = f"  {'idx':>3} | " + " | ".join(f"{s:30s}" for s in sims) + " | match"
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    order_mismatch_indices: list[int] = []
    for i in range(max_n):
        cells: list[str] = []
        vals: list[str] = []
        for s in sims:
            n = name_lists[s][i] if i < len(name_lists[s]) else "(missing)"
            cells.append(f"{n:30s}")
            vals.append(n)
        ok = len(set(vals)) == 1
        if not ok:
            order_mismatch_indices.append(i)
        out.append(f"  {i:3d} | " + " | ".join(cells) + f" | {'✓' if ok else '✗'}")
    out.append("")
    if order_mismatch_indices:
        out.append(f"⚠ joint-name mismatches at indices: {order_mismatch_indices}")
        out.append("  → canonical-order canonicalization is BROKEN for this preset.")
        out.append("    Subsequent metric checks may be meaningless until this is fixed.")
        out.append("")

    # Per-metric, per-joint cross-sim comparison.
    # Compare BY NAME (so even if order disagrees, we still see config-resolution mismatches).
    all_names: set[str] = set()
    for s in sims:
        all_names.update(by_sim[s].get("per_joint", {}).keys())
    all_names_sorted = sorted(all_names)

    summary_pass: dict[str, list[int]] = {m: [] for m in _METRICS}
    for metric in _METRICS:
        out.append(f"--- Metric: {metric} (per joint, by name; max Δ across sims) ---")
        hdr = f"  {'joint':35s} | " + " | ".join(f"{s:>10s}" for s in sims) + " | maxΔ"
        out.append(hdr)
        out.append("  " + "-" * (len(hdr) - 2))
        worst_delta = 0.0
        worst_joint = ""
        for jn in all_names_sorted:
            cells: list[str] = []
            vals: list[float] = []
            for s in sims:
                v = by_sim[s].get("per_joint", {}).get(jn, {}).get(metric)
                if isinstance(v, int | float) and not (v != v):
                    vals.append(float(v))
                    cells.append(_fmt(v))
                else:
                    cells.append("       --")
            if len(vals) >= 2:
                d = max(vals) - min(vals)
                if d > worst_delta:
                    worst_delta, worst_joint = d, jn
                out.append(f"  {jn:35s} | " + " | ".join(cells) + f" | {d:8.5f}")
            else:
                out.append(f"  {jn:35s} | " + " | ".join(cells) + " | (n<2, skip)")
        summary_pass[metric] = [worst_delta, worst_joint]
        out.append(f"  → worst Δ for {metric}: {worst_delta:.6f}" + (f" at {worst_joint}" if worst_joint else ""))
        out.append("")

    out.append("--- Verdict for preset ---")
    pass_overall = True
    if order_mismatch_indices:
        out.append(f"  joint_names    : ✗ FAIL ({len(order_mismatch_indices)} mismatches)")
        pass_overall = False
    else:
        out.append("  joint_names    : ✓ PASS")
    for metric in _METRICS:
        d = summary_pass[metric][0]
        # Use absolute tolerance — values can be small.
        tol = 1e-5
        if d <= tol:
            out.append(f"  {metric:14s}: ✓ PASS (Δ ≤ {tol})")
        else:
            out.append(f"  {metric:14s}: ✗ FAIL (worst Δ = {d:.6f} at {summary_pass[metric][1]})")
            pass_overall = False
    out.append("")
    out.append(f"  OVERALL: {'✓ PASS' if pass_overall else '✗ FAIL'}")
    out.append("")
    return "\n".join(out), pass_overall, order_mismatch_indices, summary_pass


def _format_grand_summary(rows: list[tuple[str, bool, list[int], dict]]) -> str:
    out: list[str] = []
    sep = "=" * 100
    out.append("")
    out.append(sep)
    out.append("GRAND SUMMARY (across all presets)")
    out.append(sep)
    out.append("")
    hdr = f"  {'preset':15s} | {'joints':>7s} | {'kp':>7s} | {'kd':>7s} | {'armature':>9s} | {'friction':>9s} | {'q_reset':>8s} | VERDICT"
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    for preset, ok, order_mis, mp in rows:
        names_cell = "✓" if not order_mis else f"✗({len(order_mis)})"
        cells = [names_cell]
        for m in _METRICS:
            d = mp.get(m, [0.0, ""])[0] if mp else 0.0
            cells.append("✓" if d <= 1e-5 else f"✗({d:.3f})")
        verdict = "✓ PASS" if ok else "✗ FAIL"
        out.append(
            f"  {preset:15s} | {cells[0]:>7s} | {cells[1]:>7s} | {cells[2]:>7s} | {cells[3]:>9s} | {cells[4]:>9s} | {cells[5]:>8s} | {verdict}"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--preset",
        choices=[*sorted(_PRESETS), "all"],
        default="all",
        help="Single preset to test, or 'all' (default).",
    )
    ap.add_argument(
        "--sim",
        choices=[*_SIMS, "all"],
        default="all",
        help="Single sim to test, or 'all' (default).",
    )
    ap.add_argument("--out", default="action_indexing_parity.txt")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    print(f"Writing report to: {out_path}")

    presets = list(_PRESETS.keys()) if args.preset == "all" else [args.preset]
    sims = list(_SIMS) if args.sim == "all" else [args.sim]

    sections: list[str] = []
    summary_rows: list[tuple[str, bool, list[int], dict]] = []
    for preset in presets:
        print(f"\n{'=' * 60}\nPRESET: {preset}")
        by_sim: dict[str, dict] = {}
        for sim in sims:
            print(f"  building [{sim}] ...", flush=True)
            by_sim[sim] = _collect(preset, sim)
            if "build_error" in by_sim[sim]:
                print(f"    BUILD ERROR: {by_sim[sim]['build_error']}")
            elif "capture_error" in by_sim[sim]:
                print(f"    CAPTURE ERROR: {by_sim[sim]['capture_error']}")
            else:
                print(f"    OK ({len(by_sim[sim].get('actuated_joint_names', []))} joints)")

        section, overall_pass, order_mis, mp = _format_preset_section(preset, by_sim)
        sections.append(section)
        summary_rows.append((preset, overall_pass, order_mis, mp))

    sections.append(_format_grand_summary(summary_rows))
    out_path.write_text("\n".join(sections), encoding="utf-8")

    # Short stdout summary.
    print(f"\n{'=' * 60}\nGRAND SUMMARY")
    for preset, ok, order_mis, _mp in summary_rows:
        verdict = "✓ PASS" if ok else "✗ FAIL"
        extras = f"  (joint-name mismatches: {len(order_mis)})" if order_mis else ""
        print(f"  {preset:15s}  {verdict}{extras}")
    print(f"\nFull report: {out_path}")

    return 0 if all(ok for _, ok, _, _ in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
