"""Compare g1_29dof Newton training paths: Implicit vs DelayedPD actuator.

User reports training works with ``DelayedPDActuator`` but **not** with
``ImplicitActuator`` on the same g1_29dof Newton preset. This script
builds the env twice — once with each actuator class, holding Kp / Kd /
armature / frictionloss identical — and dumps everything that could
plausibly differ at the manager / Newton-model / control-buffer level
between the two runs.

What is dumped, per variant:

  1. Actuator config that was passed to the scene
     (class, target_names_expr, stiffness, damping, armature,
     frictionloss, effort_limit, velocity_limit, delay params).
  2. Action-manager state post-build
     (has_explicit_actuators, total_action_dim, registered actuator
     subgroups + their per-joint mean stiffness/damping/effort_limit,
     manager-side scale/offset/clip stats).
  3. Newton model state post-build
     (joint_target_ke / joint_target_kd / joint_armature /
     joint_friction stats; joint_axis_mode / joint_target_mode /
     joint_dof_dim if present — these decide whether mjwarp creates an
     internal PD actuator per DOF).
  4. Post-reset initial state
     (joint_pos / joint_vel / base_pos_z, env 0).
  5. Per-step trace over a deterministic scripted action sequence:
     for each step we dump the raw action, the resulting
     ``control.joint_target_pos`` and ``control.joint_f`` (one of the
     two will be all-zero depending on path), and the resulting
     joint_pos / joint_vel / base_pos_z so we can see whether the
     simulator actually moves the robot.
  6. Side-by-side ``implicit − delayed`` per-step deltas at the end.

The script's purpose is descriptive, not prescriptive: it tells you
*where* the two paths diverge so you can decide whether the cause is
(a) ke/kd not being written to the Newton model on the Implicit path,
(b) joint_target_pos never being consumed because joint_axis_mode is
wrong, (c) effort_limit / scale mismatch driving the target to an
unreachable region, or (d) something else entirely.

Usage:
    python -m rlworld.scripts.diag.check_actuator_train_parity
    python -m rlworld.scripts.diag.check_actuator_train_parity --steps 10 --action-scale 0.3
    python -m rlworld.scripts.diag.check_actuator_train_parity --out diag/foo.txt
"""

from __future__ import annotations

import argparse
import importlib
import os
import traceback
from pathlib import Path

# Multi-build in one process — bypass the single-backend import guard.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np
import torch
import warp as wp

from rlworld.rl.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from rlworld.rl.runners import BaseRunner

_PRESET_MOD = "rlworld.rl.configs.presets.g1_29dof.base"
_PRESET_CLS = "G1FlatConfig"
_SIM = "newton"
_OUT_DEFAULT = "diag/actuator_train_parity.txt"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _build_env(actuator_variant: str, num_envs: int, seed: int):
    """Build g1_29dof Newton env with the requested actuator class.

    ``actuator_variant`` ∈ {"delayed", "implicit"}. The preset ships
    with ``DelayedPDActuatorCfg``; for the implicit run we swap it
    out for ``ImplicitActuatorCfg`` carrying the same gains so any
    behavioural delta isolates to the actuator path itself.
    """
    mod = importlib.import_module(_PRESET_MOD)
    cls = getattr(mod, _PRESET_CLS)
    preset = cls(sim_type=_SIM, num_envs=num_envs)
    cfgs = preset.build()

    # Pin seed via env cfg so set_seed inside NewtonEnv sees it.
    cfgs.env.seed = seed

    # Locate and patch the robot's actuator(s).
    robot_entity = cfgs.scene.entities["robot"]
    art = robot_entity.articulation
    old_acts = list(art.actuators)
    if len(old_acts) != 1:
        raise RuntimeError(f"Expected exactly one actuator in g1_29dof preset, got {len(old_acts)}: {old_acts}")
    old = old_acts[0]
    if not isinstance(old, DelayedPDActuatorCfg):
        raise RuntimeError(f"Expected preset's actuator to be DelayedPDActuatorCfg, got {type(old).__name__}")

    if actuator_variant == "delayed":
        new = old  # baseline: untouched
    elif actuator_variant == "implicit":
        new = ImplicitActuatorCfg(
            target_names_expr=old.target_names_expr,
            stiffness=old.stiffness,
            damping=old.damping,
            armature=old.armature,
            frictionloss=old.frictionloss,
            effort_limit=old.effort_limit,
            velocity_limit=old.velocity_limit,
        )
    else:
        raise ValueError(f"unknown actuator_variant={actuator_variant!r}")

    art.actuators = (new,)

    runner = BaseRunner.create_with_env(cfgs)
    return runner.env, new


# ---------------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------------


def _fmt_field(v) -> str:
    if isinstance(v, dict):
        if len(v) > 4:
            head = list(v.items())[:4]
            return f"dict({len(v)} entries) head={head}"
        return f"dict({v})"
    return repr(v)


def _dump_actuator_cfg(act_cfg) -> str:
    lines = [f"  class: {type(act_cfg).__name__}"]
    for fname in (
        "target_names_expr",
        "stiffness",
        "damping",
        "armature",
        "frictionloss",
        "effort_limit",
        "velocity_limit",
        "min_delay",
        "max_delay",
    ):
        if hasattr(act_cfg, fname):
            lines.append(f"  {fname}: {_fmt_field(getattr(act_cfg, fname))}")
    return "\n".join(lines)


def _dump_manager_state(env) -> str:
    mgr = env.act_manager
    lines = []
    lines.append(f"  has_explicit_actuators: {mgr.has_explicit_actuators}")
    lines.append(f"  total_action_dim: {mgr._total_action_dim}")
    lines.append(f"  num actuated joints: {len(mgr.actuated_joint_names)}")
    lines.append(f"  num registered actuator subgroups: {len(mgr._actuators)}")
    for i, (act, jidx) in enumerate(mgr._actuators):
        st_str = "—"
        kp_str = "—"
        kd_str = "—"
        eff_str = "—"
        if hasattr(act, "stiffness"):
            t = act.stiffness
            kp_str = (
                f"shape={tuple(t.shape)} mean={t.mean().item():.2f} min={t.min().item():.2f} max={t.max().item():.2f}"
            )
        if hasattr(act, "damping"):
            t = act.damping
            kd_str = (
                f"shape={tuple(t.shape)} mean={t.mean().item():.2f} min={t.min().item():.2f} max={t.max().item():.2f}"
            )
        if hasattr(act, "effort_limit"):
            t = act.effort_limit
            if isinstance(t, torch.Tensor):
                eff_str = f"shape={tuple(t.shape)} mean={t.mean().item():.2f}"
            else:
                eff_str = f"{t}"
        lines.append(f"  actuator[{i}] class={type(act).__name__} n_joints={jidx.numel()}")
        lines.append(f"    Kp: {kp_str}")
        lines.append(f"    Kd: {kd_str}")
        lines.append(f"    effort_limit: {eff_str}")
    # Manager-side action-pipeline tensors (scale/offset/clip).
    for name in ("_scale", "_offset", "_clip_low", "_clip_high"):
        if hasattr(mgr, name):
            t = getattr(mgr, name)
            if isinstance(t, torch.Tensor):
                lines.append(
                    f"  manager.{name[1:]}: shape={tuple(t.shape)} mean={t.float().mean().item():.4f} "
                    f"min={t.float().min().item():.4f} max={t.float().max().item():.4f}"
                )
            else:
                lines.append(f"  manager.{name[1:]}: {t!r}")
    return "\n".join(lines)


def _wp_to_np(arr):
    """Convert a warp array (possibly None) to numpy; return None if not present."""
    if arr is None:
        return None
    try:
        return wp.to_torch(arr).detach().cpu().numpy()
    except Exception:
        try:
            return np.asarray(arr)
        except Exception:
            return None


def _stats_line(label: str, a) -> str:
    if a is None:
        return f"  {label}: <not present>"
    if a.size == 0:
        return f"  {label}: empty"
    nz = int(np.count_nonzero(a))
    return (
        f"  {label}: shape={tuple(a.shape)} n={a.size} mean={float(a.mean()):.4f} "
        f"min={float(a.min()):.4f} max={float(a.max()):.4f} nonzero={nz}/{a.size}"
    )


def _dump_newton_model(env) -> str:
    sm = env.scene_manager
    model = sm.model
    lines = []
    for name in (
        "joint_target_ke",
        "joint_target_kd",
        "joint_armature",
        "joint_friction",
        "joint_effort_limit",
        "joint_velocity_limit",
    ):
        arr = getattr(model, name, None)
        lines.append(_stats_line(name, _wp_to_np(arr)))
    # Mode-style fields: surface whatever exists.
    for name in (
        "joint_axis_mode",
        "joint_target_mode",
        "joint_mode",
        "joint_dof_count",
        "joint_dof_dim",
        "joint_type",
    ):
        if hasattr(model, name):
            try:
                attr = getattr(model, name)
                if isinstance(attr, int):
                    lines.append(f"  {name}: {attr}")
                else:
                    np_v = _wp_to_np(attr)
                    if np_v is None:
                        lines.append(f"  {name}: <unreadable>")
                    elif np_v.dtype.kind in ("i", "u"):
                        unique = sorted(set(np_v.flatten().tolist()))
                        if len(unique) <= 8:
                            lines.append(f"  {name}: shape={tuple(np_v.shape)} unique={unique}")
                        else:
                            lines.append(
                                f"  {name}: shape={tuple(np_v.shape)} unique_count={len(unique)} "
                                f"sample={unique[:8]}"
                            )
                    else:
                        lines.append(_stats_line(name, np_v))
            except Exception as e:
                lines.append(f"  {name}: <err: {type(e).__name__}: {e}>")
    return "\n".join(lines)


def _read_control_buffers(env) -> dict:
    """Snapshot the two control buffers that distinguish the paths."""
    control = env.scene_manager.control
    tp = _wp_to_np(control.joint_target_pos)
    jf = _wp_to_np(control.joint_f)
    return {"joint_target_pos": tp, "joint_f": jf}


def _format_control_buffers(snap: dict, prefix: str = "    ") -> str:
    lines = []
    for label in ("joint_target_pos", "joint_f"):
        a = snap[label]
        if a is None:
            lines.append(f"{prefix}control.{label}: <not present>")
            continue
        nz = int(np.count_nonzero(a))
        lines.append(
            f"{prefix}control.{label}: n={a.size} abs_max={float(np.abs(a).max()):.5f} "
            f"mean={float(a.mean()):.5f} nonzero={nz}/{a.size}"
        )
    return "\n".join(lines)


def _read_robot_state(env) -> dict:
    rd = env.get_robot_data()
    out = {
        "joint_pos": rd.joint_pos[0].detach().cpu().numpy(),
        "joint_vel": rd.joint_vel[0].detach().cpu().numpy(),
    }
    try:
        # Body 0 == root for floating-base robots.
        ids = torch.tensor([0], device=env.device, dtype=torch.long)
        out["base_pos"] = rd.body_pos_w_by_ids(ids)[0, 0].detach().cpu().numpy()
    except Exception:
        out["base_pos"] = None
    return out


def _fmt_state(s: dict) -> str:
    lines = []
    jp = s["joint_pos"]
    jv = s["joint_vel"]
    lines.append(
        f"    joint_pos: mean={jp.mean():.4f} min={jp.min():.4f} max={jp.max():.4f} abs_max={np.abs(jp).max():.4f}"
    )
    lines.append(f"    joint_vel: mean={jv.mean():.4f} abs_max={np.abs(jv).max():.4f}")
    bp = s.get("base_pos")
    if bp is not None:
        lines.append(f"    base_pos: x={bp[0]:.4f} y={bp[1]:.4f} z={bp[2]:.4f}")
    else:
        lines.append("    base_pos: <unreadable>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--action-scale",
        type=float,
        default=0.2,
        help="Scale on uniform(-1, 1) scripted action — keep modest to stay near nominal pose.",
    )
    ap.add_argument("--out", default=_OUT_DEFAULT)
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    sections.append("=" * 110)
    sections.append("g1_29dof Newton actuator parity diag — DelayedPDActuator vs ImplicitActuator")
    sections.append(f"seed={args.seed}  steps={args.steps}  num_envs={args.num_envs}  action_scale={args.action_scale}")
    sections.append("=" * 110)

    runs: dict[str, dict] = {}
    for variant in ("delayed", "implicit"):
        sections.append("")
        sections.append("─" * 110)
        sections.append(f"VARIANT  [{variant}]")
        sections.append("─" * 110)
        try:
            env, act_cfg = _build_env(variant, args.num_envs, args.seed)
        except Exception as e:
            sections.append(f"BUILD FAILED: {type(e).__name__}: {e}")
            sections.append(traceback.format_exc())
            continue

        sections.append(f"\n[{variant}] actuator cfg passed to scene:")
        sections.append(_dump_actuator_cfg(act_cfg))

        sections.append(f"\n[{variant}] action manager state (post-build):")
        sections.append(_dump_manager_state(env))

        sections.append(f"\n[{variant}] Newton model state (post-build):")
        sections.append(_dump_newton_model(env))

        # Reset, then snapshot initial state + control buffers (pre-step).
        env.reset()
        sections.append(f"\n[{variant}] state after env.reset() (env 0):")
        sections.append(_fmt_state(_read_robot_state(env)))
        sections.append(f"\n[{variant}] control buffers immediately after reset (env 0):")
        sections.append(_format_control_buffers(_read_control_buffers(env)))

        # Deterministic scripted action sequence — same RNG seed yields
        # the same actions across variants so any state delta is the
        # actuator path, not the input.
        g = torch.Generator(device="cpu").manual_seed(args.seed + 12345)
        actions = []
        for _ in range(args.steps):
            a = (torch.rand(env.num_envs, env.num_actions, generator=g, device="cpu") * 2.0 - 1.0) * args.action_scale
            actions.append(a.to(env.device))

        sections.append(f"\n[{variant}] per-step trace:")
        per_step: list[dict] = []
        for step_idx, action in enumerate(actions):
            try:
                env.step(action)
            except Exception as e:
                sections.append(f"  STEP {step_idx} step() FAILED: {type(e).__name__}: {e}")
                sections.append(traceback.format_exc())
                break
            ctrl = _read_control_buffers(env)
            state = _read_robot_state(env)
            per_step.append({"action": action[0].detach().cpu().numpy(), "ctrl": ctrl, "state": state})
            a = per_step[-1]["action"]
            sections.append(f"  STEP {step_idx}")
            sections.append(
                f"    input action: mean={a.mean():+.4f} abs_max={np.abs(a).max():.4f} "
                f"(first 4: {np.array2string(a[:4], precision=3)})"
            )
            sections.append(_format_control_buffers(ctrl))
            sections.append(_fmt_state(state))

        runs[variant] = {"per_step": per_step}

        # Free GPU resources before next build.
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Side-by-side delta summary ────────────────────────────────────
    if "delayed" in runs and "implicit" in runs:
        d = runs["delayed"]["per_step"]
        i = runs["implicit"]["per_step"]
        n = min(len(d), len(i))
        sections.append("")
        sections.append("=" * 110)
        sections.append("SIDE-BY-SIDE  per-step deltas  (implicit − delayed; max-abs across joints)")
        sections.append("=" * 110)
        sections.append(
            f"  {'step':>4s} | {'Δjoint_pos':>14s} | {'Δjoint_vel':>14s} | {'Δbase_z':>12s} | "
            f"{'Δctrl.joint_target_pos':>26s} | {'Δctrl.joint_f':>16s}"
        )
        sections.append("  " + "-" * 108)
        for k in range(n):
            sd = d[k]["state"]
            si = i[k]["state"]
            dpos = float(np.abs(si["joint_pos"] - sd["joint_pos"]).max())
            dvel = float(np.abs(si["joint_vel"] - sd["joint_vel"]).max())
            if sd.get("base_pos") is not None and si.get("base_pos") is not None:
                dbz = float(si["base_pos"][2] - sd["base_pos"][2])
                dbz_str = f"{dbz:+.5f}"
            else:
                dbz_str = "—"
            ctrl_d = d[k]["ctrl"]
            ctrl_i = i[k]["ctrl"]
            if ctrl_d["joint_target_pos"] is not None and ctrl_i["joint_target_pos"] is not None:
                d_tp = float(np.abs(ctrl_i["joint_target_pos"] - ctrl_d["joint_target_pos"]).max())
                d_tp_str = f"{d_tp:.5f}"
            else:
                d_tp_str = "—"
            if ctrl_d["joint_f"] is not None and ctrl_i["joint_f"] is not None:
                d_jf = float(np.abs(ctrl_i["joint_f"] - ctrl_d["joint_f"]).max())
                d_jf_str = f"{d_jf:.4f}"
            else:
                d_jf_str = "—"
            sections.append(
                f"  {k:>4d} | {dpos:>14.5f} | {dvel:>14.5f} | {dbz_str:>12s} | " f"{d_tp_str:>26s} | {d_jf_str:>16s}"
            )

    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
