"""Step B — does the arm actually interact with the things on its bench?

Step A established the arm alone and earlier work established the props
alone; the two have never been in the same scene. Everything measured
here is about the *coupling*, which no previous diag touches:

* **They coexist.** Arm, bench and workpiece build together and read back
  through the ordinary accessors, in the right registries.
* **The bench holds the workpiece**, at the height the geometry implies.
* **The arm is not standing in the bench.** At rest the arm-vs-bench
  contact force is zero; a mount sunk into the table would put a constant
  force into every contact-derived signal downstream.
* **The bench stops the arm.** Driven down into it, the arm is held above
  the surface rather than passing through — the first thing a lift task
  would exploit if it were wrong.
* **The arm moves the workpiece.** Contact *sensing* was verified before;
  contact *force transfer* from an articulation into a rigid object never
  was. Sweep the arm through the cube and it has to move.
* **The gripper holds the workpiece.** Jaw gap admits the cube when open
  and closes past it; with the cube between the pads and the gripper
  closed, raising the arm has to raise the cube too.
* **Reset restores the workpiece**, so an episode that knocked it away
  starts the next one from the bench.

Run all three and cross-compare::

    python -m jaxrlworld.scripts.diag.yam.yam_workspace_smoke --num-envs 4

or one backend::

    python -m jaxrlworld.scripts.diag.yam.yam_workspace_smoke --sim mujoco --num-envs 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from jaxrlworld.rl.configs.presets.yam_arm.base import (
    CUBE_HALF,
    TABLE_TOP_Z,
    YamArmConfig,
)
from jaxrlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.rl.utils.quat_utils import quat_rotate_wxyz

_SIMS = ("genesis", "newton", "mujoco")

GRIPPER_JOINT = "left_finger"
WRIST_BODY = "link_6"
LEFT_PAD_BODY = "lf_down"
RIGHT_PAD_BODY = "rf_down"
CUBE_SIZE = 2 * CUBE_HALF

# Actions are scaled per joint, and the arm's scales are small: joint2
# moves 0.16 rad per unit of action against a 3.67 rad range. A unit
# command therefore barely leaves the home pose, and a reach measured
# that way reports the command's limit rather than the arm's. Clipping
# allows +-100, so drive the arm with a magnitude that can actually span
# its travel.
ARM_DRIVE = 15.0

# The model's own declared grasp point, as the ``grasp_site`` offset in
# link_6's frame. Sites are not readable through ``site_pos_w`` on two of
# the three backends, so it is applied by hand — which is still better
# than inferring a grasp point from body origins, because the pads sit on
# a linkage and their origins are not where they hold anything.
GRASP_OFFSET_IN_WRIST = (0.0, -0.03, 0.1247)


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):+.5f}" for x in v) + "]"


# ══════════════════════════════════════════════════════════════════════════
# Scene
# ══════════════════════════════════════════════════════════════════════════


def _build_env(sim: str, num_envs: int):
    cfgs = YamArmConfig(sim_type=sim, num_envs=num_envs).build()

    def _vs(name: str, other: str) -> ContactSensorCfg:
        return ContactSensorCfg(
            name=name,
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            # "any part of the other entity" — one spelling on all three
            # backends, which is what mode="entity" exists for.
            secondary=ContactMatch(mode="entity", entity=other),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            history_length=cfgs.env.decimation,
        )

    field = "sensors" if sim == "mujoco" else "contact_sensors"
    existing = tuple(getattr(cfgs.scene, field) or ())
    # One column per pad, so "both jaws are on it" is directly observable
    # instead of inferred from positions.
    pads = ContactSensorCfg(
        name="pads_vs_cube",
        primary=ContactMatch(mode="body", pattern="[lr]f_down", entity="robot"),
        secondary=ContactMatch(mode="entity", entity="cube"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=cfgs.env.decimation,
    )
    added = [_vs("arm_table_contact", "table"), _vs("arm_cube_contact", "cube"), pads]
    setattr(cfgs.scene, field, list(existing) + added)

    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return env, cfgs


def _step_n(env, action, n: int) -> bool:
    reset_seen = False
    for _ in range(n):
        _, _, dones, _, _ = env.step(action)
        if bool(dones.any()):
            reset_seen = True
    env._invalidate_cache()
    return reset_seen


def _step_watching(env, action, n: int, group: str) -> tuple[float, bool]:
    """Step, sampling a contact group every step.

    A collision during a sweep is over by the time the sweep ends, so a
    single reading afterwards reports zero for a contact that certainly
    happened.
    """
    peak = 0.0
    seen = False
    for _ in range(n):
        env.step(action)
        env._invalidate_cache()
        peak = max(peak, float(env.contact_manager.contact_force(group).norm(dim=-1).max()))
        seen = seen or bool(env.contact_manager.is_contact(group).any())
    return peak, seen


# ══════════════════════════════════════════════════════════════════════════
# The diag
# ══════════════════════════════════════════════════════════════════════════


def run_single(sim: str, num_envs: int, settle: int) -> dict:
    env, cfgs = _build_env(sim, num_envs)
    results: dict[str, bool] = {}
    measured: dict[str, object] = {}

    arm = env.get_entity_data("robot")
    cube = env.get_entity_data("cube")
    table = env.get_entity_data("table")
    all_ids = torch.arange(env.num_envs, device=env.device)
    n_act = env.act_manager.num_actions
    act_names = list(env.act_manager.actuated_joint_names)
    grip_idx = act_names.index(GRIPPER_JOINT)
    zeros = torch.zeros(env.num_envs, n_act, device=env.device)

    def _body(name: str) -> torch.Tensor:
        return arm.body_pos_w((name,))[:, 0, :]

    def _pad_mid() -> torch.Tensor:
        return 0.5 * (_body(LEFT_PAD_BODY) + _body(RIGHT_PAD_BODY))

    wrist_idx = arm.find_body_index(WRIST_BODY)
    grasp_offset = torch.tensor(GRASP_OFFSET_IN_WRIST, device=env.device).expand(env.num_envs, 3)

    def _grasp_point() -> torch.Tensor:
        """Where the jaws actually hold something, in world coordinates."""
        wrist = _body(WRIST_BODY)
        quat = arm.body_quat_w_all[:, wrist_idx, :]
        return wrist + quat_rotate_wxyz(quat, grasp_offset)

    def _pad_gap() -> float:
        return float((_body(LEFT_PAD_BODY) - _body(RIGHT_PAD_BODY)).norm(dim=-1).mean())

    print("=" * 78)
    print(f"YAM WORKSPACE DIAG  [sim={sim}]")
    print("=" * 78)

    # ── B1. the three of them coexist ────────────────────────────────────
    in_entities = "robot" in env.scene_manager.entities
    props = set(env.scene_manager.rigid_objects)
    print(f"[scene] entities = {sorted(env.scene_manager.entities)}   rigid_objects = {sorted(props)}")
    print(f"[scene] arm base = {_fmt(arm.root_link_pos_w[0])}")
    print(f"[scene] table    = {_fmt(table.root_link_pos_w[0])}   cube = {_fmt(cube.root_link_pos_w[0])}")
    results["scene_assembles"] = in_entities and props == {"table", "cube"}

    origins = env.scene_manager.env_origins
    _table_pos = cfgs.preset_kwargs.get("table_pos", YamArmConfig().table_pos)
    cfg_table_xy = (float(origins[0, 0]) + _table_pos[0], float(origins[0, 1]) + _table_pos[1])
    cube_expected_z = float(origins[0, 2]) + TABLE_TOP_Z + CUBE_HALF
    _step_n(env, zeros, settle)
    rest_z = cube.root_link_pos_w[:, 2]
    rest_err = float((rest_z - cube_expected_z).abs().max())
    print(f"[bench] cube rest z per env = {[round(float(v), 4) for v in rest_z]}")
    print(
        f"[bench] expected {cube_expected_z:.4f} (table top {TABLE_TOP_Z} + half cube {CUBE_HALF})  err = {rest_err:.4f}"
    )
    results["table_supports_cube"] = rest_err < 2e-3
    measured["cube_rest_z"] = round(float(rest_z[0]), 5)

    # ── B2. the mount is not inside the bench ────────────────────────────
    f_table = float(env.contact_manager.contact_force("arm_table_contact").norm(dim=-1).max())
    print(f"[bench] arm-vs-table force at rest = {f_table:.4f} N (expect 0)")
    results["arm_clears_table_at_rest"] = f_table < 1e-3
    measured["arm_table_force_at_rest"] = round(f_table, 5)

    # ── B3. reach: which way is down, and how far down does it go? ───────
    # Never assume a joint's sign: it is a property of the model's frames.
    # Getting it backwards drives the arm UP and then reports, truthfully
    # but uselessly, that it did not go through the table.
    def _sign_that_lowers(joint: str) -> float:
        env.reset()
        _step_n(env, zeros, settle)
        z0 = float(_body(WRIST_BODY)[:, 2].mean())
        probe = zeros.clone()
        probe[:, act_names.index(joint)] = ARM_DRIVE
        _step_n(env, probe, settle * 3)
        return -1.0 if float(_body(WRIST_BODY)[:, 2].mean()) > z0 else 1.0

    s2 = _sign_that_lowers("joint2")
    s3 = _sign_that_lowers("joint3")
    s4 = _sign_that_lowers("joint4")
    print(f"[reach] sign that lowers the wrist: joint2 {s2:+.0f}  joint3 {s3:+.0f}  joint4 {s4:+.0f}")
    measured["lower_sign_joint2"] = s2
    measured["lower_sign_joint3"] = s3
    measured["lower_sign_joint4"] = s4

    env.reset()
    down = zeros.clone()
    down[:, act_names.index("joint2")] = s2 * ARM_DRIVE
    down[:, act_names.index("joint3")] = s3 * ARM_DRIVE
    down[:, act_names.index("joint4")] = s4 * ARM_DRIVE
    table_top_w = float(origins[0, 2]) + TABLE_TOP_Z
    # Track the LOWEST point reached, not the final one: the arm can swing
    # through its closest approach and settle somewhere higher.
    f_press = 0.0
    lowest = float("inf")
    for _ in range(settle * 10):
        env.step(down)
        env._invalidate_cache()
        lowest = min(lowest, float(_grasp_point()[:, 2].mean()))
        f_press = max(f_press, float(env.contact_manager.contact_force("arm_table_contact").norm(dim=-1).max()))
    wrist_z = _body(WRIST_BODY)[:, 2]
    grasp_z = _grasp_point()[:, 2]
    print(f"[reach] pressed down: wrist z = {float(wrist_z.mean()):.4f}   grasp point z = {float(grasp_z.mean()):.4f}")
    print(f"[reach] lowest grasp point reached = {lowest:.4f}   (cube top sits at {table_top_w + CUBE_SIZE:.4f})")
    print(f"[reach] table top = {table_top_w:.4f}   peak arm-vs-table force = {f_press:.3f} N")
    measured["lowest_grasp_z"] = round(lowest, 5)
    # Two separate facts. The arm has to be ABLE to bring its grasp point to
    # the bench (otherwise no lift task is possible here at all), and it must
    # not be able to pass through it.
    # Driving joints to their stops is one pose, not the reachable set, and
    # the force above shows it is mostly the arm hitting the bench. Answer
    # reachability by sampling instead: does ANY pose put the grasp point
    # down at a cube standing on the bench, with the point over the bench?
    torch.manual_seed(0)
    tx, ty = cfg_table_xy
    best_reach = float("inf")
    probe_action = zeros.clone()
    for _ in range(60):
        probe_action.uniform_(-ARM_DRIVE, ARM_DRIVE)
        probe_action[:, grip_idx] = 1.0
        _step_n(env, probe_action, 10)
        gp = _grasp_point()
        over_bench = (gp[:, 0] - tx).abs() < 0.6 - CUBE_HALF
        over_bench &= (gp[:, 1] - ty).abs() < 0.4 - CUBE_HALF
        if bool(over_bench.any()):
            best_reach = min(best_reach, float(gp[over_bench, 2].min()))
    cube_top = table_top_w + CUBE_SIZE
    print(f"[reach] best grasp-point height over the bench across 60 sampled poses = {best_reach:.4f}")
    print(f"[reach] a cube on the bench occupies {table_top_w:.4f} .. {cube_top:.4f}")
    results["gripper_reaches_the_bench"] = best_reach < cube_top
    measured["best_sampled_reach_z"] = round(best_reach, 5)
    results["arm_does_not_sink_through_table"] = bool((wrist_z > table_top_w - 0.05).all())
    measured["wrist_z_pressed"] = round(float(wrist_z.mean()), 5)
    measured["grasp_z_pressed"] = round(float(grasp_z.mean()), 5)
    measured["arm_table_force_pressed"] = round(f_press, 4)

    env.reset()
    _step_n(env, zeros, settle)

    # ── B4. the arm moves the workpiece ──────────────────────────────────
    # Put the cube where the wrist already is, then keep driving: contact
    # sensing was proven before, force TRANSFER from an articulation into a
    # rigid object was not.
    writer = env.get_root_state_writer("cube")
    quat = torch.zeros(env.num_envs, 4, device=env.device)
    quat[:, 0] = 1.0
    z3 = torch.zeros(env.num_envs, 3, device=env.device)

    def _put_cube(pos: torch.Tensor) -> None:
        writer.set_root_pose(pos, quat, env_ids=all_ids)
        writer.set_root_velocity(z3, z3, env_ids=all_ids)
        writer.eval_fk(env_ids=all_ids)
        env._invalidate_cache()

    # On the bench, at the grasp point's own xy, with the arm already low.
    # Left floating between the open jaws it simply falls out before
    # anything touches it, which measures gravity rather than the arm.
    open_action = zeros.clone()
    open_action[:, grip_idx] = 1.0
    lowered = open_action.clone()
    lowered[:, act_names.index("joint2")] = s2 * ARM_DRIVE
    lowered[:, act_names.index("joint3")] = s3 * ARM_DRIVE
    lowered[:, act_names.index("joint4")] = s4 * ARM_DRIVE
    _step_n(env, lowered, settle * 8)

    on_bench = _grasp_point().clone()
    on_bench[:, 2] = float(origins[0, 2]) + TABLE_TOP_Z + CUBE_HALF
    _put_cube(on_bench)
    _step_n(env, lowered, 2)
    before_push = cube.root_link_pos_w.clone()

    sweep = lowered.clone()
    sweep[:, act_names.index("joint1")] = ARM_DRIVE
    f_cube, touched = _step_watching(env, sweep, settle * 6, "arm_cube_contact")
    after_push = cube.root_link_pos_w
    shifted = float((after_push - before_push)[:, :2].norm(dim=-1).max())
    print(f"[push]  cube moved {shifted:.4f} m horizontally while the jaws swept sideways")
    print(f"[push]  arm-vs-cube contact seen = {touched}   peak |force| = {f_cube:.3f} N")
    results["arm_pushes_cube"] = shifted > 5e-3
    results["arm_cube_contact_detected"] = touched and f_cube > 1e-3
    measured["cube_push_distance"] = round(shifted, 5)
    measured["arm_cube_force"] = round(f_cube, 4)

    # ── B5. the gripper holds the workpiece ──────────────────────────────
    lift_sign = -s2
    env.reset()
    # Sign convention, measured on the PADS (``lf_down`` / ``rf_down``, the
    # links carrying the contact spheres) rather than on the finger mounts.
    # The finger is a linkage, so mount separation and pad separation move
    # in opposite directions: a positive command spreads the pads.
    open_action = zeros.clone()
    open_action[:, grip_idx] = 1.0
    _step_n(env, open_action, settle * 3)
    gap_open = _pad_gap()

    close_probe = zeros.clone()
    close_probe[:, grip_idx] = -1.0
    _step_n(env, close_probe, settle * 3)
    gap_closed = _pad_gap()
    print(f"[grasp] pad gap: open = {gap_open:.4f} m   closed = {gap_closed:.4f} m   cube = {CUBE_SIZE:.3f} m")
    # Where the declared grasp point sits relative to the pads. If it is not
    # roughly equidistant and inside the gap, the cube is being teleported
    # into a pad rather than between them, and every grasp number below is
    # measuring an ejection.
    _step_n(env, open_action, settle * 2)
    gp = _grasp_point()
    d_left = float((gp - _body(LEFT_PAD_BODY)).norm(dim=-1).mean())
    d_right = float((gp - _body(RIGHT_PAD_BODY)).norm(dim=-1).mean())
    print(f"[grasp] grasp point to pads: left = {d_left:.4f} m   right = {d_right:.4f} m")
    # Reported, not asserted. The pad BODY origins are not the contact
    # surfaces — the spheres sit some 66 mm further along the finger — so
    # "is the grasp point between them" cannot be decided from these two
    # numbers. Whether the anchor is right is settled by the lift below.
    measured["grasp_to_left_pad"] = round(d_left, 5)
    measured["grasp_to_right_pad"] = round(d_right, 5)
    results["pad_gap_admits_cube"] = gap_closed < CUBE_SIZE < gap_open
    measured["pad_gap_open"] = round(gap_open, 5)
    measured["pad_gap_closed"] = round(gap_closed, 5)

    # WHERE the jaws close on something is measured, not assumed: hold the
    # cube at a series of depths along the wrist's approach axis, close, and
    # read which pads report contact. Position arithmetic cannot settle this
    # — the pads ride a linkage and their body origins are some 66 mm short
    # of the contact spheres.
    def _point_at(depth: float) -> torch.Tensor:
        off = torch.tensor([0.0, -0.03, depth], device=env.device).expand(env.num_envs, 3)
        return _body(WRIST_BODY) + quat_rotate_wxyz(arm.body_quat_w_all[:, wrist_idx, :], off)

    def _pinch_at(depth: float) -> tuple[bool, bool, float]:
        """Pin the cube at ``depth``, close the jaws, report each pad."""
        env.reset()
        _step_n(env, open_action, settle * 3)
        for _ in range(settle * 2):
            _put_cube(_point_at(depth))
            env.step(close_probe)
            env._invalidate_cache()
        cols = env.contact_manager.is_contact("pads_vs_cube").reshape(env.num_envs, -1)
        force = float(env.contact_manager.contact_force("pads_vs_cube").norm(dim=-1).max())
        return bool(cols[:, 0].all()), bool(cols[:, 1].all()), force

    print(f"[grasp] pad sensor columns = {env.contact_manager.tracked_names('pads_vs_cube')}")
    best_depth = None
    for d in (0.06, 0.08, 0.10, 0.1247, 0.14):
        a, b, f = _pinch_at(d)
        print(f"[grasp]   depth {d:.4f} m: pads = [{a}, {b}]   |force| = {f:.3f} N")
        if a and b and best_depth is None:
            best_depth = d
    measured["grasp_depth_that_pinches"] = best_depth
    results["jaws_close_on_the_cube"] = best_depth is not None

    # Grip where it actually pinched; fall back to the model's declared
    # grasp site so the lift below still produces a number either way.
    depth = best_depth if best_depth is not None else GRASP_OFFSET_IN_WRIST[2]
    env.reset()
    _step_n(env, open_action, settle * 3)
    for _ in range(settle * 2):
        _put_cube(_point_at(depth))
        env.step(close_probe)
        env._invalidate_cache()
    # Released: from here the grip alone has to keep it.
    _step_n(env, close_probe, settle)
    held_z0 = cube.root_link_pos_w[:, 2].clone()
    grip0 = _point_at(depth)
    print(f"[grasp] gripping at depth {depth:.4f} m")
    print(
        f"[grasp] after release, cube-to-grip-point = {float((cube.root_link_pos_w - grip0).norm(dim=-1).mean()):.4f} m"
    )

    lift = close_probe.clone()
    lift[:, act_names.index("joint2")] = 3.0 * lift_sign
    _step_n(env, lift, settle * 6)
    held_z1 = cube.root_link_pos_w[:, 2]
    grip1 = _point_at(depth)

    ee_rise = float((grip1 - grip0)[:, 2].mean())
    cube_rise = float((held_z1 - held_z0).mean())
    slip = float((cube.root_link_pos_w - grip1).norm(dim=-1).max())
    print(f"[grasp] gripper rose {ee_rise:+.4f} m, cube rose {cube_rise:+.4f} m")
    print(f"[grasp] cube-to-pad-centre distance after the lift = {slip:.4f} m")
    # Held means: it went up with the gripper and stayed between the pads.
    results["gripper_lifts_cube"] = cube_rise > 0.5 * abs(ee_rise) and ee_rise > 0.01
    results["cube_stays_between_pads"] = slip < 0.06
    measured["ee_rise"] = round(ee_rise, 5)
    measured["cube_rise"] = round(cube_rise, 5)
    measured["cube_slip"] = round(slip, 5)

    # ── B6. reset puts the workpiece back ────────────────────────────────
    env.reset()
    back = cube.root_link_pos_w
    spawn = origins + torch.tensor(cfgs.preset_kwargs.get("cube_pos", YamArmConfig().cube_pos), device=env.device)
    back_err = float((back - spawn).abs().max())
    print(f"[reset] cube back at {_fmt(back[0])}   expected {_fmt(spawn[0])}   err = {back_err:.4f}")
    results["reset_restores_cube"] = back_err < 1e-3

    # ── verdict ──────────────────────────────────────────────────────────
    print("=" * 78)
    print("VERDICT")
    ok = True
    for k, v in results.items():
        print(f"  {k:<34}: {'PASS' if v else 'FAIL'}")
        ok = ok and v
    print(f"  {'OVERALL':<34}: {'PASS' if ok else 'FAIL'}")
    print()
    print("REPORTED")
    for k, v in measured.items():
        print(f"  {k:<34}: {v}")
    print("=" * 78)
    return {"results": results, "measured": measured, "ok": ok}


# ══════════════════════════════════════════════════════════════════════════
# Cross-sim harness
# ══════════════════════════════════════════════════════════════════════════


def run_all(num_envs: int, settle: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="yam_workspace_"))
    out: dict[str, dict] = {}
    env_vars = dict(os.environ, JAXRLWORLD_ALLOW_MULTI_SIM="1")

    for sim in _SIMS:
        result_path = tmp / f"{sim}.json"
        cmd = [
            sys.executable,
            "-m",
            "jaxrlworld.scripts.diag.yam.yam_workspace_smoke",
            "--sim",
            sim,
            "--result-json",
            str(result_path),
            "--num-envs",
            str(num_envs),
            "--settle-steps",
            str(settle),
        ]
        print()
        print("#" * 78)
        print(f"# $ {' '.join(cmd)}")
        print("#" * 78)
        subprocess.run(cmd, env=env_vars, check=False)
        if result_path.exists():
            out[sim] = json.loads(result_path.read_text())

    if not out:
        print("No backend produced a result.")
        return 1

    keys: list[str] = []
    for r in out.values():
        for k in r["results"]:
            if k not in keys:
                keys.append(k)

    print()
    print("=" * 78)
    print("CROSS-SIM SUMMARY")
    print("=" * 78)
    print(f"{'check':<36}" + "".join(f"{s:>10}" for s in _SIMS))
    print("-" * 78)
    overall = True
    for k in keys:
        row = f"{k:<36}"
        for s in _SIMS:
            v = out.get(s, {}).get("results", {}).get(k)
            row += f"{'—' if v is None else ('PASS' if v else 'FAIL'):>10}"
            overall = overall and bool(v)
        print(row)

    mkeys: list[str] = []
    for r in out.values():
        for k in r["measured"]:
            if k not in mkeys:
                mkeys.append(k)
    print()
    print("REPORTED values (compared by hand)")
    for k in mkeys:
        vals = [str(out.get(s, {}).get("measured", {}).get(k)) for s in _SIMS]
        agree = "AGREE" if len(set(vals)) == 1 else "<-- DIFFER"
        print(f"  {k:<28}" + "".join(f"{v:>16}" for v in vals) + f"   {agree}")

    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 78)
    return 0 if overall else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=list(_SIMS), default=None, help="Run one backend. Omit to run all three.")
    ap.add_argument("--result-json", default=None, help="Child mode: where to write this backend's result.")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--settle-steps", type=int, default=25)
    args = ap.parse_args()

    if args.sim is None:
        return run_all(args.num_envs, args.settle_steps)

    result = run_single(args.sim, args.num_envs, args.settle_steps)
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
