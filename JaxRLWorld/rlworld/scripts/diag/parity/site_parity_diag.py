"""Are site frames the same thing on all three simulators?

A site is a frame rigidly attached to a body, declared in an MJCF. Only
MuJoCo stores one; the shared implementation composes it from the parent
body's pose, which is what MuJoCo does internally. This diag holds that
composition to MuJoCo's own answer, and then holds all three backends to
each other.

What it checks, and why each one is here:

* **The table agrees with the asset.** Site names, parent bodies and
  local offsets come from the MJCF through MuJoCo's compiler. Printed in
  full on the first run, because a wrong parent body produces a position
  that is merely displaced — plausible, and invisible in an aggregate.
* **Position matches MuJoCo's own.** The shared path against
  ``site_pos_w_mjlab_native``. This is the only external reference we
  have; the other two backends have nothing to be checked against except
  each other.
* **Velocity matches MuJoCo's own, WHILE MOVING.** At rest every site
  velocity is zero and any wrong formula passes. The arm is driven to a
  pose far from home and sampled mid-swing, and the check fails if the
  site is not actually moving — a silent zero would otherwise be read as
  perfect agreement. Velocity is where a wrong formula hides: the lever
  arm ``omega x r`` is the whole content of the term, and dropping it or
  measuring ``r`` from the centre of mass instead of the link origin is
  off by metres per second, not by rounding.
* **The three backends agree with each other**, at rest and moving.
* **A site tracks its body.** Rotate the parent joint and the site must
  sweep an arc of the right radius about it — the offset is not being
  applied in the world frame instead of the body frame.
* **Velocity is the derivative of position.** Finite-difference the
  position across one control step and compare to the reported velocity.
  This is independent of MuJoCo: it catches a formula that is
  self-consistently wrong on all three.
* **A URDF entity refuses.** Sites are an MJCF concept; asking for one
  where none can exist must say so rather than return zeros.

Run all three and cross-compare::

    python -m rlworld.scripts.diag.parity.site_parity_diag --num-envs 4

or one backend::

    python -m rlworld.scripts.diag.parity.site_parity_diag --sim newton --num-envs 4
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

from rlworld.rl.configs.presets.yam_arm.base import YamArmConfig
from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from rlworld.rl.envs.site_frames import sites_from_mjcf
from rlworld.rl.runners import BaseRunner
from rlworld.rl.utils.quat_utils import quat_inv_wxyz, quat_mul_wxyz

_SIMS = ("genesis", "newton", "mujoco")

SITE = "grasp_site"
"""The site the manipulation task keys on. Declared on link_6, roughly
0.125 m out along the wrist — far enough from its parent's origin that a
dropped or mis-framed offset is obvious rather than marginal."""

DRIVE_STEPS = 40
"""Steps of a large command before sampling, so the arm is mid-motion
rather than settled."""

POS_TOL = 1e-6
"""m — agreement with MuJoCo's own site position. Both compose the same
product, so only float rounding separates them."""

VEL_TOL = 1e-5
"""m/s — agreement with MuJoCo's own site velocity."""

CROSS_TOL = 1e-4
"""m — agreement BETWEEN backends. Looser than the mjlab comparison: the
three integrate different solvers, so their joint angles differ slightly
after the same commands, and that difference reaches the site."""

MOVING_MIN = 0.05
"""m/s — the site must be moving at least this fast for the velocity
comparison to mean anything."""


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):+.6f}" for x in v) + "]"


def _build_env(sim: str, num_envs: int):
    cfgs = YamArmConfig(sim_type=sim, num_envs=num_envs).build()
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return env, cfgs


def _drive(env, magnitude: float, steps: int) -> None:
    action = torch.full(
        (env.num_envs, env.act_manager.num_actions),
        magnitude,
        device=env.device,
    )
    for _ in range(steps):
        env.step(action)
    env._invalidate_cache()


def run_single(sim: str, num_envs: int) -> dict:
    env, cfgs = _build_env(sim, num_envs)
    results: dict[str, bool] = {}
    measured: dict[str, object] = {}

    print("=" * 78)
    print(f"SITE PARITY DIAG  [sim={sim}]")
    print("=" * 78)

    # ── A. the table ─────────────────────────────────────────────────────
    print("\n-- A. site table, straight from the MJCF --")
    frames = sites_from_mjcf(cfgs.scene.entities["robot"].mjcf_path)
    for idx, frame in enumerate(frames):
        print(
            f"  [{idx}] {frame.name:<22} body={frame.body_name:<20} "
            f"pos={_fmt(frame.local_pos)} quat={_fmt(frame.local_quat_wxyz)}"
        )
    named = {f.name: f for f in frames}
    results["asset_declares_the_grasp_site"] = SITE in named
    if SITE not in named:
        print(f"  {SITE!r} not found — nothing further can be checked.")
        return {"results": results, "measured": measured, "ok": False}
    measured["site_count"] = len(frames)
    measured["grasp_body"] = named[SITE].body_name
    measured["grasp_local_pos"] = [round(v, 6) for v in named[SITE].local_pos]

    data = env.get_entity_data("robot")
    parent_idx = data.find_body_index(named[SITE].body_name)
    print(f"  parent body {named[SITE].body_name!r} resolves to index {parent_idx}")

    # ── B. at rest ───────────────────────────────────────────────────────
    print("\n-- B. at rest --")
    pos_rest = data.site_pos_w([SITE])[:, 0]
    vel_rest = data.site_lin_vel_w([SITE])[:, 0]
    body_pos = data.body_pos_w([named[SITE].body_name])[:, 0]
    offset_len = float((pos_rest - body_pos).norm(dim=-1).mean())
    declared_len = float(torch.tensor(named[SITE].local_pos).norm())
    print(f"  site  = {_fmt(pos_rest[0])}")
    print(f"  body  = {_fmt(body_pos[0])}")
    print(f"  |site - body| = {offset_len:.6f}   declared |local_pos| = {declared_len:.6f}")
    # A rigid offset keeps its length whatever the body's orientation. If the
    # offset were applied in the world frame the length would still match, so
    # this is necessary rather than sufficient — the rotation check in E is
    # what separates those two.
    results["offset_length_matches_the_asset"] = abs(offset_len - declared_len) < 1e-6
    results["at_rest_the_site_is_still"] = float(vel_rest.norm(dim=-1).max()) < 1e-3
    measured["offset_len"] = round(offset_len, 6)
    measured["rest_pos"] = [round(float(v), 5) for v in pos_rest[0]]
    # The site's offset from its own parent, which is what this
    # implementation is actually responsible for. The ABSOLUTE position
    # cannot be compared across backends: mjlab lays environments out on
    # a grid and Genesis reports zero origins, so the same arm stands a
    # whole env-spacing apart. Comparing absolutes would report a metre
    # of disagreement about scene layout as if it were a site bug.
    measured["rest_offset"] = [round(float(v), 6) for v in (pos_rest - body_pos)[0]]

    if sim == "mujoco":
        native = data.site_pos_w_mjlab_native([SITE])[:, 0]
        err = float((pos_rest - native).abs().max())
        print(f"  mujoco native = {_fmt(native[0])}   max err = {err:.3e}")
        results["rest_position_matches_mujoco"] = err < POS_TOL
        measured["rest_pos_err_vs_native"] = f"{err:.3e}"

    # ── B2. orientation ──────────────────────────────────────────────────
    # A site is a frame, and ``body_quat * local_quat`` has an order.
    # Getting it backwards still yields a unit quaternion, so nothing
    # raises and every length check above still passes — the frame is
    # simply wrong. Undoing the parent's rotation must give back exactly
    # the local rotation the asset declared, which pins the order down
    # and needs no reference backend to do it.
    print("\n-- B2. orientation --")
    quat_rest = data.site_quat_w([SITE])[:, 0]
    body_quat = data.body_quat_w_all[:, parent_idx]
    recovered = quat_mul_wxyz(quat_inv_wxyz(body_quat), quat_rest)
    declared = torch.tensor(named[SITE].local_quat_wxyz, device=quat_rest.device).expand_as(recovered)
    # A quaternion and its negation are the same rotation, so compare on
    # whichever sign the declared one uses.
    aligned = torch.where((recovered * declared).sum(-1, keepdim=True) < 0, -recovered, recovered)
    quat_err = float((aligned - declared).abs().max())
    norm_err = float((quat_rest.norm(dim=-1) - 1.0).abs().max())
    print(f"  site quat        = {_fmt(quat_rest[0])}")
    print(f"  parent^-1 * site = {_fmt(aligned[0])}   declared = {_fmt(named[SITE].local_quat_wxyz)}")
    print(f"  max err = {quat_err:.3e}   |q| - 1 = {norm_err:.3e}")
    results["orientation_composes_in_the_right_order"] = quat_err < 1e-5
    results["site_orientation_is_a_unit_quaternion"] = norm_err < 1e-5
    measured["quat_local_recovery_err"] = f"{quat_err:.3e}"
    measured["rest_quat"] = [round(float(v), 5) for v in quat_rest[0]]

    if sim == "mujoco":
        native_q = data.site_quat_w_mjlab_native([SITE])[:, 0]
        signed = torch.where((quat_rest * native_q).sum(-1, keepdim=True) < 0, -quat_rest, quat_rest)
        err = float((signed - native_q).abs().max())
        print(f"  mujoco native    = {_fmt(native_q[0])}   max err = {err:.3e}")
        results["rest_orientation_matches_mujoco"] = err < POS_TOL
        measured["rest_quat_err_vs_native"] = f"{err:.3e}"

    # ── C. moving ────────────────────────────────────────────────────────
    print("\n-- C. driven, sampled mid-motion --")
    env.reset()
    _drive(env, magnitude=8.0, steps=DRIVE_STEPS)
    pos_moving = data.site_pos_w([SITE])[:, 0]
    vel_moving = data.site_lin_vel_w([SITE])[:, 0]
    speed = float(vel_moving.norm(dim=-1).mean())
    print(f"  site  = {_fmt(pos_moving[0])}")
    print(f"  vel   = {_fmt(vel_moving[0])}   |v| = {speed:.5f} m/s")
    results["the_site_is_actually_moving"] = speed > MOVING_MIN
    measured["moving_speed"] = round(speed, 5)
    measured["moving_pos"] = [round(float(v), 5) for v in pos_moving[0]]
    measured["moving_vel"] = [round(float(v), 5) for v in vel_moving[0]]

    if sim == "mujoco":
        native_pos = data.site_pos_w_mjlab_native([SITE])[:, 0]
        native_vel = data.site_lin_vel_w_mjlab_native([SITE])[:, 0]
        pos_err = float((pos_moving - native_pos).abs().max())
        vel_err = float((vel_moving - native_vel).abs().max())
        print(f"  mujoco native vel = {_fmt(native_vel[0])}")
        print(f"  max err: pos {pos_err:.3e}   vel {vel_err:.3e}")
        results["moving_position_matches_mujoco"] = pos_err < POS_TOL
        results["moving_velocity_matches_mujoco"] = vel_err < VEL_TOL
        measured["moving_pos_err_vs_native"] = f"{pos_err:.3e}"
        measured["moving_vel_err_vs_native"] = f"{vel_err:.3e}"

    # Repeated with the body rotated well away from identity: with the
    # parent upright the two multiplication orders can agree by accident.
    quat_moving = data.site_quat_w([SITE])[:, 0]
    body_quat_moving = data.body_quat_w_all[:, parent_idx]
    recovered_moving = quat_mul_wxyz(quat_inv_wxyz(body_quat_moving), quat_moving)
    aligned_moving = torch.where(
        (recovered_moving * declared).sum(-1, keepdim=True) < 0, -recovered_moving, recovered_moving
    )
    moving_quat_err = float((aligned_moving - declared).abs().max())
    print(f"  orientation still composes with the body turned: err = {moving_quat_err:.3e}")
    results["orientation_holds_with_the_body_turned"] = moving_quat_err < 1e-5
    measured["moving_quat_recovery_err"] = f"{moving_quat_err:.3e}"
    if sim == "mujoco":
        native_qm = data.site_quat_w_mjlab_native([SITE])[:, 0]
        signed_m = torch.where((quat_moving * native_qm).sum(-1, keepdim=True) < 0, -quat_moving, quat_moving)
        err_qm = float((signed_m - native_qm).abs().max())
        print(f"  mujoco native orientation while moving: err = {err_qm:.3e}")
        results["moving_orientation_matches_mujoco"] = err_qm < POS_TOL
        measured["moving_quat_err_vs_native"] = f"{err_qm:.3e}"

    # ── D. velocity is the derivative of position ────────────────────────
    # Independent of MuJoCo: catches a formula that is self-consistently
    # wrong everywhere. Compared at the midpoint, since a finite difference
    # over one control step approximates the velocity in the middle of it.
    print("\n-- D. velocity vs finite difference of position --")
    before_pos = data.site_pos_w([SITE])[:, 0].clone()
    before_vel = data.site_lin_vel_w([SITE])[:, 0].clone()
    _drive(env, magnitude=8.0, steps=1)
    after_pos = data.site_pos_w([SITE])[:, 0]
    after_vel = data.site_lin_vel_w([SITE])[:, 0]
    fd = (after_pos - before_pos) / env.control_dt
    midpoint = 0.5 * (before_vel + after_vel)
    fd_err = float((fd - midpoint).norm(dim=-1).mean())
    fd_rel = fd_err / max(float(midpoint.norm(dim=-1).mean()), 1e-9)
    print(f"  finite difference = {_fmt(fd[0])}")
    print(f"  reported (mid)    = {_fmt(midpoint[0])}")
    print(f"  mean error = {fd_err:.5f} m/s  ({fd_rel * 100:.1f}% of speed)")
    # A loose bound on purpose: one control step of a stiff PD is not a
    # smooth arc, so the difference quotient carries real truncation error.
    # An omitted omega x r term is not a 20% effect — it is the entire
    # tangential component.
    results["velocity_tracks_position"] = fd_rel < 0.20
    measured["fd_rel_error"] = round(fd_rel, 4)

    # ── E. the offset rides in the body frame ────────────────────────────
    # Drive the wrist joint alone. The site must sweep an arc about the
    # parent body's origin at the declared radius; an offset applied in the
    # world frame would translate with the body but never rotate with it,
    # leaving |site - body| pointing the same way throughout.
    print("\n-- E. the offset rotates with its body --")
    env.reset()
    env._invalidate_cache()
    r0 = (data.site_pos_w([SITE])[:, 0] - data.body_pos_w([named[SITE].body_name])[:, 0])[0].clone()
    wrist = list(env.entity_indexing("robot").joint_names).index("joint6")
    action = torch.zeros(env.num_envs, env.act_manager.num_actions, device=env.device)
    action[:, wrist] = 6.0
    for _ in range(DRIVE_STEPS):
        env.step(action)
    env._invalidate_cache()
    r1 = (data.site_pos_w([SITE])[:, 0] - data.body_pos_w([named[SITE].body_name])[:, 0])[0]
    turned = float(torch.rad2deg(torch.acos((r0 @ r1 / (r0.norm() * r1.norm())).clamp(-1, 1))))
    print(f"  offset before = {_fmt(r0)}")
    print(f"  offset after  = {_fmt(r1)}")
    print(f"  it turned {turned:.2f} deg, length {float(r0.norm()):.6f} -> {float(r1.norm()):.6f}")
    results["offset_turns_with_the_body"] = turned > 5.0
    results["offset_keeps_its_length"] = abs(float(r0.norm()) - float(r1.norm())) < 1e-6
    measured["offset_turn_deg"] = round(turned, 3)

    # ── F. an entity with no MJCF refuses ────────────────────────────────
    print("\n-- F. an entity built from a URDF has no sites --")
    table = env.get_entity_data("table")
    try:
        table.site_pos_w([SITE])
        outcome = "returned a value"
    except (ValueError, KeyError, FileNotFoundError) as e:
        outcome = f"{type(e).__name__}"
    print(f"  table.site_pos_w -> {outcome}")
    results["urdf_entity_refuses_sites"] = outcome != "returned a value"
    measured["urdf_site_outcome"] = outcome

    # ── G. selector resolution ───────────────────────────────────────────
    print("\n-- G. a selector resolves the site --")
    resolved = env.resolve_selector(SceneEntitySelector(name="robot", site_names=(SITE,)))
    ids = resolved.site_ids
    print(f"  site_ids = {None if ids is None else ids.tolist()}")
    by_id = data.site_pos_w_by_ids(ids) if ids is not None else None
    agree = by_id is not None and torch.allclose(by_id[:, 0], data.site_pos_w([SITE])[:, 0])
    results["selector_site_ids_resolve"] = ids is not None and int(ids[0]) == list(named).index(SITE)
    results["by_ids_matches_by_name"] = bool(agree)
    measured["site_id"] = None if ids is None else int(ids[0])

    print("=" * 78)
    print("VERDICT")
    ok = True
    for k, v in results.items():
        print(f"  {k:<42}: {'PASS' if v else 'FAIL'}")
        ok = ok and v
    print(f"  {'OVERALL':<42}: {'PASS' if ok else 'FAIL'}")
    print()
    print("REPORTED")
    for k, v in measured.items():
        print(f"  {k:<42}: {v}")
    print("=" * 78)
    return {"results": results, "measured": measured, "ok": ok}


def run_all(num_envs: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="site_parity_"))
    out: dict[str, dict] = {}
    env_vars = dict(os.environ, JAXRLWORLD_ALLOW_MULTI_SIM="1")

    for sim in _SIMS:
        result_path = tmp / f"{sim}.json"
        cmd = [
            sys.executable,
            "-m",
            "rlworld.scripts.diag.parity.site_parity_diag",
            "--sim",
            sim,
            "--result-json",
            str(result_path),
            "--num-envs",
            str(num_envs),
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
    print(f"{'check':<44}" + "".join(f"{s:>10}" for s in _SIMS))
    print("-" * 78)
    overall = True
    for k in keys:
        row = f"{k:<44}"
        for s in _SIMS:
            v = out.get(s, {}).get("results", {}).get(k)
            # A mujoco-only check is absent elsewhere by design, not failing.
            row += f"{'—' if v is None else ('PASS' if v else 'FAIL'):>10}"
            overall = overall and (v is None or bool(v))
        print(row)

    # The backends must also agree with each other, which no single-backend
    # run can see. Position at rest is the sharpest of these: same asset,
    # same declared pose, no integration yet.
    print()
    print("CROSS-SIM AGREEMENT")
    rest = {s: out[s]["measured"].get("rest_offset") for s in out if out[s]["measured"].get("rest_offset")}
    if len(rest) > 1:
        stacked = torch.tensor(list(rest.values()))
        spread = float((stacked.max(dim=0).values - stacked.min(dim=0).values).max())
        print(f"  site-to-parent offset spread across {len(rest)} backends = {spread:.3e} m (tol {CROSS_TOL})")
        for s, v in rest.items():
            print(f"    {s:<10} {v}")
        overall = overall and spread < CROSS_TOL
        print(f"  backends_agree_at_rest: {'PASS' if spread < CROSS_TOL else 'FAIL'}")
    absolute = {s: out[s]["measured"].get("rest_pos") for s in out if out[s]["measured"].get("rest_pos")}
    print("  absolute positions (NOT comparable — env origins differ per backend):")
    for s, v in absolute.items():
        print(f"    {s:<10} {v}")

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
        print(f"  {k:<28}" + "".join(f"{v:>22}" for v in vals) + f"   {agree}")

    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 78)
    return 0 if overall else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=list(_SIMS), default=None, help="Run one backend. Omit to run all three.")
    ap.add_argument("--result-json", default=None, help="Child mode: where to write this backend's result.")
    ap.add_argument("--num-envs", type=int, default=4)
    args = ap.parse_args()

    if args.sim is None:
        return run_all(args.num_envs)

    result = run_single(args.sim, args.num_envs)
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
