"""Step A — is a fixed-base arm actually usable, on all three backends?

Every robot in this repo before the YAM arm has a floating base, so the
welded path is unexercised. This diag answers the questions that make
the difference between "the preset builds" and "the arm is a sound
foundation for a manipulation task", with numbers rather than a clean
import:

* **It loads and is welded.** ``is_fixed_base`` is true and the base sits
  where the config declared, per environment.
* **The base does not move.** Drive the joints hard and the mount stays
  put; a base that drifts means the weld did not take.
* **Reset is clean.** After a reset the joints are at the home pose,
  inside their limits. Newton used to write a root pose into the first
  seven *joint* coordinates of a welded articulation, which produced
  exactly this failure and nothing else — so it is checked here at the
  preset level too, not only in the writer's own diag.
* **The joints track a target.** A commanded offset actually moves the
  arm; on Newton a missing ``<actuator>`` block silently yields ``nu=0``
  and an arm that never responds.
* **The gripper's fingers mirror.** The two fingers are coupled by a
  MuJoCo ``<equality>`` constraint. Genesis and Newton both claim to
  parse it — this measures whether the right finger actually follows the
  left, because a gripper that closes one-sided cannot grasp anything.
* **It holds itself up.** Commanded to its home pose it stays there under
  gravity. The XML's ``gravcomp`` was stripped so all three backends see
  real gravity; if the gains cannot hold the arm, that shows up here.
* **It is not standing in the floor.** The base clears the ground plane,
  so no permanent contact force pollutes later contact-derived signals.

Run all three and cross-compare::

    python -m rlworld.scripts.diag.yam_fixed_base_smoke --num-envs 4

or one backend::

    python -m rlworld.scripts.diag.yam_fixed_base_smoke --sim mujoco --num-envs 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from rlworld.rl.configs.presets.yam_arm.base import YamArmConfig
from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from rlworld.rl.runners import BaseRunner

_SIMS = ("genesis", "newton", "mujoco")

GRIPPER_JOINT = "left_finger"
# The links that actually carry the contact pads. NOT the finger mounts
# (``link_left_finger`` / ``link_right_finger``): the finger is a linkage,
# so mount separation moves OPPOSITE to pad separation, and measuring the
# mounts reports the gripper closing while the jaws are opening.
LEFT_FINGER_BODY = "lf_down"
RIGHT_FINGER_BODY = "rf_down"
WRIST_BODY = "link_6"


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):+.5f}" for x in v) + "]"


def _finite(t) -> bool:
    return bool(torch.isfinite(t).all())


# ══════════════════════════════════════════════════════════════════════════
# Scene construction
# ══════════════════════════════════════════════════════════════════════════


def _build_env(sim: str, num_envs: int):
    cfgs = YamArmConfig(sim_type=sim, num_envs=num_envs).build()

    # A ground-contact group, added here rather than in the preset: the
    # preset has no use for it, but "the base is not sunk into the floor"
    # is only provable by measuring the force.
    # Each backend names the ground differently: Newton reaches it as a
    # shape with no parent body, Genesis resolves contacts by link so it
    # takes the terrain entity whole, and mjlab exposes it as a body.
    ground_secondary = {
        "newton": ContactMatch(mode="geom", pattern="ground_plane", entity="terrain"),
        "genesis": ContactMatch(mode="entity", entity="terrain"),
        "mujoco": ContactMatch(mode="body", pattern="terrain"),
    }[sim]
    ground = ContactSensorCfg(
        name="arm_ground_contact",
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        secondary=ground_secondary,
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    # Contact sensors live under a different field name per backend, and
    # Genesis requires the history to match the decimation.
    field = "sensors" if sim == "mujoco" else "contact_sensors"
    existing = tuple(getattr(cfgs.scene, field) or ())
    ground = replace(ground, history_length=cfgs.env.decimation)
    setattr(cfgs.scene, field, list(existing) + [ground])

    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return env, cfgs


def _step_n(env, action, n: int) -> bool:
    """Step ``n`` times, reporting whether any env reset mid-way."""
    reset_seen = False
    for _ in range(n):
        _, _, dones, _, _ = env.step(action)
        if bool(dones.any()):
            reset_seen = True
    env._invalidate_cache()
    return reset_seen


# ══════════════════════════════════════════════════════════════════════════
# The diag
# ══════════════════════════════════════════════════════════════════════════


def _newton_control_evidence(env, act_names: list[str]) -> None:
    """Dump the joint targets as written, and as MuJoCo Warp sees them."""
    import newton
    import warp as wp

    sm = env.scene_manager
    model = sm.model
    n_worlds = model.world_count

    # The counts the writer derives its per-world stride from, against the
    # length of the array it actually writes into. A mismatch means every
    # world past the first lands at the wrong offset.
    n_target = len(sm.control.joint_target_q)
    print(f"[newton] use_coord_layout_targets = {newton.use_coord_layout_targets}")
    print(
        f"[newton] joint_count={model.joint_count} joint_coord_count={model.joint_coord_count} "
        f"joint_dof_count={model.joint_dof_count}  world_count={n_worlds}"
    )
    print(
        f"[newton] len(control.joint_target_q) = {n_target} -> {n_target / n_worlds} per world; "
        f"writer used {model.joint_coord_count // n_worlds} (coord) / {model.joint_dof_count // n_worlds} (dof)"
    )
    leaf = lambda lbl: lbl.rsplit("/", 1)[-1]  # noqa: E731
    labels = list(model.joint_label)
    per_world_joints = max(1, len(labels) // n_worlds)
    print(f"[newton] world-0 joints ({per_world_joints}) = {[leaf(x) for x in labels[:per_world_joints]]}")
    # Newton says to index joint_target_q through this, "regardless of
    # layout" (Model.joint_target_q_start). If its per-world step is not the
    # dof count, a writer that assumes a uniform stride puts every world but
    # the first at the wrong offset.
    tq_start = model.joint_target_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    n_show = min(len(tq_start), 2 * per_world_joints + 2)
    print(f"[newton] joint_target_q_start[:{n_show}] = {tq_start[:n_show].tolist()}")
    print(f"[newton] joint_qd_start[:{n_show}]       = {qd_start[:n_show].tolist()}")
    print(f"[newton] q_indices  = {env.act_manager.actuated_q_indices.tolist()}")
    print(f"[newton] qd_indices = {env.act_manager.actuated_qd_indices.tolist()}")

    target = wp.to_torch(sm.control.joint_target_q).reshape(n_worlds, -1)
    print(f"[newton] control.joint_target_q shape = {tuple(target.shape)}")
    for w in range(n_worlds):
        print(f"[newton]   world {w} target = {_fmt(target[w])}")

    mjw = sm.solver.mjw_data
    ctrl = wp.to_torch(mjw.ctrl)
    print(f"[newton] mjwarp data.ctrl shape = {tuple(ctrl.shape)}  (worlds x nu)")
    for w in range(min(n_worlds, ctrl.shape[0])):
        print(f"[newton]   world {w} ctrl   = {_fmt(ctrl[w])}")
    qpos = wp.to_torch(mjw.qpos)
    print(f"[newton] mjwarp data.qpos shape = {tuple(qpos.shape)}")
    for w in range(min(n_worlds, qpos.shape[0])):
        print(f"[newton]   world {w} qpos   = {_fmt(qpos[w])}")


def run_single(sim: str, num_envs: int, settle_steps: int) -> dict:
    env, cfgs = _build_env(sim, num_envs)
    results: dict[str, bool] = {}
    measured: dict[str, object] = {}

    data = env.get_entity_data("robot")
    n_act = env.act_manager.num_actions
    zeros = torch.zeros(env.num_envs, n_act, device=env.device)

    print("=" * 78)
    print(f"YAM FIXED-BASE ARM DIAG  [sim={sim}]")
    print("=" * 78)

    # ── A1. loads, and the actuated set is what the preset declared ──────
    jp = data.joint_pos
    jv = data.joint_vel
    act_names = list(env.act_manager.actuated_joint_names)
    print(f"[load] joint_pos.shape = {tuple(jp.shape)}   joint_vel.shape = {tuple(jv.shape)}")
    print(f"[load] num_actions = {n_act}   actuated = {act_names}")
    results["loads_with_expected_action_dim"] = n_act == 7 and tuple(jp.shape) == (env.num_envs, 7)
    # right_finger must NOT be actuated: it is driven by the equality.
    results["mirrored_finger_not_actuated"] = all("right_finger" not in n for n in act_names)
    results["joint_reads_finite"] = _finite(jp) and _finite(jv)
    measured["actuated_dof_names"] = act_names

    # ── A2. welded, and where the config said ────────────────────────────
    is_fixed = data.is_fixed_base
    pos = data.root_link_pos_w
    quat = data.root_link_quat_w
    origins = env.scene_manager.env_origins
    declared = torch.tensor(cfgs.scene.entities["robot"].init_state.pos, device=env.device).unsqueeze(0)
    expected = origins + declared
    place_err = float((pos - expected).abs().max())
    print(f"[base] is_fixed_base = {is_fixed} (expect True)")
    print(f"[base] root_link_pos_w env0 = {_fmt(pos[0])}   expected {_fmt(expected[0])}")
    print(f"[base] env_origins[0] = {_fmt(origins[0])}   max err over {env.num_envs} envs = {place_err:.3e}")
    print(f"[base] root_link_quat_w = {_fmt(quat[0])}  |q| = {float(torch.linalg.norm(quat[0])):.6f}")
    results["base_is_fixed"] = bool(is_fixed)
    results["base_at_declared_pose"] = place_err < 1e-4
    measured["base_pos"] = [round(float(v), 5) for v in pos[0]]

    # ── A3. the base stays put while the arm works ───────────────────────
    base_before = pos.clone()
    torch.manual_seed(0)
    shove = torch.empty(env.num_envs, n_act, device=env.device).uniform_(-1.0, 1.0)
    reset_during = _step_n(env, shove, settle_steps)
    base_drift = float((data.root_link_pos_w - base_before).abs().max())
    print(f"[base] after {settle_steps} steps of random actions: max|Δbase| = {base_drift:.3e} (expect ~0)")
    print(f"[base] (an env reset during those steps: {reset_during})")
    results["base_immovable_under_load"] = base_drift < 1e-5

    # ── A4. reset returns the arm to its home pose, inside the limits ────
    env.reset()
    jp_reset = data.joint_pos
    offset = env.act_manager.offset
    home_err = float((jp_reset - offset).abs().max())
    noise_hi = max(abs(v) for v in YamArmConfig().reset_joint_position_noise)
    # The limits the simulator itself enforces, in canonical actuated
    # order. Not ``_soft_joint_limits``: mjlab already stores the soft
    # pair, so applying the factor again shrinks the band a second time
    # and reports a joint out of bounds that the sim considers legal.
    lower, upper = env.act_manager._get_joint_limits()
    outside = (jp_reset < lower - 1e-4) | (jp_reset > upper + 1e-4)
    within = not bool(outside.any())
    print(f"[reset] joint_pos env0 = {_fmt(jp_reset[0])}")
    print(f"[reset] home      env0 = {_fmt(offset[0] if offset.ndim > 1 else offset)}")
    print(f"[reset] max |joint_pos - home| = {home_err:.4f}   (reset noise is ±{noise_hi})")
    print(f"[reset] every joint inside its soft limits: {within}")
    if not within:
        for j in outside.any(dim=0).nonzero().flatten().tolist():
            env_i = int(outside[:, j].nonzero()[0])
            print(
                f"[reset]   OUT: {act_names[j]:<12} q = {float(jp_reset[env_i, j]):+.5f}  "
                f"soft = [{float(lower[j]):+.5f}, {float(upper[j]):+.5f}]"
            )
    # The Newton root-write bug wrote coordinates into the joint slots,
    # which lands far outside both the noise band and the limits.
    results["reset_returns_home"] = home_err < noise_hi + 1e-3
    results["reset_stays_within_limits"] = within
    measured["reset_home_err"] = round(home_err, 5)

    # ── A5. the joints follow a commanded target ─────────────────────────
    # A unit action is one action-scale step away from home, per joint.
    target_action = torch.full((env.num_envs, n_act), 0.5, device=env.device)
    before_track = data.joint_pos.clone()
    _step_n(env, target_action, settle_steps * 4)
    after_track = data.joint_pos
    delta = (after_track - before_track).abs()
    moved = delta.max(dim=0).values
    print(f"[drive] per-joint max |Δq| under a constant +0.5 action: {[round(float(v), 4) for v in moved]}")
    print(f"[drive] per-env max |Δq| = {[round(float(v), 4) for v in delta.max(dim=-1).values]}")
    measured["drive_delta_per_env"] = [round(float(v), 4) for v in delta.max(dim=-1).values]
    results["joints_respond_to_action"] = bool((moved > 1e-3).all())
    measured["drive_delta"] = [round(float(v), 4) for v in moved]

    # ── A6. the gripper's fingers mirror each other ──────────────────────
    # Measured on the finger BODIES, not the joint values: the mirrored
    # joint is not actuated, so its coordinate is not in ``joint_pos``,
    # and what matters physically is that both jaws close.
    grip_idx = act_names.index(GRIPPER_JOINT)

    def _finger_offsets() -> tuple[torch.Tensor, torch.Tensor]:
        wrist = data.body_pos_w((WRIST_BODY,))[:, 0, :]
        left = data.body_pos_w((LEFT_FINGER_BODY,))[:, 0, :]
        right = data.body_pos_w((RIGHT_FINGER_BODY,))[:, 0, :]
        return left - wrist, right - wrist

    # Sign convention, measured at the pads: a POSITIVE command spreads
    # them. (The finger slide's own coordinate runs the other way, which is
    # why the linkage has to be measured rather than reasoned about.)
    open_action = zeros.clone()
    open_action[:, grip_idx] = 1.0
    _step_n(env, open_action, settle_steps * 3)
    left_open, right_open = _finger_offsets()

    close_action = zeros.clone()
    close_action[:, grip_idx] = -1.0
    _step_n(env, close_action, settle_steps * 3)
    left_closed, right_closed = _finger_offsets()

    d_left = float((left_closed - left_open).norm(dim=-1).mean())
    d_right = float((right_closed - right_open).norm(dim=-1).mean())
    span_open = float((left_open - right_open).norm(dim=-1).mean())
    span_closed = float((left_closed - right_closed).norm(dim=-1).mean())
    print(f"[grip] finger travel from open to closed:  left = {d_left:.5f} m   right = {d_right:.5f} m")
    print(f"[grip] jaw span: open = {span_open:.5f} m -> closed = {span_closed:.5f} m")
    # Both jaws must move, by comparable amounts, and the span must shrink.
    results["gripper_left_moves"] = d_left > 1e-3
    results["gripper_right_mirrors_left"] = d_right > 1e-3 and abs(d_left - d_right) < 0.5 * max(d_left, 1e-9)
    results["gripper_span_closes"] = span_closed < span_open - 1e-3
    measured["finger_travel_left"] = round(d_left, 5)
    measured["finger_travel_right"] = round(d_right, 5)
    measured["jaw_span_open"] = round(span_open, 5)
    measured["jaw_span_closed"] = round(span_closed, 5)

    # ── A7. it holds its home pose under gravity ─────────────────────────
    env.reset()
    _step_n(env, zeros, settle_steps * 6)
    held = data.joint_pos
    sag = float((held - offset).abs().max())
    vel = float(data.joint_vel.abs().max())
    print(f"[hold] commanded home for {settle_steps * 6} steps: max |q - home| = {sag:.4f} rad, max |qd| = {vel:.4f}")
    # Per joint, because a single max hides WHICH joint is not tracking —
    # and a position actuator that never reaches a large home angle looks
    # identical, in the max, to one that sags uniformly.
    home_row = offset[0] if offset.ndim > 1 else offset
    print(f"[hold] {'joint':<12}{'target':>10}{'reached':>10}{'error':>10}")
    for j, name in enumerate(act_names):
        tgt = float(home_row[j])
        got = float(held[0, j])
        print(f"[hold] {name:<12}{tgt:>10.4f}{got:>10.4f}{got - tgt:>10.4f}")
    measured["hold_per_joint_error"] = [round(float(held[0, j] - home_row[j]), 4) for j in range(len(act_names))]
    # Per env as well: a static equilibrium is engine-independent, so every
    # env should land on the same numbers. One env drifting while env 0
    # sits still is a replication problem, not a gain problem, and the
    # max alone cannot tell those apart.
    per_env = (held - offset).abs().max(dim=-1).values
    print(f"[hold] max |q - home| per env = {[round(float(v), 4) for v in per_env]}")
    measured["hold_sag_per_env"] = [round(float(v), 4) for v in per_env]
    results["holds_home_in_every_env"] = float(per_env.max() - per_env.min()) < 1e-3
    # The XML's gravcomp was removed, so this is the real gravity load.
    results["holds_home_under_gravity"] = sag < 0.25 and vel < 1.0
    measured["gravity_sag"] = round(sag, 5)

    # ── Newton control-plumbing evidence ─────────────────────────────────
    # Only env 0 tracks its target once several envs exist, and env 0 is the
    # one env whose absolute indices equal its per-world indices — the
    # signature of a per-world stride being wrong somewhere between the
    # action manager and the solver. Print the two buffers on that path so
    # the break can be located rather than guessed at.
    if sim == "newton":
        _newton_control_evidence(env, act_names)

    # ── A8. the base is not standing in the floor ────────────────────────
    force = env.contact_manager.contact_force("arm_ground_contact")
    fmag = float(force.norm(dim=-1).max())
    touching = bool(env.contact_manager.is_contact("arm_ground_contact").any())
    print(f"[floor] arm-vs-ground: any contact = {touching}   max |force| = {fmag:.4f} N (expect 0)")
    results["arm_clears_the_ground"] = fmag < 1e-3 and not touching
    measured["ground_contact_force"] = round(fmag, 5)

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


def run_all(num_envs: int, settle_steps: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="yam_fixed_base_"))
    out: dict[str, dict] = {}
    env_vars = dict(os.environ, JAXRLWORLD_ALLOW_MULTI_SIM="1")

    for sim in _SIMS:
        result_path = tmp / f"{sim}.json"
        cmd = [
            sys.executable,
            "-m",
            "rlworld.scripts.diag.yam_fixed_base_smoke",
            "--sim",
            sim,
            "--result-json",
            str(result_path),
            "--num-envs",
            str(num_envs),
            "--settle-steps",
            str(settle_steps),
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
        print(f"  {k:<30}" + "".join(f"{v:>26}" for v in vals) + f"   {agree}")

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
