"""Two robots, three action terms — does each command reach its own joints?

Every preset before this one has a single articulation driven by a
single term, which makes several distinct mappings coincide: the term's
action slice equals its joint ids, the manager's joint list equals the
robot's, and "the entity" is unambiguous. None of those hold with two
robots, and each one that silently stops holding produces a plausible
number rather than an error, so this diag measures them:

* **The action space is the sum of the terms.** 6 (left arm) + 1 (left
  gripper) + 7 (right arm) = 14.
* **Each entity has its own joint indexing**, of its own width. Sharing
  one means the second arm's reads and writes land on the first, which
  shows up as a doubled joint list.
* **A term moves its own robot and only its own.** Command one arm and
  check the other stayed where it was resting. The two arms are mounted
  0.6 m apart, wider than either can reach, so the untouched arm has no
  physical route to move.
* **Two terms on one robot coexist.** The arm term and the gripper term
  drive the same articulation. A term that writes its command into a
  fresh full-width buffer commands every joint it does NOT own to zero,
  so whichever term runs last wins and the other's joints collapse.
* **The terms compose.** Commanding all three at once must land every
  joint where commanding that term alone landed it. This is the check that
  needs no tolerance on tracking accuracy, and the one that catches both a
  crossed action slice and an erased target.
* **Reset restores both.** The joint reset acts on a named entity, so
  the second arm needs its own term — an env that resets only the first
  arm looks fine until the second one drifts across an episode.

Commanded targets are computed by inverting each term's own
``scale``/``offset``, and every comparison is against a MEASURED resting
pose rather than the declared home. This arm sags ~0.16 rad under gravity
and tracks with a standing PD error of the same order, so a check written
against the declared home reports drift on a perfectly isolated arm.

Run all three and cross-compare::

    python -m rlworld.scripts.diag.yam_dual_smoke --num-envs 4

or one backend::

    python -m rlworld.scripts.diag.yam_dual_smoke --sim mujoco --num-envs 4
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

from rlworld.rl.configs.presets.yam_dual.base import RIGHT_ROBOT, YamDualArmConfig
from rlworld.rl.runners import BaseRunner

_SIMS = ("genesis", "newton", "mujoco")

LEFT_ROBOT = "robot"

SETTLE_STEPS = 120
"""Steps to hold a command before reading the result. The gains give a
2 Hz closed loop and a control step is 20 ms, so this is a few seconds
of settling — long enough that a joint short of its target is short for
a reason other than not having arrived yet."""

MOVED_MIN = 0.15
"""rad — how far a commanded arm must travel from its resting pose for the
command to count as having arrived somewhere. The requests below ask for
0.25 rad on several joints."""

HOLD_TOL = 0.02
"""rad — how far an UNcommanded joint may move from its resting pose. The
reference is the measured resting pose, not the declared home: this arm
sags ~0.16 rad at joint4 under gravity even when commanded home."""

COMPOSE_TOL = 0.02
"""rad — how far a joint may land from where the same command put it when
issued alone. This is the tolerance that actually matters; a crossed action
slice or an erased target misses by an order of magnitude more."""


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):+.4f}" for x in v) + "]"


def _finite(t) -> bool:
    return bool(torch.isfinite(t).all())


def _build_env(sim: str, num_envs: int):
    preset = YamDualArmConfig(sim_type=sim, num_envs=num_envs)
    env = BaseRunner._create_env_from_config(preset.build())
    env.reset()
    return env, preset


def _step_n(env, action, n: int) -> bool:
    reset_seen = False
    for _ in range(n):
        _, _, dones, _, _ = env.step(action)
        if bool(dones.any()):
            reset_seen = True
    env._invalidate_cache()
    return reset_seen


def _raw_for_target(term, target: torch.Tensor) -> torch.Tensor:
    """The raw action that asks ``term`` for ``target``.

    Inverts the term's own ``processed = raw * scale + offset``, so the
    diag commands a position rather than an arbitrary number and can
    then check that position was reached.
    """
    return (target - term._offset) / term._scale


def run_single(sim: str, num_envs: int) -> dict:
    env, preset = _build_env(sim, num_envs)
    results: dict[str, bool] = {}
    measured: dict[str, object] = {}

    am = env.act_manager
    terms = am.terms
    slices = am.term_action_slices
    n_act = am.num_actions
    zeros = torch.zeros(env.num_envs, n_act, device=env.device)

    left_idx = env.entity_indexing(LEFT_ROBOT)
    right_idx = env.entity_indexing(RIGHT_ROBOT)

    print("=" * 78)
    print(f"YAM DUAL-ARM DIAG  [sim={sim}]")
    print("=" * 78)

    # ── A. structure ─────────────────────────────────────────────────────
    print("\n-- A. structure --")
    for name, term in terms.items():
        sl = slices[name]
        print(
            f"  term {name:<14} entity={term.entity_name:<12} dim={term.action_dim:<3} "
            f"actions[{sl.start}:{sl.stop}]  joints={term.joint_names}"
        )
    print(f"  {LEFT_ROBOT:<12} joints      = {list(left_idx.joint_names)}")
    print(f"  {LEFT_ROBOT:<12} sim_indices = {left_idx.sim_indices.tolist()}")
    print(f"  {RIGHT_ROBOT:<12} joints      = {list(right_idx.joint_names)}")
    print(f"  {RIGHT_ROBOT:<12} sim_indices = {right_idx.sim_indices.tolist()}")

    expected_dim = sum(t.action_dim for t in terms.values())
    results["action_dim_is_the_sum_of_terms"] = n_act == expected_dim == 14
    measured["action_dim"] = n_act

    results["term_slices_tile_the_action_space"] = [(slices[n].start, slices[n].stop) for n in terms] == [
        (0, 6),
        (6, 7),
        (7, 14),
    ]

    results["both_arms_have_the_same_joint_names"] = list(left_idx.joint_names) == list(right_idx.joint_names)

    # NOT a disjointness check: ``sim_indices`` is entity-LOCAL on all three
    # backends (mjlab indexes an entity's own joint list, Genesis uses
    # ``dofs_idx_local``, Newton the articulation view's own order), so two
    # copies of the same arm SHOULD carry the same numbers. What has to hold
    # is that each entity owns a separate indexing object of its own width —
    # the failure this replaces is one indexing shared by both arms, which
    # shows up as a doubled joint list. Whether the indices actually address
    # different joints is settled by the motion checks below, not here.
    results["each_arm_has_its_own_indexing"] = (
        left_idx is not right_idx
        and len(left_idx.joint_names) == len(right_idx.joint_names) == 7
        and left_idx.sim_indices.numel() == 7
    )
    measured["left_joint_count"] = len(left_idx.joint_names)
    measured["right_joint_count"] = len(right_idx.joint_names)
    measured["left_sim_indices"] = left_idx.sim_indices.tolist()
    measured["right_sim_indices"] = right_idx.sim_indices.tolist()

    left_data = env.get_entity_data(LEFT_ROBOT)
    right_data = env.get_entity_data(RIGHT_ROBOT)
    left_y = float(left_data.root_link_pos_w[:, 1].mean())
    right_y = float(right_data.root_link_pos_w[:, 1].mean())
    print(f"  base y: {LEFT_ROBOT}={left_y:+.4f}  {RIGHT_ROBOT}={right_y:+.4f}")
    results["arms_are_placed_where_declared"] = (
        abs(left_y - preset.base_pos[1]) < 1e-3 and abs(right_y - preset.right_base_pos[1]) < 1e-3
    )
    measured["left_base_y"] = round(left_y, 4)
    measured["right_base_y"] = round(right_y, 4)

    home_left = env._resolve_default_joint_pos(LEFT_ROBOT)
    home_right = env._resolve_default_joint_pos(RIGHT_ROBOT)
    print(f"  home {LEFT_ROBOT:<12} = {_fmt(home_left)}")
    print(f"  home {RIGHT_ROBOT:<12} = {_fmt(home_right)}")

    # Targets: a quarter-radian off home on every arm joint, clamped into
    # the soft limits so the request is one the joint can actually honour.
    def _arm_target(entity: str, joints: list[str], delta: float) -> torch.Tensor:
        names = list(env.entity_indexing(entity).joint_names)
        home = env._resolve_default_joint_pos(entity)
        mid, half = am.soft_joint_limits_of(entity)
        ids = [names.index(j) for j in joints]
        want = home[ids] + delta
        return want.clamp(mid[ids] - half[ids], mid[ids] + half[ids])

    def _home(entity: str, joints: list[str]) -> torch.Tensor:
        names = list(env.entity_indexing(entity).joint_names)
        home = env._resolve_default_joint_pos(entity)
        return home[[names.index(j) for j in joints]]

    def _joint_pos(entity: str, joints: list[str]) -> torch.Tensor:
        names = list(env.entity_indexing(entity).joint_names)
        data = env.get_entity_data(entity)
        ids = [names.index(j) for j in joints]
        return data.joint_pos[:, ids]

    left_arm_joints = terms["left_arm"].joint_names
    left_grip_joints = terms["left_gripper"].joint_names
    right_joints = terms["right_arm"].joint_names

    # ── B. the resting baseline ──────────────────────────────────────────
    # Every later check is relative to this, NOT to the declared home pose.
    # A zero action commands home, but the arm settles a measurable distance
    # short of it — this robot sags ~0.16 rad at joint4 under gravity, which
    # is 8x any believable action leak. Comparing against the home pose would
    # therefore report "the arm nobody commanded moved" on a perfectly
    # isolated arm, and comparing tracking error against an absolute
    # tolerance would report "the commanded arm never arrived" on one that
    # tracked exactly as well as the single-arm preset does.
    print("\n-- B. resting baseline (zero action = command both arms home) --")
    env.reset()
    _step_n(env, zeros, SETTLE_STEPS)
    base_left = _joint_pos(LEFT_ROBOT, list(left_idx.joint_names))
    base_right = _joint_pos(RIGHT_ROBOT, list(right_idx.joint_names))
    sag_left = float((base_left - home_left).abs().max())
    sag_right = float((base_right - home_right).abs().max())
    print(f"  left  settles at {_fmt(base_left[0])}   sag from home = {sag_left:.5f}")
    print(f"  right settles at {_fmt(base_right[0])}   sag from home = {sag_right:.5f}")
    results["both_arms_rest_identically"] = float((base_left - base_right).abs().max()) < HOLD_TOL
    measured["baseline_sag_left"] = round(sag_left, 5)
    measured["baseline_sag_right"] = round(sag_right, 5)

    def _delta_from_baseline(entity: str, baseline: torch.Tensor) -> float:
        return float((_joint_pos(entity, list(env.entity_indexing(entity).joint_names)) - baseline).abs().max())

    # ── C. one term drives one robot ─────────────────────────────────────
    print("\n-- C. a term drives its own robot, and only its own --")
    env.reset()
    action_left = zeros.clone()
    tgt_left = _arm_target(LEFT_ROBOT, left_arm_joints, +0.25)
    action_left[:, slices["left_arm"]] = _raw_for_target(terms["left_arm"], tgt_left)
    reset_seen = _step_n(env, action_left, SETTLE_STEPS)
    left_alone = _joint_pos(LEFT_ROBOT, list(left_idx.joint_names))
    moved_left = _delta_from_baseline(LEFT_ROBOT, base_left)
    still_right = _delta_from_baseline(RIGHT_ROBOT, base_right)
    err_left = float((_joint_pos(LEFT_ROBOT, left_arm_joints) - tgt_left).abs().max())
    print(f"  commanded left arm = {_fmt(tgt_left)}")
    print(f"  reached            = {_fmt(_joint_pos(LEFT_ROBOT, left_arm_joints)[0])}")
    print(f"  left moved from baseline  = {moved_left:.5f} (expect >> 0)")
    print(f"  right moved from baseline = {still_right:.5f} (expect ~0)")
    print(f"  (left tracking error vs the request = {err_left:.5f} — reported, not asserted)")
    results["left_command_moves_the_left_arm"] = moved_left > MOVED_MIN and _finite(left_alone)
    results["left_command_leaves_the_right_arm_alone"] = still_right < HOLD_TOL
    measured["left_cmd_moved_left"] = round(moved_left, 5)
    measured["left_cmd_moved_right"] = round(still_right, 5)
    measured["left_cmd_tracking_err"] = round(err_left, 5)
    measured["episode_reset_midway"] = reset_seen

    env.reset()
    action_right = zeros.clone()
    tgt_right = _arm_target(RIGHT_ROBOT, right_joints, -0.25)
    action_right[:, slices["right_arm"]] = _raw_for_target(terms["right_arm"], tgt_right)
    _step_n(env, action_right, SETTLE_STEPS)
    right_alone = _joint_pos(RIGHT_ROBOT, list(right_idx.joint_names))
    moved_right = _delta_from_baseline(RIGHT_ROBOT, base_right)
    still_left = _delta_from_baseline(LEFT_ROBOT, base_left)
    err_right = float((_joint_pos(RIGHT_ROBOT, right_joints) - tgt_right).abs().max())
    print(f"  commanded right arm = {_fmt(tgt_right)}")
    print(f"  reached             = {_fmt(_joint_pos(RIGHT_ROBOT, right_joints)[0])}")
    print(f"  right moved from baseline = {moved_right:.5f} (expect >> 0)")
    print(f"  left moved from baseline  = {still_left:.5f} (expect ~0)")
    print(f"  (right tracking error vs the request = {err_right:.5f} — reported, not asserted)")
    results["right_command_moves_the_right_arm"] = moved_right > MOVED_MIN and _finite(right_alone)
    results["right_command_leaves_the_left_arm_alone"] = still_left < HOLD_TOL
    measured["right_cmd_moved_right"] = round(moved_right, 5)
    measured["right_cmd_moved_left"] = round(still_left, 5)
    measured["right_cmd_tracking_err"] = round(err_right, 5)

    # ── D. two terms on one robot ────────────────────────────────────────
    print("\n-- D. two terms on one robot --")
    env.reset()
    action_grip = zeros.clone()
    tgt_grip = _arm_target(LEFT_ROBOT, left_grip_joints, -0.015)
    action_grip[:, slices["left_gripper"]] = _raw_for_target(terms["left_gripper"], tgt_grip)
    _step_n(env, action_grip, SETTLE_STEPS)
    grip_alone = _joint_pos(LEFT_ROBOT, left_grip_joints)
    grip_moved = float((grip_alone - _home(LEFT_ROBOT, left_grip_joints)).abs().max())
    base_arm = base_left[:, [list(left_idx.joint_names).index(j) for j in left_arm_joints]]
    arm_still = float((_joint_pos(LEFT_ROBOT, left_arm_joints) - base_arm).abs().max())
    err_grip = float((grip_alone - tgt_grip).abs().max())
    print(f"  commanded gripper = {_fmt(tgt_grip)}   reached = {_fmt(grip_alone[0])}")
    print(f"  gripper moved from home = {grip_moved:.5f}   tracking error = {err_grip:.5f}")
    print(f"  its own arm moved from baseline = {arm_still:.5f} (expect ~0)")
    results["gripper_command_moves_the_gripper"] = grip_moved > 0.005 and err_grip < 0.005
    results["gripper_command_leaves_its_own_arm_alone"] = arm_still < HOLD_TOL
    measured["grip_cmd_moved"] = round(grip_moved, 5)
    measured["grip_cmd_err"] = round(err_grip, 5)
    measured["grip_cmd_arm_moved"] = round(arm_still, 5)

    # ── E. every term at once ────────────────────────────────────────────
    # The sharpest statement of independence available, and the one that
    # needs no tolerance on tracking at all: commanding all three together
    # must put every joint exactly where commanding that term ALONE put it.
    # A term that reads another's action slice, or erases another's target,
    # lands somewhere else.
    print("\n-- E. all three terms at once == each term alone --")
    env.reset()
    action_all = zeros.clone()
    action_all[:, slices["left_arm"]] = _raw_for_target(terms["left_arm"], tgt_left)
    action_all[:, slices["left_gripper"]] = _raw_for_target(terms["left_gripper"], tgt_grip)
    action_all[:, slices["right_arm"]] = _raw_for_target(terms["right_arm"], tgt_right)
    _step_n(env, action_all, SETTLE_STEPS)
    d_arm = float((_joint_pos(LEFT_ROBOT, left_arm_joints) - left_alone[:, : len(left_arm_joints)]).abs().max())
    d_grip = float((_joint_pos(LEFT_ROBOT, left_grip_joints) - grip_alone).abs().max())
    d_right = float((_joint_pos(RIGHT_ROBOT, right_joints) - right_alone[:, : len(right_joints)]).abs().max())
    print(f"  left arm vs commanded-alone = {d_arm:.5f}")
    print(f"  gripper  vs commanded-alone = {d_grip:.5f}")
    print(f"  right arm vs commanded-alone = {d_right:.5f}")
    results["terms_compose_without_interfering"] = max(d_arm, d_grip, d_right) < COMPOSE_TOL
    measured["compose_left_delta"] = round(d_arm, 5)
    measured["compose_grip_delta"] = round(d_grip, 5)
    measured["compose_right_delta"] = round(d_right, 5)

    # ── E. reset ─────────────────────────────────────────────────────────
    print("\n-- E. reset --")
    env.reset()
    env._invalidate_cache()
    back_left = (_joint_pos(LEFT_ROBOT, list(left_idx.joint_names)) - home_left).abs().max()
    back_right = (_joint_pos(RIGHT_ROBOT, list(right_idx.joint_names)) - home_right).abs().max()
    noise = max(abs(v) for v in preset.reset_joint_position_noise)
    print(f"  after reset: left max|q - home| = {float(back_left):.5f}  right = {float(back_right):.5f}")
    print(f"  (reset noise is +-{noise}, so both should be under it)")
    results["reset_restores_the_left_arm"] = float(back_left) <= noise + 1e-3
    results["reset_restores_the_right_arm"] = float(back_right) <= noise + 1e-3
    measured["reset_left_offset"] = round(float(back_left), 5)
    measured["reset_right_offset"] = round(float(back_right), 5)

    # ── F. what the policy currently sees ────────────────────────────────
    # Reported, not asserted: the observation terms read the driven robot
    # through the single-robot shortcut, so the second arm is not in the
    # observation yet. Stated here so the number is on the record rather
    # than discovered later.
    obs = env.obs_manager.get_observation()
    obs_dim = int(obs["actor"].shape[-1])
    print(f"\n-- F. observation width = {obs_dim} (actor group) --")
    print("     Reads the driven robot only — the second arm is not observed yet.")
    measured["obs_dim"] = obs_dim

    print("=" * 78)
    print("VERDICT")
    ok = True
    for k, v in results.items():
        print(f"  {k:<44}: {'PASS' if v else 'FAIL'}")
        ok = ok and v
    print(f"  {'OVERALL':<44}: {'PASS' if ok else 'FAIL'}")
    print()
    print("REPORTED")
    for k, v in measured.items():
        print(f"  {k:<44}: {v}")
    print("=" * 78)
    return {"results": results, "measured": measured, "ok": ok}


def run_all(num_envs: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="yam_dual_"))
    out: dict[str, dict] = {}
    env_vars = dict(os.environ, JAXRLWORLD_ALLOW_MULTI_SIM="1")

    for sim in _SIMS:
        result_path = tmp / f"{sim}.json"
        cmd = [
            sys.executable,
            "-m",
            "rlworld.scripts.diag.yam_dual_smoke",
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
    print(f"{'check':<46}" + "".join(f"{s:>10}" for s in _SIMS))
    print("-" * 78)
    overall = True
    for k in keys:
        row = f"{k:<46}"
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
        print(f"  {k:<28}" + "".join(f"{v:>18}" for v in vals) + f"   {agree}")

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
