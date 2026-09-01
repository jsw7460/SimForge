"""Does the lift task's reward actually lead where the task means to go?

A reward function is a claim about which states are better. Training is
an expensive and slow way to discover the claim was wrong, and a wrong
one usually still produces a rising curve — of whatever it happens to
reward. So the claims are checked directly, by putting the scene in
states whose ranking is known in advance and reading the terms back.

The states are constructed rather than reached: the cube is teleported
and the arm commanded, so each comparison isolates one variable.

What it checks:

* **The command samples inside its declared box**, per env, and adds the
  environment's own origin — otherwise every env in a grid-laid-out
  backend gets goals belonging to env 0.
* **Reaching rises as the hand nears the cube.** Monotone over a swept
  distance, not merely different at two points.
* **Bringing is gated on reaching.** The whole design is a product: with
  the hand far away, moving the cube ONTO the goal must not pay. If it
  does, the policy can farm the bringing term without ever grasping, and
  a sum would have been written instead of a product.
* **The goal is achievable.** Placing the cube at the goal registers as
  at_goal, and the success flag latches for the rest of the episode.
* **Dropping the cube is both penalised and terminal**, and neither
  fires while the cube is on the table.
* **The observations are the vectors they claim to be**, checked against
  a hand computation from world positions rather than against
  themselves.

Run all three and cross-compare::

    python -m jaxrlworld.scripts.diag.yam.yam_lift_task_diag --num-envs 4
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

from jaxrlworld.rl.configs.presets.yam_lift.base import CUBE, DROPPED_Z, GRASP_SITE, YamLiftConfig
from jaxrlworld.rl.configs.rewards.reward_term_config import get_weight_value
from jaxrlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.rl.utils.quat_utils import quat_inv_wxyz, quat_rotate_wxyz

_SIMS = ("genesis", "newton", "mujoco")


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):+.4f}" for x in v) + "]"


def _build(sim: str, num_envs: int):
    preset = YamLiftConfig(sim_type=sim, num_envs=num_envs)
    env = BaseRunner._create_env_from_config(preset.build())
    env.reset()
    return env, preset


def _place_cube(env, pos: torch.Tensor) -> None:
    """Teleport the cube and make the write visible to every reader.

    A write alone is not enough. Newton needs its forward kinematics
    evaluated, and mjlab keeps derived state — body poses, site poses,
    sensors — until a ``forward()`` recomputes it from qpos. Skipping
    either leaves every reader looking at the pose from before the
    write, which reads as a reward that ignores the cube rather than as
    a cube that did not move.
    """
    writer = env.get_root_state_writer(CUBE)
    quat = torch.zeros(env.num_envs, 4, device=env.device)
    quat[:, 0] = 1.0
    writer.set_root_pose(pos, quat)
    writer.set_root_velocity(
        torch.zeros(env.num_envs, 3, device=env.device),
        torch.zeros(env.num_envs, 3, device=env.device),
    )
    writer.eval_fk()
    env._post_reset_forward()
    env._invalidate_cache()


def _reward(env, name: str) -> torch.Tensor:
    """One reward term's raw value, before weight and dt."""
    manager = env.reward_manager
    term = manager.reward_terms[name]
    return manager._resolved_fns[name](env, **term.params)


def run_single(sim: str, num_envs: int) -> dict:
    env, preset = _build(sim, num_envs)
    results: dict[str, bool] = {}
    measured: dict[str, object] = {}

    command = env.command_manager.get_term("lift")
    grasp = env.resolve_selector(SceneEntitySelector(name="robot", site_names=(GRASP_SITE,)))
    data = env.get_entity_data("robot")
    origins = env.scene_manager.env_origins

    def ee_pos() -> torch.Tensor:
        return data.site_pos_w_by_ids(grasp.site_ids)[:, 0]

    print("=" * 78)
    print(f"YAM LIFT TASK DIAG  [sim={sim}]")
    print("=" * 78)

    # ── A. the command ───────────────────────────────────────────────────
    print("\n-- A. command sampling --")
    env.reset()
    env._invalidate_cache()
    goal_local = command.target_pos - origins
    cube_local = env.get_entity_data(CUBE).root_link_pos_w - origins
    cfg = env.command_manager.config.terms["lift"]
    print(f"  goal (env frame) = {_fmt(goal_local[0])}   declared x{cfg.target_x} y{cfg.target_y} z{cfg.target_z}")
    print(f"  cube (env frame) = {_fmt(cube_local[0])}   declared x{cfg.object_x} y{cfg.object_y} z{cfg.object_z}")
    in_box = lambda v, lo, hi: bool(((v >= lo - 1e-4) & (v <= hi + 1e-4)).all())  # noqa: E731
    results["goal_inside_its_declared_box"] = (
        in_box(goal_local[:, 0], *cfg.target_x)
        and in_box(goal_local[:, 1], *cfg.target_y)
        and in_box(goal_local[:, 2], *cfg.target_z)
    )
    results["cube_inside_its_declared_box"] = in_box(cube_local[:, 0], *cfg.object_x) and in_box(
        cube_local[:, 1], *cfg.object_y
    )
    # Per env, not one goal broadcast to all: a shared goal trains a policy
    # that ignores the observation telling it where to go.
    spread = float((goal_local.max(dim=0).values - goal_local.min(dim=0).values).max()) if num_envs > 1 else 1.0
    print(f"  goal spread across envs = {spread:.4f} (expect > 0 with several envs)")
    results["goals_differ_between_envs"] = spread > 1e-6 or num_envs == 1
    measured["goal_env0"] = [round(float(v), 4) for v in goal_local[0]]
    measured["cube_env0"] = [round(float(v), 4) for v in cube_local[0]]

    # ── B. reaching rises as the hand nears the cube ─────────────────────
    # The cube is moved TOWARD the resting hand rather than the arm driven,
    # so the only thing that changes between samples is the distance.
    print("\n-- B. reaching --")
    env.reset()
    for _ in range(40):
        env.step(torch.zeros(env.num_envs, env.act_manager.num_actions, device=env.device))
    env._invalidate_cache()
    hand = ee_pos().clone()
    goal = command.target_pos.clone()
    reaching: list[float] = []
    for frac in (1.0, 0.75, 0.5, 0.25, 0.0):
        # Along the line from the cube's start to the hand.
        start = torch.tensor([preset.cube_pos[0], preset.cube_pos[1], preset.cube_pos[2]], device=env.device) + origins
        _place_cube(env, start + (hand - start) * (1.0 - frac))
        reaching.append(float(_reward(env, "staged").mean()))
    print("  staged reward as the cube approaches the hand:")
    for frac, value in zip((1.0, 0.75, 0.5, 0.25, 0.0), reaching, strict=True):
        print(f"    gap {frac:.2f} of the way : {value:.5f}")
    rising = all(b >= a - 1e-6 for a, b in zip(reaching, reaching[1:], strict=False))
    results["reaching_rises_as_the_hand_nears"] = rising
    measured["reaching_curve"] = [round(v, 5) for v in reaching]

    # ── C. bringing is gated on reaching ─────────────────────────────────
    # The claim the product makes: with the hand far from the cube, putting
    # the cube on the goal must pay almost nothing. A sum would pay in full.
    print("\n-- C. the bringing bonus is gated --")
    far = goal + torch.tensor([0.0, 0.0, 0.0], device=env.device)
    _place_cube(env, far)  # cube AT the goal, hand still at rest far below
    staged_cube_at_goal = float(_reward(env, "staged").mean())
    bring_cube_at_goal = float(_reward(env, "bring").mean())
    gap = float((ee_pos() - command.target_pos).norm(dim=-1).mean())
    print(f"  cube placed ON the goal, hand {gap:.4f} m away")
    print(f"    bring  (ungated) = {bring_cube_at_goal:.5f}   <- near 1, the goal IS met")
    print(f"    staged (gated)   = {staged_cube_at_goal:.5f}   <- near 0, the hand is not there")
    results["bring_pays_when_the_cube_is_at_the_goal"] = bring_cube_at_goal > 0.5
    results["staged_withholds_until_the_hand_arrives"] = staged_cube_at_goal < 0.1 * bring_cube_at_goal
    measured["bring_at_goal"] = round(bring_cube_at_goal, 5)
    measured["staged_at_goal_hand_away"] = round(staged_cube_at_goal, 5)

    # ── D. success ───────────────────────────────────────────────────────
    print("\n-- D. success --")
    _place_cube(env, command.target_pos.clone())
    err = float(command.position_error.mean())
    at_goal = bool(command.at_goal.all())
    command._update_command()
    latched = bool((command.episode_success > 0.5).all())
    _place_cube(env, command.target_pos + torch.tensor([0.0, 0.0, -0.3], device=env.device))
    command._update_command()
    still_latched = bool((command.episode_success > 0.5).all())
    print(f"  cube on the goal: error {err:.5f} m, at_goal {at_goal}, latched {latched}")
    print(f"  cube then moved away: still latched {still_latched} (success is a whole-episode fact)")
    results["cube_on_the_goal_counts_as_at_goal"] = at_goal and err < cfg.success_threshold
    results["success_latches_for_the_episode"] = latched and still_latched
    measured["goal_error_when_placed"] = round(err, 5)

    # ── E. dropping ──────────────────────────────────────────────────────
    print("\n-- E. dropping the cube --")
    on_table = origins + torch.tensor([preset.cube_pos[0], preset.cube_pos[1], preset.cube_pos[2]], device=env.device)
    _place_cube(env, on_table)
    drop_penalty_on_table = float(_reward(env, "dropped").mean())
    from jaxrlworld.rl.configs.presets.yam_lift.base import _cube_below

    # The termination's own selector, for the same reason as in section F:
    # this must exercise what the task is configured with.
    drop_cfg = env.termination_manager._all_terms["cube_dropped"].params["object_cfg"]
    term_on_table = bool(_cube_below(env, object_cfg=drop_cfg, min_height=DROPPED_Z).reset.any())
    below = on_table.clone()
    below[:, 2] = DROPPED_Z - 0.05
    _place_cube(env, below)
    drop_penalty_off = float(_reward(env, "dropped").mean())
    term_off = bool(_cube_below(env, object_cfg=drop_cfg, min_height=DROPPED_Z).reset.all())
    print(f"  on the table : penalty {drop_penalty_on_table:.3f}  terminates {term_on_table}")
    print(f"  below it     : penalty {drop_penalty_off:.3f}  terminates {term_off}")
    results["no_drop_penalty_while_on_the_table"] = drop_penalty_on_table < 1e-6 and not term_on_table
    results["dropping_is_penalised_and_terminal"] = drop_penalty_off > 0.5 and term_off
    measured["drop_penalty_on_table"] = round(drop_penalty_on_table, 5)
    measured["drop_penalty_off_table"] = round(drop_penalty_off, 5)

    # ── E2. every term's WEIGHTED sign ───────────────────────────────────
    # The check that was missing, and it cost a training run. Each term is
    # read here after its weight, which is the number the policy actually
    # optimises. A penalty whose function already returns a negative value
    # and is then given a negative weight becomes a BONUS for the thing it
    # exists to discourage — no error, no shape mismatch, and a training
    # curve that rises on the one term that should be falling.
    print("\n-- E2. weighted sign of every reward term --")
    manager = env.reward_manager
    # A state where every penalty should be biting: joints driven hard
    # into their stops, actions changing every step, cube off the table.
    env.reset()
    big = torch.full((env.num_envs, env.act_manager.num_actions), 60.0, device=env.device)
    for step in range(20):
        env.step(big if step % 2 == 0 else -big)
    env._invalidate_cache()
    _place_cube(env, below)

    expected_sign = {
        "staged": +1,
        "bring": +1,
        "action_rate": -1,
        "joint_pos_limits": -1,
        "joint_vel": -1,
        "dropped": -1,
    }
    sign_ok = True
    for name, term in manager.reward_terms.items():
        raw = float(_reward(env, name).mean())
        weight = float(get_weight_value(term.weight, manager.env_step_calls))
        weighted = raw * weight
        want = expected_sign.get(name)
        mark = ""
        if want is not None:
            good = weighted >= -1e-9 if want > 0 else weighted <= 1e-9
            sign_ok = sign_ok and good
            mark = "  <-- WRONG SIGN" if not good else ""
        print(f"  {name:<18} raw {raw:+.5f}  x weight {weight:+.4f}  = {weighted:+.5f}{mark}")
    print("  (penalties must be <= 0 after their weight; task terms >= 0)")
    results["every_term_has_the_sign_it_should"] = sign_ok

    # ── F. the observations are what they claim ──────────────────────────
    # Recomputed from world positions here, so the check does not simply
    # run the same expression twice.
    print("\n-- F. observations --")
    env.reset()
    for _ in range(20):
        env.step(torch.zeros(env.num_envs, env.act_manager.num_actions, device=env.device))
    env._invalidate_cache()
    from jaxrlworld.rl.envs.mdp.observations.common import manipulation as manip

    # The selector the preset itself hands the terms, so this checks what
    # the task runs with rather than a cube-shaped guess at it.
    obj_cfg = env.reward_manager.reward_terms["bring"].params["object_cfg"]
    obj_w = env.get_entity_data(CUBE).root_link_pos_w
    base_q = data.root_link_quat_w
    expect_ee_to_cube = quat_rotate_wxyz(quat_inv_wxyz(base_q), obj_w - ee_pos())
    got_ee_to_cube = manip.ee_to_object_distance(env, object_cfg=obj_cfg, asset_cfg=grasp)
    err_ee = float((expect_ee_to_cube - got_ee_to_cube).abs().max())

    expect_to_goal = quat_rotate_wxyz(quat_inv_wxyz(base_q), command.target_pos - obj_w)
    got_to_goal = manip.object_to_goal_distance(env, object_cfg=obj_cfg, command_name="lift", asset_cfg=grasp)
    err_goal = float((expect_to_goal - got_to_goal).abs().max())

    height = manip.object_height(env, object_cfg=obj_cfg, reference_height=preset.table_pos[2] * 2)
    print(f"  ee_to_cube      : {_fmt(got_ee_to_cube[0])}   err vs hand-computed {err_ee:.2e}")
    print(f"  cube_to_goal    : {_fmt(got_to_goal[0])}   err vs hand-computed {err_goal:.2e}")
    print(f"  cube height over the table = {float(height.mean()):.4f} m")
    results["ee_to_cube_is_the_vector_it_claims"] = err_ee < 1e-5
    results["cube_to_goal_is_the_vector_it_claims"] = err_goal < 1e-5
    # A cube at rest on the table is one half-cube above it.
    results["object_height_is_measured_from_the_table"] = abs(float(height.mean()) - 0.02) < 0.01
    measured["obs_ee_to_cube_err"] = f"{err_ee:.2e}"
    measured["obs_cube_to_goal_err"] = f"{err_goal:.2e}"
    measured["cube_height_over_table"] = round(float(height.mean()), 5)

    print("=" * 78)
    print("VERDICT")
    ok = True
    for k, v in results.items():
        print(f"  {k:<48}: {'PASS' if v else 'FAIL'}")
        ok = ok and v
    print(f"  {'OVERALL':<48}: {'PASS' if ok else 'FAIL'}")
    print()
    print("REPORTED")
    for k, v in measured.items():
        print(f"  {k:<48}: {v}")
    print("=" * 78)
    return {"results": results, "measured": measured, "ok": ok}


def run_all(num_envs: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="yam_lift_task_"))
    out: dict[str, dict] = {}
    env_vars = dict(os.environ, JAXRLWORLD_ALLOW_MULTI_SIM="1")

    for sim in _SIMS:
        result_path = tmp / f"{sim}.json"
        cmd = [
            sys.executable,
            "-m",
            "jaxrlworld.scripts.diag.yam.yam_lift_task_diag",
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
    print(f"{'check':<50}" + "".join(f"{s:>10}" for s in _SIMS))
    print("-" * 78)
    overall = True
    for k in keys:
        row = f"{k:<50}"
        for s in _SIMS:
            v = out.get(s, {}).get("results", {}).get(k)
            row += f"{'—' if v is None else ('PASS' if v else 'FAIL'):>10}"
            overall = overall and (v is None or bool(v))
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
        print(f"  {k:<30}" + "".join(f"{v:>24}" for v in vals) + f"   {agree}")

    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 78)
    return 0 if overall else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=list(_SIMS), default=None)
    ap.add_argument("--result-json", default=None)
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
