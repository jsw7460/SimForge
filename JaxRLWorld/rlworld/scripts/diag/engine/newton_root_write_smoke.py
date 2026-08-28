"""Newton root-write diag — where a root pose lands, and whether it lands.

``NewtonRobotStateWriter`` used to write a root pose by slicing
``joint_q[:, 0:7]``. That is only the root for a *floating* articulation.
A welded one has no root coordinates at all, so those seven slots are its
first seven JOINT angles — a root write replaced them with position and
quaternion numbers, silently, because the array is long enough for the
slice to succeed. The destination is now resolved once from
``ArticulationView.get_root_transforms``, which branches on base type.

This diag pins that behaviour down with numbers:

* the binding aliases simulator memory rather than a staging copy (a
  write into a copy vanishes with no error, which reads as "reset
  quietly stopped working");
* a per-env root write reads back exactly, through the ordinary
  ``RobotData`` accessors, not the writer's own buffer;
* a root write leaves every joint angle untouched — the regression that
  motivated the change;
* the reset event places the robot per env.

Run::

    python -m rlworld.scripts.diag.engine.newton_root_write_smoke --num-envs 4
"""

from __future__ import annotations

import argparse

import torch

from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.runners import BaseRunner


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):+.5f}" for x in v) + "]"


def _build(num_envs: int):
    cfgs = Go2FlatConfig(sim_type="newton", num_envs=num_envs).build()
    # Pin the spawn so a reset is reproducible and the expected pose is known.
    cfgs.event.reset_root.params["pose_range"] = {}
    cfgs.event.reset_root.params["velocity_range"] = {}
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return env, cfgs


def run(num_envs: int) -> int:
    env, cfgs = _build(num_envs)
    results: dict[str, bool] = {}
    reported: dict[str, object] = {}

    name = "robot"
    writer = env.get_root_state_writer(name)
    data = env.get_entity_data(name)
    view = env.scene_manager.articulation_views[name]
    all_ids = torch.arange(env.num_envs, device=env.device)

    print("=" * 78)
    print(f"NEWTON ROOT-WRITE DIAG  [entity={name}, num_envs={env.num_envs}]")
    print("=" * 78)

    # ── binding ──────────────────────────────────────────────────────────
    bound = writer._root_pose
    print(f"[bind] is_floating_base = {view.is_floating_base}   count_per_world = {view.count_per_world}")
    print(f"[bind] bound root-pose view shape = {tuple(bound.shape)} (expect ({env.num_envs}, 7))")
    results["bind_shape"] = tuple(bound.shape) == (env.num_envs, 7)
    reported["is_floating_base"] = view.is_floating_base

    # The binding must alias the array the reads come from. Compare against a
    # read taken through the ordinary accessor rather than trusting the ptr.
    pos_before = data.root_link_pos_w.clone()
    print(f"[bind] root_link_pos_w (read path) = {_fmt(pos_before[0])}")
    print(f"[bind] bound view      [0, 0:3]    = {_fmt(bound[0, 0:3])}")
    results["bind_matches_read_path"] = bool(torch.allclose(bound[:, 0:3], pos_before, atol=1e-6))

    # ── a root write must reach the read path ────────────────────────────
    joints_before = data.joint_pos.clone()
    target_pos = torch.stack(
        [torch.tensor([1.0 + i, -0.5 * i, 0.30 + 0.01 * i], device=env.device) for i in range(env.num_envs)]
    )
    target_quat = torch.zeros(env.num_envs, 4, device=env.device)
    target_quat[:, 0] = 1.0

    writer.set_root_pose(target_pos, target_quat, env_ids=all_ids)
    writer.eval_fk(env_ids=all_ids)
    env._invalidate_cache()

    pos_after = data.root_link_pos_w
    quat_after = data.root_link_quat_w
    err_pos = float((pos_after - target_pos).abs().max())
    err_quat = float((quat_after - target_quat).abs().max())
    print(f"[write] per-env target x = {[round(float(v), 3) for v in target_pos[:, 0]]}")
    print(f"[write] read back     x = {[round(float(v), 3) for v in pos_after[:, 0]]}   max err = {err_pos:.3e}")
    print(f"[write] quat max err = {err_quat:.3e}")
    results["root_write_reaches_read_path"] = err_pos < 1e-5
    results["root_write_quat_exact"] = err_quat < 1e-5
    reported["root_write_pos_err"] = err_pos

    # ── and must not touch a single joint angle ──────────────────────────
    joints_after = data.joint_pos
    joint_drift = float((joints_after - joints_before).abs().max())
    print(f"[write] joint_pos.shape = {tuple(joints_after.shape)}")
    print(f"[write] max |Δjoint_pos| caused by the root write = {joint_drift:.3e} (expect 0)")
    results["root_write_leaves_joints_alone"] = joint_drift < 1e-9
    reported["joint_drift"] = joint_drift

    # ── a subset write must leave the other envs alone ───────────────────
    subset = all_ids[: max(1, env.num_envs // 2)]
    before_all = data.root_link_pos_w.clone()
    moved = before_all[subset] + torch.tensor([0.0, 0.0, 0.25], device=env.device)
    writer.set_root_pose(moved, target_quat[subset], env_ids=subset)
    writer.eval_fk(env_ids=subset)
    env._invalidate_cache()
    after_all = data.root_link_pos_w
    touched_err = float((after_all[subset] - moved).abs().max())
    untouched = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    untouched[subset] = False
    other_drift = float((after_all[untouched] - before_all[untouched]).abs().max()) if untouched.any() else 0.0
    print(f"[subset] wrote envs {subset.tolist()}: max err = {touched_err:.3e}")
    print(f"[subset] other envs drift = {other_drift:.3e} (expect 0)")
    results["subset_write_lands"] = touched_err < 1e-5
    results["subset_write_is_scoped"] = other_drift < 1e-9

    # ── the reset event still places the robot ───────────────────────────
    env.reset()
    reset_pos = env.get_entity_data(name).root_link_pos_w
    expected = env.scene_manager.env_origins + torch.tensor(
        cfgs.scene.entities[name].init_state.pos, device=env.device
    ).unsqueeze(0)
    reset_err = float((reset_pos - expected).abs().max())
    print(f"[reset] root_link_pos_w env0 = {_fmt(reset_pos[0])}   expected {_fmt(expected[0])}")
    print(f"[reset] max err over {env.num_envs} envs = {reset_err:.3e}")
    results["reset_places_robot"] = reset_err < 1e-4

    # ── velocity destination ─────────────────────────────────────────────
    has_vel = writer._root_vel is not None
    print(f"[vel] root velocity destination bound: {has_vel} (floating base -> True)")
    results["velocity_binding_matches_base_type"] = has_vel == view.is_floating_base

    print("=" * 78)
    print("VERDICT")
    ok = True
    for k, v in results.items():
        print(f"  {k:<36}: {'PASS' if v else 'FAIL'}")
        ok = ok and v
    print(f"  {'OVERALL':<36}: {'PASS' if ok else 'FAIL'}")
    print()
    print("REPORTED")
    for k, v in reported.items():
        print(f"  {k:<36}: {v}")
    print("=" * 78)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4)
    args = ap.parse_args()
    return run(args.num_envs)


if __name__ == "__main__":
    raise SystemExit(main())
