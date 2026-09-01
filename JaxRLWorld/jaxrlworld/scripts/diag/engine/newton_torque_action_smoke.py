"""Newton direct-torque (``JointEffortAction``) smoke / verification.

``JointEffortAction`` interprets the raw policy output as a **joint torque**
(``processed = raw*scale + offset``, optional ``effort_limit`` clamp) and
writes it straight to the simulator's ``joint_f``, bypassing the
position-target / actuator-PD path. The term + the
``_apply_joint_effort_via_indices -> _apply_force`` path exist on all three
backends but NO preset wires them, so they have never been exercised in a
real run. This diag verifies the Newton path end-to-end with scripted
torques (no RL, no learning) and dumps every relevant quantity.

It builds the Go2 Newton env, replaces the action config with a single
``JointEffortAction`` term (Go2 already drives joints via explicit IdealPD,
so the Newton joints sit in direct-torque / NONE mode), then runs four
checks:

  Phase 1 (indexing/scatter) : command a ONE-HOT torque; verify ``joint_f``
      reads back the command on exactly that joint and 0 elsewhere, and that
      ``act_manager.applied_torque`` matches. Catches qd-index / scatter bugs.
  Phase 2 (scale)            : command all joints; verify ``joint_f == raw*scale``.
  Phase 3 (effort_limit)     : command beyond the limit; verify ``joint_f`` is
      clamped to +/- the limit (both signs).
  Phase 0 (NONE-mode)        : command ZERO torque from the standing reset
      pose; the base must drop under gravity. If an internal PD were active
      (POSITION mode) the robot would hold its pose -> would mean the joints
      are NOT in direct-torque mode.
  Phase 4 (dynamics dump)    : apply a constant modest torque and dump per-step
      joint pos/vel + joint_f so the torque->motion sign/magnitude can be
      eyeballed.

Run (on the GPU box):
    python jaxrlworld/scripts/diag/newton_torque_action_smoke.py
    python jaxrlworld/scripts/diag/newton_torque_action_smoke.py --steps 30
"""

from __future__ import annotations

import argparse

import torch
import warp as wp

from jaxrlworld.rl.configs.newton_config_classes import NewtonActionConfig
from jaxrlworld.rl.configs.presets.go2.base import Go2FlatConfig
from jaxrlworld.rl.envs.mdp.actions import JointEffortAction, JointEffortActionCfg
from jaxrlworld.rl.runners import BaseRunner

SCALE = 2.0  # raw action -> torque multiplier (non-trivial so the scale path is tested)
EFFORT_LIMIT = 30.0  # N*m saturation
TOL = 1e-3


def _build_torque_env(num_envs: int):
    """Go2 Newton env with the action term swapped to direct torque."""
    cfg = Go2FlatConfig(sim_type="newton", num_envs=num_envs)
    cfgs = cfg.build()
    r = cfg.robot

    cfgs.action = NewtonActionConfig(
        actuated_dof_names=r.actuated_dof_patterns,
        clip_actions=(-1.0e9, 1.0e9),  # no pre-clip; we test scale + effort_limit explicitly
        action_terms={
            "effort": JointEffortActionCfg(
                class_type=JointEffortAction,
                joint_names=list(r.actuated_dof_patterns),
                scale=SCALE,
                offset=0.0,
                effort_limit=EFFORT_LIMIT,
            )
        },
    )
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _read_joint_f_actuated(env) -> torch.Tensor:
    """``control.joint_f`` reduced to actuated joints in canonical order: (num_worlds, n_act)."""
    sm = env.scene_manager
    model = sm.model
    num_worlds = model.world_count
    dof_per_world = model.joint_dof_count // num_worlds
    jf = wp.to_torch(sm.control.joint_f).reshape(num_worlds, dof_per_world)
    qd_idx = env.act_manager._indexing.newton_qd_indices
    return jf[:, qd_idx]


def _base_height(env) -> float:
    return float(env.get_robot_data().root_link_pos_w[0, 2])


def _zeros_action(env) -> torch.Tensor:
    return torch.zeros((env.num_envs, env.act_manager.total_action_dim), device=env.device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20, help="steps for the dynamics dump (Phase 4)")
    args = ap.parse_args()

    env = _build_torque_env(args.num_envs)
    names = list(env.act_manager.actuated_joint_names)
    n = len(names)
    dev = env.device

    # ---- Header: everything plausibly relevant on first run ----
    sm = env.scene_manager
    model = sm.model
    print("=" * 78)
    print("NEWTON DIRECT-TORQUE (JointEffortAction) SMOKE")
    print("=" * 78)
    print(
        f"num_envs={env.num_envs}  control_dt={env.control_dt:.5f}  physics_dt={env.physics_dt:.5f}  "
        f"decimation={env.decimation}"
    )
    print(f"actuated joints (n={n}): {names}")
    print(f"scale={SCALE}  effort_limit={EFFORT_LIMIT} N*m  total_action_dim={env.act_manager.total_action_dim}")
    print(f"term-based path active: {bool(env.act_manager._has_action_terms)}")
    qd_idx = env.act_manager._indexing.newton_qd_indices
    print(f"newton_qd_indices: {qd_idx.detach().cpu().tolist()}")
    print(f"gravity: {wp.to_torch(model.gravity).detach().cpu().tolist() if hasattr(model, 'gravity') else 'n/a'}")
    try:
        print(f"body masses [kg]: {wp.to_torch(model.body_mass).detach().cpu().tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"body masses: n/a ({e})")
    # Direct-torque mode marker, if Newton exposes it.
    for attr in ("joint_dof_mode", "joint_target_mode", "joint_axis_mode"):
        if hasattr(model, attr):
            print(f"model.{attr}: {wp.to_torch(getattr(model, attr)).detach().cpu().tolist()}")
    print()

    results: dict[str, bool] = {}
    env.reset()

    # ---- Phase 1: indexing / scatter (one-hot torque) ----
    print("-" * 78)
    print("PHASE 1 — indexing/scatter (one-hot raw=+5 on joint 0)")
    a = _zeros_action(env)
    a[:, 0] = 5.0
    env.step(a)
    jf = _read_joint_f_actuated(env)[0]
    applied = env.act_manager.applied_torque[0]
    expected = (a[0] * SCALE).clamp(-EFFORT_LIMIT, EFFORT_LIMIT)
    print(f"  expected joint_f : {expected.detach().cpu().tolist()}")
    print(f"  read    joint_f  : {jf.detach().cpu().tolist()}")
    print(f"  applied_torque   : {applied.detach().cpu().tolist()}")
    ok_jf = torch.allclose(jf.cpu(), expected.cpu(), atol=TOL)
    ok_app = torch.allclose(applied.cpu(), expected.cpu(), atol=TOL)
    others_zero = bool((jf[1:].abs() < TOL).all())
    print(f"  joint_f==expected: {ok_jf} | applied==expected: {ok_app} | others==0: {others_zero}")
    results["P1_indexing"] = ok_jf and ok_app and others_zero
    print()

    # ---- Phase 2: scale (all joints) ----
    print("-" * 78)
    print(f"PHASE 2 — scale (raw=+5 on all joints, scale={SCALE:.1f})")
    a = _zeros_action(env) + 5.0
    env.step(a)
    jf = _read_joint_f_actuated(env)[0]
    expected = (a[0] * SCALE).clamp(-EFFORT_LIMIT, EFFORT_LIMIT)
    ok = torch.allclose(jf.cpu(), expected.cpu(), atol=TOL)
    print(f"  expected (raw*scale): {expected[0].item():.3f} (all)  read[0]={jf[0].item():.3f}  match={ok}")
    results["P2_scale"] = ok
    print()

    # ---- Phase 3: effort_limit clamp (both signs) ----
    print("-" * 78)
    print(f"PHASE 3 — effort_limit clamp (raw=+/-50 -> |torque| would be 100 > {EFFORT_LIMIT:.0f})")
    a = _zeros_action(env) + 50.0
    env.step(a)
    jf_pos = _read_joint_f_actuated(env)[0]
    a = _zeros_action(env) - 50.0
    env.step(a)
    jf_neg = _read_joint_f_actuated(env)[0]
    ok_pos = torch.allclose(jf_pos.cpu(), torch.full_like(jf_pos.cpu(), EFFORT_LIMIT), atol=TOL)
    ok_neg = torch.allclose(jf_neg.cpu(), torch.full_like(jf_neg.cpu(), -EFFORT_LIMIT), atol=TOL)
    print(f"  +clamp: read[0]={jf_pos[0].item():.3f} (want +{EFFORT_LIMIT}) match={ok_pos}")
    print(f"  -clamp: read[0]={jf_neg[0].item():.3f} (want -{EFFORT_LIMIT}) match={ok_neg}")
    results["P3_effort_limit"] = ok_pos and ok_neg
    print()

    # ---- Phase 0: NONE-mode (zero torque must NOT hold the pose) ----
    print("-" * 78)
    print("PHASE 0 — NONE-mode check (zero torque from standing reset; base must drop under gravity)")
    env.reset()
    h0 = _base_height(env)
    q0 = env.get_robot_data().joint_pos[0].clone()
    for _ in range(30):
        env.step(_zeros_action(env))
    h1 = _base_height(env)
    q1 = env.get_robot_data().joint_pos[0]
    dq = float((q1 - q0).abs().max())
    print(f"  base height: {h0:.4f} -> {h1:.4f} m (drop={h0 - h1:.4f})")
    print(f"  max |Δjoint| over 30 steps: {dq:.4f} rad")
    # With no internal PD, zero torque cannot hold a quadruped up -> base drops / joints move.
    none_mode_ok = (h0 - h1) > 0.05 or dq > 0.05
    print(f"  NONE-mode (no hidden PD holding pose): {none_mode_ok}")
    results["P0_none_mode"] = none_mode_ok
    print()

    # ---- Phase 4: dynamics dump (informational) ----
    print("-" * 78)
    print(f"PHASE 4 — dynamics dump (constant raw=+4 on all joints for {args.steps} steps)")
    env.reset()
    a = _zeros_action(env) + 4.0  # processed = 8 N*m, within limit
    rd = env.get_robot_data()
    print(f"  {'step':>4} | {'base_h':>7} | {'q[0]':>8} | {'qd[0]':>8} | {'joint_f[0]':>10}")
    for s in range(args.steps):
        env.step(a)
        rd = env.get_robot_data()
        jf0 = _read_joint_f_actuated(env)[0, 0].item()
        print(
            f"  {s:4d} | {_base_height(env):7.4f} | {rd.joint_pos[0, 0].item():8.4f} | "
            f"{rd.joint_vel[0, 0].item():8.4f} | {jf0:10.3f}"
        )
    print()

    # ---- Verdict ----
    print("=" * 78)
    print("VERDICT")
    for k, v in results.items():
        print(f"  {k:18s}: {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print(f"  OVERALL           : {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
