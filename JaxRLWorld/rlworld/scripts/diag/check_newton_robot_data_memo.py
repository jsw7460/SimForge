"""Bitwise check: NewtonRobotData per-step memoization.

``newton/robot_data.py`` memoizes its zero-argument reads on
``env._cache_generation`` (bumped every physics substep and after
resets/events). This diag proves, on a live Newton env stepped with
random actions:

1. **Freshness** — every memoized read equals a direct call of its
   undecorated function (``__wrapped__``) at every step, bitwise. A
   stale cache (missed generation bump) fails here immediately.
2. **Hit behaviour** — a second read within the same generation returns
   the SAME tensor object (the cache actually engages).
3. **Contact-force cache** — the Newton contact manager's per-group
   memoized force equals a direct ``sensor.compute_force()``.

Usage:
    jaxpy -m rlworld.scripts.diag.check_newton_robot_data_memo
    jaxpy -m rlworld.scripts.diag.check_newton_robot_data_memo --steps 50
"""

from __future__ import annotations

import argparse
import importlib

import torch

_MEMO_PROPERTIES = (
    "root_link_pos_w",
    "root_link_quat_w",
    "root_link_lin_vel_w",
    "root_link_ang_vel_w",
    "root_link_lin_vel_b",
    "root_link_ang_vel_b",
    "root_com_pos_w",
    "root_com_lin_vel_w",
    "root_com_lin_vel_b",
    "projected_gravity_b",
    "heading_w",
    "body_pos_w_all",
    "body_quat_w_all",
    "body_lin_vel_w_all",
    "body_ang_vel_w_all",
    "body_com_pos_w_all",
    "body_com_lin_vel_w_all",
    "joint_pos",
    "joint_vel",
    "applied_torque",
)
_MEMO_METHODS = ("_body_q_view", "_body_qd_view", "_joint_coords", "_joint_dofs", "_angular_momentum_w_cached")


def _build_env(preset: str, num_envs: int):
    from rlworld.rl.runners import BaseRunner

    if ":" in preset:
        mod_path, cls_name = preset.split(":", 1)
    else:
        mod_path, cls_name = (
            "rlworld.rl.configs.presets.go2.newton.gait_conditioned",
            "Go2GaitConditionedNewtonConfig",
        )
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type="newton", num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="go2_gait")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = _build_env(args.preset, args.num_envs)
    rd = env.get_robot_data(env.robot_entity_name)
    rd_cls = type(rd)

    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    failures = 0
    for step in range(args.steps):
        env.step(0.1 * torch.randn_like(action))

        for name in _MEMO_PROPERTIES:
            cached = getattr(rd, name)
            fresh = getattr(rd_cls, name).fget.__wrapped__(rd)
            if not torch.equal(cached, fresh):
                failures += 1
                print(f"[step {step:3d}] STALE property {name}")
            if getattr(rd, name) is not cached:
                failures += 1
                print(f"[step {step:3d}] NO CACHE HIT on property {name}")

        for name in _MEMO_METHODS:
            cached = getattr(rd, name)()
            fresh = getattr(rd_cls, name).__wrapped__(rd)
            if not torch.equal(cached, fresh):
                failures += 1
                print(f"[step {step:3d}] STALE method {name}")
            if getattr(rd, name)() is not cached:
                failures += 1
                print(f"[step {step:3d}] NO CACHE HIT on method {name}")

        cm = env.contact_manager
        for gname, group in cm._groups.items():
            cached = cm._compute_group_contact_force(group)
            fresh = cm._group_sensors[gname].compute_force()
            if cached is None or not torch.equal(cached, fresh):
                failures += 1
                print(f"[step {step:3d}] STALE contact force for group {gname!r}")
            if cm._compute_group_contact_force(group) is not cached:
                failures += 1
                print(f"[step {step:3d}] NO CACHE HIT on contact group {gname!r}")

        if failures == 0 and step % 10 == 0:
            print(f"[step {step:3d}] all memoized reads fresh + cache hits engaged")

    if failures:
        print(f"\nFAIL — {failures} stale/uncached reads over {args.steps} steps")
        return 1
    n = len(_MEMO_PROPERTIES) + len(_MEMO_METHODS)
    print(f"\nPASS — {n} memoized reads bit-fresh with cache hits, {args.steps} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
