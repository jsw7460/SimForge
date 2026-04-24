"""Trace root pose at each stage of the reset pipeline for T1 getup on mjlab.

Answers: does the root pose get written correctly by the reset event
term, or does something downstream overwrite it? mjlab ends up with
root_z=0.700 despite config requesting 0.665 — this bisects why.

Usage:
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_reset_pose.py --sim mujoco
    uv run python JaxRLWorld/rlworld/scripts/diag/t1_reset_pose.py --sim newton
"""
from __future__ import annotations

import argparse

import torch

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner


def _fmt(t) -> str:
    if isinstance(t, torch.Tensor):
        v = t.detach().flatten().cpu().numpy()
    else:
        v = list(t)
    return "[" + ", ".join(f"{float(x):+.6f}" for x in v) + "]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=("newton", "mujoco", "genesis"), required=True)
    args = parser.parse_args()

    cfg = T1GetupConfig(
        sim_type=args.sim,
        num_envs=1,
        fallen_prob=0.0,
        fall_velocity_range=(0.0, 0.0),
        standing_z_offset=0.0,
    )
    cfgs = cfg.build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    rd = env.robot_data

    print(f"\n=== Config-level base_init_height ===")
    print(f"  cfg.robot.base_init_height = {cfg.robot.base_init_height}")
    print(f"  cfg.standing_z_offset      = {cfg.standing_z_offset}")
    print(f"  Expected root z after standing reset = {cfg.robot.base_init_height + cfg.standing_z_offset}")

    print(f"\n=== Right after env construction (before any explicit reset) ===")
    print(f"  root_pos_w = {_fmt(rd.root_link_pos_w)}")

    print(f"\n=== After env.reset() — expected: z=0.665 ===")
    env.reset()
    print(f"  root_pos_w = {_fmt(rd.root_link_pos_w)}")

    # Manually invoke the writer to force-set z=0.665 and see if it sticks.
    print(f"\n=== After manual writer.set_root_pose(pos=[0,0,0.665], quat=[1,0,0,0]) ===")
    writer = env.get_robot_state_writer("robot")
    env_ids = torch.arange(env.num_envs, device=env.device)
    pos = torch.tensor([[0.0, 0.0, 0.665]], device=env.device).expand(env.num_envs, -1).contiguous()
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).expand(env.num_envs, -1).contiguous()
    writer.set_root_pose(pos, quat, env_ids=env_ids)
    if hasattr(writer, "eval_fk"):
        writer.eval_fk(env_ids=env_ids)
    print(f"  root_pos_w = {_fmt(rd.root_link_pos_w)}")

    # Mjlab-specific: check env_origins, HOME_KEYFRAME qpos, xfrc_applied
    if args.sim == "mujoco":
        sm = env.scene_manager
        mj = sm.mj_model
        import mujoco
        print(f"\n=== Mjlab-specific introspection ===")
        # env_origins (from mjlab Scene)
        scene_obj = getattr(sm, "scene", None)
        if scene_obj is not None:
            origins = getattr(scene_obj, "env_origins", None)
            if origins is not None:
                print(f"  scene.env_origins[:1] = {_fmt(origins[:1])}")
            else:
                print("  scene.env_origins: None")
        # Home keyframe z from compiled model
        if mj.nkey > 0:
            for k in range(mj.nkey):
                kname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_KEY, k) or f"key_{k}"
                qpos_k = mj.key_qpos[k]
                # First 7 = root (xyz + xyzw quat)
                print(f"  keyframe[{k}] {kname!r}: root_z={qpos_k[2]:.4f} root_quat_xyzw={qpos_k[3:7]}")
        else:
            print("  no keyframes compiled")
        # Base body mass / geom inertia for sanity
        root_body = 1 if mj.body_parentid[1] == 0 else -1
        if root_body > 0:
            rname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, root_body) or "?"
            print(f"  root-body ({rname}) pos={mj.body_pos[root_body]} mass={mj.body_mass[root_body]}")


if __name__ == "__main__":
    main()
