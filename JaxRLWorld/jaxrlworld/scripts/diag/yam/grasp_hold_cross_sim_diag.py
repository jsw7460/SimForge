"""Does the same grasp hold the cube the same way on every backend?

The contact parameter diag shows the backends give the finger pads and
the cube the same friction coefficients. A coefficient is not a grip:
how far a squeezed cube creeps is set by the contact model as a whole
(soft constraints and the friction cone on the MuJoCo family, a rigid
solver with its own friction model elsewhere). This puts the cube in
the same grasp on every backend and measures how it moves in the hand.

One scripted episode, identical on every backend, ``--num-envs`` copies
that differ only in how hard the arm is shaken:

1. **open**: the arm at its home pose, the gripper fully open.
2. **close**: the cube is placed at the grasp site, axis-aligned with it,
   and held there by rewriting its pose every step while the gripper
   closes onto it.
3. **hold**: the cube is released; the arm stays still.
4. **shake**: the base yaw and the wrist pitch swing sinusoidally, with
   an amplitude that grows from the first env to the last.

Recorded every control step: the cube's position in the grasp site's
frame and the left finger's angle. Reported per backend: the creep
during the hold, the largest displacement during the shake, and how
many envs lost the cube (moved over ``--lost-mm`` from where it was
released, or ended the episode).

Each backend runs in its own process and saves its trace; ``--compare``
reads them all::

    jaxpy -m jaxrlworld.scripts.diag.yam.grasp_hold_cross_sim_diag --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.yam.grasp_hold_cross_sim_diag --sim newton
    jaxpy -m jaxrlworld.scripts.diag.yam.grasp_hold_cross_sim_diag --sim genesis
    jaxpy -m jaxrlworld.scripts.diag.yam.grasp_hold_cross_sim_diag --compare

``--sim`` takes any registered backend, plugins included.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

from jaxrlworld.rl.configs.presets.yam_lift.base import CUBE, GRASP_SITE, YamLiftConfig
from jaxrlworld.rl.configs.robots.yam import YAM_ACTION_SCALE, YAM_GRIPPER_JOINT
from jaxrlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.rl.utils.quat_utils import quat_inv_wxyz, quat_mul_wxyz, quat_rotate_wxyz

OUT = Path("outputs/grasp_hold_cross_sim")

# Gripper targets: the joint's limits in the MJCF (left_finger range).
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = -0.00205

# Phase lengths in control steps (0.02 s each).
OPEN_STEPS = 50
CLOSE_STEPS = 100
HOLD_STEPS = 100
SHAKE_STEPS = 150
SHAKE_HZ = 1.0
SHAKE_JOINTS = ("joint1", "joint4")


def _build(sim: str, num_envs: int):
    preset = YamLiftConfig(sim_type=sim, num_envs=num_envs, end_on_success=False)
    preset.reset_joint_position_noise = (0.0, 0.0)
    preset.episode_length_s = 60.0
    cfgs = preset.build()
    # The cube is placed by this script alone: the command's timer must not
    # move it mid-grasp.
    cfgs.command.terms["lift"].place_object_on_resample = False
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return env, preset


def _actions(env, preset, targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """Joint position targets -> the action that commands them (target = offset + scale * action)."""
    offset = preset.robot.get_action_offset()
    names = list(env.act_manager.actuated_joint_names)
    columns = []
    for name in names:
        target = targets.get(name, torch.full((env.num_envs,), offset[name], device=env.device))
        columns.append((target - offset[name]) / YAM_ACTION_SCALE[name])
    return torch.stack(columns, dim=-1)


def record(args) -> int:
    env, preset = _build(args.sim, args.num_envs)
    n = env.num_envs
    device = env.device
    grasp = env.resolve_selector(SceneEntitySelector(name="robot", site_names=(GRASP_SITE,)))
    robot = env.get_entity_data("robot")
    cube = env.get_entity_data(CUBE)
    cube_writer = env.get_root_state_writer(CUBE)
    finger = list(env.act_manager.actuated_joint_names).index(YAM_GRIPPER_JOINT)
    home = preset.robot.get_action_offset()
    zeros3 = torch.zeros(n, 3, device=device)
    amplitude = torch.linspace(0.1, 0.8, n, device=device)

    def site_pose() -> tuple[torch.Tensor, torch.Tensor]:
        env._invalidate_cache()
        return robot.site_pos_w_by_ids(grasp.site_ids)[:, 0], robot.site_quat_w_by_ids(grasp.site_ids)[:, 0]

    def pin_cube_at_site():
        pos, quat = site_pose()
        cube_writer.set_root_pose(pos, quat)
        cube_writer.set_root_velocity(zeros3, zeros3)
        cube_writer.eval_fk()
        env._post_reset_forward()
        env._invalidate_cache()

    def cube_in_site_frame() -> tuple[torch.Tensor, torch.Tensor]:
        pos, quat = site_pose()
        inv = quat_inv_wxyz(quat)
        return quat_rotate_wxyz(inv, cube.root_link_pos_w - pos), quat_mul_wxyz(inv, cube.root_link_quat_w)

    gripper = {YAM_GRIPPER_JOINT: torch.full((n,), GRIPPER_OPEN, device=device)}
    for _ in range(OPEN_STEPS):
        env.step(_actions(env, preset, gripper))

    gripper = {YAM_GRIPPER_JOINT: torch.full((n,), GRIPPER_CLOSED, device=device)}
    for _ in range(CLOSE_STEPS):
        pin_cube_at_site()
        env.step(_actions(env, preset, gripper))

    trace, rotation, finger_q, finger_force = [], [], [], []
    ended = torch.zeros(n, dtype=torch.bool, device=device)
    for step in range(HOLD_STEPS + SHAKE_STEPS):
        targets = dict(gripper)
        if step >= HOLD_STEPS:
            phase = 2.0 * math.pi * SHAKE_HZ * (step - HOLD_STEPS) * env.control_dt
            for joint in SHAKE_JOINTS:
                targets[joint] = home[joint] + amplitude * math.sin(phase)
        env.step(_actions(env, preset, targets))
        ended |= env.termination_manager.terminated
        pos, quat = cube_in_site_frame()
        trace.append(pos.cpu().numpy())
        rotation.append(quat.cpu().numpy())
        finger_q.append(robot.joint_pos[:, finger].cpu().numpy())
        finger_force.append(robot.applied_torque[:, finger].cpu().numpy())

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.sim}.npz"
    np.savez_compressed(
        path,
        cube_in_site=np.stack(trace),
        cube_quat_in_site=np.stack(rotation),
        finger_q=np.stack(finger_q),
        finger_force=np.stack(finger_force),
        ended=ended.cpu().numpy(),
        amplitude=amplitude.cpu().numpy(),
        control_dt=env.control_dt,
    )
    print(f"saved {path}: {len(trace)} steps x {n} envs")
    return 0


def compare(args) -> int:
    paths = sorted(OUT.glob("*.npz"))
    if not paths:
        print(f"no traces under {OUT}")
        return 1
    print(
        f"{'backend':<10} {'finger q':>9} {'squeeze N':>10} {'hold creep mm (median, max)':>28} "
        f"{'hold rot deg':>13} {'shake max mm':>13} {'shake rot deg':>14} {'lost':>6} {'ended':>6}"
    )
    per_env = {}
    for path in paths:
        data = np.load(path)
        trace = data["cube_in_site"]  # (T, N, 3)
        moved = np.linalg.norm(trace - trace[0], axis=-1) * 1000.0  # (T, N) mm
        # Rotation since release: angle of q0^-1 * q, from the vector part (stable near zero).
        quat = data["cube_quat_in_site"]  # (T, N, 4) wxyz
        w0, v0 = quat[0, :, :1], quat[0, :, 1:]
        w, v = quat[..., :1], quat[..., 1:]
        rel_w = w0 * w + (v0 * v).sum(-1, keepdims=True)
        rel_v = w0 * v - w * v0 - np.cross(v0, v)
        turned = np.degrees(2.0 * np.arctan2(np.linalg.norm(rel_v, axis=-1), np.abs(rel_w[..., 0])))  # (T, N)
        hold, shake = moved[:HOLD_STEPS], moved[HOLD_STEPS:]
        lost = (moved.max(axis=0) > args.lost_mm) | data["ended"]
        print(
            f"{path.stem:<10} {float(np.median(data['finger_q'][HOLD_STEPS - 1])):>9.4f} "
            f"{float(np.median(data['finger_force'][:HOLD_STEPS])):>10.3f} "
            f"{float(np.median(hold[-1])):>18.2f}, {float(hold[-1].max()):>7.2f} "
            f"{float(np.median(turned[HOLD_STEPS - 1])):>13.2f} "
            f"{float(np.median(shake.max(axis=0))):>13.2f} {float(np.median(turned[HOLD_STEPS:].max(axis=0))):>14.2f} "
            f"{int(lost.sum()):>3}/{lost.size:<2} {int(data['ended'].sum()):>3}/{lost.size:<2}"
        )
        per_env[path.stem] = (shake.max(axis=0), turned[HOLD_STEPS:].max(axis=0))
    print("\nper env, shake max displacement mm / rotation deg:")
    for name, (disp, rot) in per_env.items():
        print(f"  {name:<10} " + "  ".join(f"{d:6.1f}/{r:5.1f}" for d, r in zip(disp, rot)))
    print(f"\nshake amplitude per env (rad): {np.round(np.load(paths[0])['amplitude'], 2).tolist()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", help="backend to record")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--lost-mm", type=float, default=20.0, help="displacement in the hand that counts as lost")
    args = ap.parse_args()
    if args.compare:
        return compare(args)
    if args.sim is None:
        ap.error("--sim or --compare")
    return record(args)


if __name__ == "__main__":
    sys.exit(main())
