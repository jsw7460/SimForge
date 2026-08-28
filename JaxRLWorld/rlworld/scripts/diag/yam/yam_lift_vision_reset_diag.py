"""Why does the arm pick the cube after a manual reset but not later?

Watching a replay, the arm grasps reliably from a reset and then, some
time in, stops managing it. What LOOKS like an automatic reset need not
be one: this task's command term resamples on a timer, and resampling
does not only move the goal — it teleports the cube to a fresh spot,
taking it out of the gripper if that is where it was. The arm is left
wherever it stood, which a real reset would never do.

So rather than guess, this narrates the episode. Every step it reports
what actually happened:

* which termination term fired, per env, by name
* when the command resampled, and how far the cube jumped as a result
* whether the cube is up off the table, i.e. whether the arm is holding it
* how long after each event the arm next gets the cube up

Run::

    jaxpy -m rlworld.scripts.diag.yam.yam_lift_vision_reset_diag \\
        --policy_path outputs/models/<date>/<run>/checkpoint_latest/
    jaxpy -m rlworld.scripts.diag.yam.yam_lift_vision_reset_diag \\
        --policy_path ... --eval_sim newton --steps 2000
"""

from __future__ import annotations

import argparse

import torch

from rlworld.rl.configs.presets.yam_arm.base import CUBE_HALF, TABLE_TOP_Z
from rlworld.rl.configs.presets.yam_lift.vision import CAMERA_SENSOR
from rlworld.rl.configs.sensors.camera_sensor_config import resolve_mjcf_camera
from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.utils.quat_utils import quat_inv_wxyz, quat_mul_wxyz, quat_rotate_wxyz

_HELD_MARGIN = 0.03
"""Metres above its resting height before a cube counts as picked up."""


def _cube_in_view(env, placement, fovy_deg: float, cube_pos_w: torch.Tensor) -> torch.Tensor:
    """Is the cube inside the wrist camera's frustum this step?

    A policy that cannot see the cube has no information about it at
    all, so "it stopped working" and "it stopped looking at anything"
    are the same sentence. Computed from the camera's pose rather than
    from the image: it needs no per-shape channel and so works on every
    backend.
    """
    robot_data = env.get_robot_data("robot")
    index = robot_data.find_body_index(placement.body)
    link_pos = robot_data.body_pos_w_all[:, index]
    link_quat = robot_data.body_quat_w_all[:, index]

    offset_pos = torch.tensor(placement.pos, dtype=torch.float32, device=link_pos.device)
    offset_quat = torch.tensor(placement.quat, dtype=torch.float32, device=link_pos.device)
    camera_pos = link_pos + quat_rotate_wxyz(link_quat, offset_pos.expand_as(link_pos))
    camera_quat = quat_mul_wxyz(link_quat, offset_quat.expand_as(link_quat))

    # Into the camera's own frame, where it looks down -Z.
    local = quat_rotate_wxyz(quat_inv_wxyz(camera_quat), cube_pos_w - camera_pos)
    forward = -local[:, 2]
    half = torch.tan(torch.deg2rad(torch.tensor(fovy_deg, device=local.device)) * 0.5)
    # The image is square and mjwarp crops the sensor to match, so the
    # horizontal half-angle equals the vertical one.
    return (forward > 0) & (local[:, 0].abs() <= half * forward) & (local[:, 1].abs() <= half * forward)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--eval_sim", default="mujoco", choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="Override the episode length. The eval scripts set it to 1e10, which removes the "
        "time-based reset the policy was trained with; pass 20 to restore the training value.",
    )
    args = ap.parse_args()

    overrides: dict = {"env": {"num_envs": args.num_envs}}
    if args.episode_length_s is not None:
        overrides["env"]["episode_length_s"] = args.episode_length_s

    evaluator = PolicyEvaluator(
        policy_path=args.policy_path,
        eval_target=args.eval_sim,
        extra_overrides=overrides,
    )
    env = evaluator.env
    policy = evaluator.policy

    command = env.command_manager.get_term("lift")
    resting_z = TABLE_TOP_Z + CUBE_HALF

    print("=" * 78)
    print(f"YAM LIFT VISION: what happens between the grasps  [{args.eval_sim}, {args.num_envs} envs]")
    print("=" * 78)
    print(f"  episode_length_s        {env.env_cfg.episode_length_s}")
    print(f"  command resampling      {command.cfg.resampling_time_range} s")
    print(f"  command places the cube {command.cfg.place_object}")
    print(f"  difficulty              {command.cfg.difficulty}")
    print("  (a resample moves the goal AND, when place_object is set, teleports the cube)")
    print()

    mjcf_path = env.scene_manager.config.entities["robot"].mjcf_path
    placement = resolve_mjcf_camera(mjcf_path, CAMERA_SENSOR)
    print(f"  camera rides {placement.body!r}, fovy {placement.fovy:.2f} deg")
    print()

    obs = env.obs_manager.get_observation()
    robot_states = None

    previous_cube = command.object_pos_w.clone()
    previous_target = command.target_pos.clone()
    held_before = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Step at which each env last had its world rearranged, and by what.
    last_event_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    last_event_kind = ["start"] * env.num_envs
    grasp_delays: dict[str, list[int]] = {}
    # Whether the arm ever got the cube up again before the NEXT event.
    # Counting only the times it did would report the median of the
    # successes and hide every time it never recovered at all.
    outcomes: dict[str, list[bool]] = {}
    recovered = [True] * env.num_envs
    in_view_since_event = [[] for _ in range(env.num_envs)]
    view_when_stuck: dict[str, list[float]] = {}

    for step in range(args.steps):
        action = policy.get_action(obs, robot_states)
        obs, _, terminated, truncated, infos = env.step(action)

        cube = command.object_pos_w
        target = command.target_pos
        jump = (cube - previous_cube).norm(dim=-1)
        target_moved = (target - previous_target).norm(dim=-1) > 1e-6
        held = cube[:, 2] > resting_z + _HELD_MARGIN
        in_view = _cube_in_view(env, placement, placement.fovy, cube)

        dones = terminated | truncated
        term_dones = env.termination_manager.term_dones

        for env_index in range(env.num_envs):
            # A reset is announced by the termination manager and names
            # itself; a resample announces nothing, so it is recognised by
            # the cube moving further in one step than physics allows.
            fired = [name for name, mask in term_dones.items() if bool(mask[env_index])]
            teleported = bool(jump[env_index] > 0.05) and not bool(dones[env_index])

            if fired:
                kind = "reset:" + ",".join(fired)
            elif teleported:
                kind = "resample" + (" (+goal moved)" if bool(target_moved[env_index]) else "")
            else:
                kind = None

            if kind is not None:
                was_held = bool(held_before[env_index])
                # Close the books on the previous event before opening
                # this one, so a stretch that never produced a grasp is
                # counted as a failure rather than left out.
                previous = last_event_kind[env_index]
                outcomes.setdefault(previous, []).append(recovered[env_index])
                if not recovered[env_index]:
                    seen = in_view_since_event[env_index]
                    share = sum(seen) / max(len(seen), 1)
                    view_when_stuck.setdefault(previous, []).append(share)
                    print(
                        f"  step {step:5d} env {env_index}: never recovered from the last {previous} "
                        f"({step - int(last_event_step[env_index])} steps), "
                        f"cube in view {100.0 * share:.1f}% of them"
                    )
                print(
                    f"  step {step:5d} env {env_index}: {kind:<28} "
                    f"cube jumped {float(jump[env_index]):.3f} m   "
                    f"{'WAS HOLDING IT' if was_held else 'cube was down'}"
                )
                last_event_step[env_index] = step
                last_event_kind[env_index] = kind.split(":")[0].split(" ")[0]
                recovered[env_index] = False
                in_view_since_event[env_index] = []

            in_view_since_event[env_index].append(bool(in_view[env_index]))

            # First time the cube comes up after an event: how long it took.
            if bool(held[env_index]) and not bool(held_before[env_index]) and not recovered[env_index]:
                delay = step - int(last_event_step[env_index])
                grasp_delays.setdefault(last_event_kind[env_index], []).append(delay)
                recovered[env_index] = True
                print(
                    f"  step {step:5d} env {env_index}: PICKED UP, {delay} steps after the last {last_event_kind[env_index]}"
                )

        previous_cube = cube.clone()
        previous_target = target.clone()
        held_before = held

    print("\n" + "=" * 78)
    print("  did the arm get the cube back, and how long did it take")
    print(f"  {'after':<12} {'recovered':>12} {'median steps':>13} {'worst':>7}")
    for kind in sorted(outcomes):
        results = outcomes[kind]
        rate = sum(results) / max(len(results), 1)
        delays = grasp_delays.get(kind, [])
        if delays:
            values = torch.tensor(delays, dtype=torch.float32)
            timing = f"{float(values.median()):13.1f} {int(values.max()):7d}"
        else:
            timing = f"{'-':>13} {'-':>7}"
        print(f"  {kind:<12} {sum(results):4d}/{len(results):<3d} {100.0 * rate:5.0f}% {timing}")

    print("\n  when it did NOT recover, how much of that time was the cube even in frame")
    if not view_when_stuck:
        print("    it always recovered")
    for kind, shares in sorted(view_when_stuck.items()):
        values = torch.tensor(shares, dtype=torch.float32)
        print(f"    after {kind:<10} n={len(shares):3d}  median {100.0 * float(values.median()):5.1f}%")
    print("  A policy that cannot see the cube has no information about it, so a low share")
    print("  here means the arm is not failing to solve the task - it is failing to look.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
