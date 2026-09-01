"""Genesis never reports T1's trunk or waist touching the ground. Why?

``contact_pair_parity_diag`` leaves T1 with one difference that survives
every settle length tried:

    <ground> + Trunk(mesh)     mujoco 7   newton 7   genesis 0
    <ground> + Waist(sphere)   mujoco 10  newton 9   genesis 0

The obvious reading -- Genesis settles into a different pose -- does not
survive the numbers beside it. Newton and Genesis come to rest at almost
the same orientation and almost the same height, Genesis the LOWER of
the two, and Newton reports seven rows where Genesis reports none.

So this asks about DETECTION by taking the pose out of the question
entirely: settle one backend, capture the exact state it reached, write
that state into all three, and read what each says is touching. Same
root pose, same joint angles, same geometry, one step each. Whatever
differs is not where the robot ended up.

**Do not do this by driving the robot through the floor.** The first
version of this diagnostic swept the root height down past the ground,
and every backend hit a buffer limit -- mjwarp printed `nefc overflow`
and Genesis silently exhausted `max_collision_pairs` -- so all three
under-reported and even the feet read zero. A buried robot generates
hundreds of contacts and measures the buffers, not the collider. An
injected settled state generates the normal number.

Usage:
    python -m jaxrlworld.scripts.diag.t1.t1_trunk_ground_contact_diag
    python -m jaxrlworld.scripts.diag.t1.t1_trunk_ground_contact_diag --source newton
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import torch  # noqa: E402

from jaxrlworld.scripts.diag.parity.contact_pair_parity_diag import (  # noqa: E402
    GROUND,
    NUM_ENVS,
    ROBOTS,
    SETTLE,
    build,
    genesis_pairs,
    mjwarp_pairs,
)

SIMS = ("mujoco", "newton", "genesis")


def ground_contacts(env, sim: str, vocab: set[str]) -> tuple[dict[str, int], set[str], int]:
    env._invalidate_cache()
    pairs, seen = genesis_pairs(env, vocab) if sim == "genesis" else mjwarp_pairs(env, sim, vocab)
    out: dict[str, int] = {}
    for (a, b), count in pairs.items():
        if a == GROUND or b == GROUND:
            other = b if a == GROUND else a
            out[other] = out.get(other, 0) + count
    return out, seen, sum(pairs.values())


def settle(env, steps: int) -> None:
    zero = torch.zeros(NUM_ENVS, env.act_manager.num_actions, device=env.device)
    for _ in range(steps):
        env.step(zero)


def capture(env) -> dict[str, torch.Tensor]:
    data = env.robot_data
    return {
        "pos": data.root_link_pos_w.detach().clone(),
        "quat": data.root_link_quat_w.detach().clone(),
        "joints": data.joint_pos.detach().clone(),
    }


def inject(env, sim: str, state: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """Put the robot where the reference backend left it; report both errors.

    Two different questions, and conflating them wastes a run. The FIRST
    readback is taken before stepping and asks whether the write landed
    at all. The SECOND is taken after the single step the collider needs
    and is pure drift: one dt of gravity and of a PD controller pulling
    toward its default target moves the joints by tens of milliradians on
    every backend, the reference one included. The first version of this
    check only looked after the step and declared the write broken
    everywhere, including on the backend the state was copied FROM.
    """
    writer = env.get_robot_state_writer("robot")
    device = env.device
    env_ids = torch.arange(NUM_ENVS, device=device)
    zeros3 = torch.zeros(NUM_ENVS, 3, device=device)
    writer.set_root_pose(state["pos"].to(device), state["quat"].to(device), env_ids=env_ids)
    writer.set_root_velocity(zeros3, zeros3.clone(), env_ids=env_ids)
    writer.set_dof_positions(state["joints"].to(device), env_ids=env_ids)
    writer.set_dof_velocities(torch.zeros_like(state["joints"]).to(device), env_ids=env_ids)

    # mjlab's writer leaves the derived state stale: its eval_fk is a
    # no-op, so nothing downstream of qpos updates until something asks
    # the model to. Newton's eval_fk does the work and Genesis recomputes
    # on read, so only this backend needs the nudge.
    if sim == "mujoco":
        env.scene_manager.forward()
    env._invalidate_cache()
    landed = readback(env, state)

    # One step, because the collider only runs inside one.
    env.step(torch.zeros(NUM_ENVS, env.act_manager.num_actions, device=device))
    env._invalidate_cache()
    return landed, readback(env, state)


def readback(env, state: dict[str, torch.Tensor]) -> dict[str, float]:
    """Did the write actually land? Ask before comparing anything else.

    A writer that silently does nothing turns this whole diagnostic into
    a comparison of each backend against its own settled pose, which is
    exactly the confound it exists to remove. Genesis has had a writer
    quietly drop part of a reset before, so the state goes in and comes
    straight back out before a single number is believed.
    """
    data = env.robot_data
    device = env.device
    return {
        "pos": float((data.root_link_pos_w - state["pos"].to(device)).abs().max()),
        "quat": float((data.root_link_quat_w.abs() - state["quat"].to(device).abs()).abs().max()),
        "joints": float((data.joint_pos - state["joints"].to(device)).abs().max()),
    }


def run(robot: str, sims: list[str], source: str, steps: int) -> list[str]:
    print("=" * 100)
    print(f"  {robot.upper()} — the same state, read on every backend")
    print("=" * 100)

    env = build(robot, source)
    settle(env, steps)
    state = capture(env)
    reference, vocab, _ = ground_contacts(env, source, set())
    del env
    print(f"    state captured from {source} after {steps} steps")
    print(f"    root height {state['pos'][:, 2].mean():.4f} m, {len(reference)} bodies on the ground there\n")

    readings: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    landed: dict[str, dict[str, float]] = {}
    drift: dict[str, dict[str, float]] = {}
    for sim in sims:
        env = build(robot, sim)
        settle(env, 1)  # let the scene finish initialising before writing to it
        landed[sim], drift[sim] = inject(env, sim, state)
        readings[sim], _, totals[sim] = ground_contacts(env, sim, vocab)
        del env

    TOL = {"pos": 1e-5, "quat": 1e-5, "joints": 1e-5}
    print("    DID THE WRITE LAND — max |written - read back|, BEFORE stepping")
    print(f"      {'':<20}" + "".join(f"{s:>14}" for s in sims))
    for field in TOL:
        print(f"      {field:<20}" + "".join(f"{landed[s][field]:>14.6f}" for s in sims))
    stale = [s for s in sims if any(landed[s][f] > t for f, t in TOL.items())]
    if stale:
        print(f"      the write did NOT land on {', '.join(stale)} — every number below is void")
        failures_early = [f"{robot}: the injected state did not land on {s}" for s in stale]
    else:
        failures_early = []

    print("\n    DRIFT after the one step the collider needs (all backends drift; compare them)")
    print(f"      {'':<20}" + "".join(f"{s:>14}" for s in sims))
    for field in TOL:
        print(f"      {field:<20}" + "".join(f"{drift[s][field]:>14.6f}" for s in sims))
    print()

    failures: list[str] = list(failures_early)
    bodies = sorted({name for r in readings.values() for name in r})
    print(f"    {'body / geom type':<40}" + "".join(f"{s:>14}" for s in sims))
    for name in bodies:
        row = [readings[sim].get(name, 0) for sim in sims]
        touched = [c > 0 for c in row]
        mark = "" if all(touched) or not any(touched) else "   <-- ONLY SOME"
        print(f"    {name[:38]:<40}" + "".join(f"{c:>14}" for c in row) + mark)
        if not all(touched):
            missing = [s for s, t in zip(sims, touched) if not t]
            failures.append(f"{robot}: {name} reports no ground contact on {', '.join(missing)}")
    print(f"    {'TOTAL contact rows (all pairs)':<40}" + "".join(f"{totals[s]:>14}" for s in sims))
    print("    a total near a backend's contact budget means it dropped rows; compare, do not trust")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=["t1"], choices=sorted(ROBOTS))
    ap.add_argument("--sims", nargs="+", default=list(SIMS), choices=list(SIMS))
    ap.add_argument("--source", default="mujoco", choices=list(SIMS), help="backend the state is taken from")
    ap.add_argument("--settle", type=int, default=SETTLE)
    args = ap.parse_args()

    print("=" * 100)
    print("  DETECTION, NOT POSE")
    print("=" * 100)
    print(f"  {NUM_ENVS} envs; one backend settles, every backend is put in that exact state")

    failures: list[str] = []
    for robot in args.robots:
        failures += run(robot, args.sims, args.source, args.settle)

    print("\n" + "=" * 100)
    if failures:
        print(f"  {len(failures)} BODIES DISAGREE AT AN IDENTICAL STATE")
        for line in failures:
            print(f"    {line}")
        print("  Same pose, same geometry, different answer: something is refusing")
        print("  the pair rather than missing it.")
    else:
        print("  at one identical state, every backend reports the same bodies on the ground")
    print("=" * 100)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
