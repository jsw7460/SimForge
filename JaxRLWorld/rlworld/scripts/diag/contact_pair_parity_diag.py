"""Do the three backends let the same geoms touch?

Not "is the friction the same" -- whether the same PAIRS collide at all.
A robot whose foot rests on four spheres in one simulator and on a flat
box in another is not carrying a parameter difference; it has a different
foot, and every reward, termination and gait that depends on the contact
patch is reading a different robot.

That is a real case, not a hypothetical. The K1 asset gives each foot a
shell box plus four spheres and masks the shell away from the ground with
``contype`` / ``conaffinity``. Those bits are file-local: MuJoCo compares
them freely because mjlab attaches the robot spec INTO the scene spec and
compiles one model, while Newton and Genesis both refuse to compare bits
across separately-imported entities -- correctly, in general. So the
shell lands on the ground on two backends out of three, and nothing
anywhere reports it.

**Comparing by link is not enough.** All three would say
``left_foot_link touches the ground`` and agree, while one of them means
the spheres and another means the box as well. The key here is therefore
``(body leaf name, geom type)``, which splits a foot link into its BOX
and its SPHEREs and needs no geom names -- Genesis discards those on MJCF
import, so any key that depends on them cannot cross the three.

Ground goes in as ``<ground>`` whatever a backend calls it (``terrain``,
``ground_plane_0``, ``plane_baselink``).

Usage:
    python -m rlworld.scripts.diag.contact_pair_parity_diag
    python -m rlworld.scripts.diag.contact_pair_parity_diag --robots k1
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections import Counter

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROBOTS: dict[str, tuple[str, str]] = {
    "g1": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "go2": ("rlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "k1": ("rlworld.rl.configs.presets.k1_joystick.base", "K1JoystickConfig"),
    "t1": ("rlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
}
SIMS = ("mujoco", "newton", "genesis")

NUM_ENVS = 16
SETTLE = 120

# A pair has to reach this many contact rows before its absence on another
# backend counts as a difference. A robot left under gravity with no action
# does not come to rest in a pose; it collapses into a heap, and in a heap
# every limb grazes every other. Those grazes are chaos, not robot identity:
# on T1 at 400 settling steps, 17 of 20 reported differences were a single
# row of one limb touching another, and they were a different 17 at 120
# steps. What survived both was structural and worth reading.
STRUCTURAL = 3

GROUND = "<ground>"
GROUND_HINTS = ("ground", "plane", "terrain")

# The two engines number their geom types differently -- Genesis has BOX at
# 5 and MuJoCo at 6 -- so both are mapped onto one vocabulary rather than
# compared as integers.
MUJOCO_TYPE = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}
GENESIS_TYPE = {
    0: "plane",
    1: "sphere",
    2: "ellipsoid",
    3: "cylinder",
    4: "capsule",
    5: "box",
    6: "mesh",
    7: "terrain",
}


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    wp_array = getattr(x, "wp_array", None)
    if wp_array is not None:
        return np.asarray(wp_array.numpy())
    for name in ("to_numpy", "numpy", "to_torch"):
        accessor = getattr(x, name, None)
        if accessor is not None:
            out = accessor()
            return _np(out) if isinstance(out, torch.Tensor) else np.asarray(out)
    return np.asarray(x)


def is_ground(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in GROUND_HINTS)


def leaf(body: str, vocab: set[str]) -> str:
    """One name for a body, whichever way a backend spells it.

    mjlab keeps the MJCF's name (sometimes as ``robot/name``), Genesis
    keeps the link's, and Newton flattens the whole ancestry with
    underscores -- ``K1_worldbody_Trunk_..._left_foot_link`` -- where no
    separator marks the body off from its parent. Splitting on "/" gets
    two of the three and silently gives the third a name of its own, so
    the same body counts as two different ones and every pair on that
    backend reads as missing everywhere else.

    So the first backend measured supplies the vocabulary and the rest
    match into it by longest suffix. mjlab goes first because its names
    are the MJCF's.
    """
    name = body.rsplit("/", 1)[-1]
    if name in vocab:
        return name
    matches = [known for known in vocab if name.endswith(known)]
    return max(matches, key=len) if matches else name


def key(body: str, geom_type: str, vocab: set[str]) -> str:
    name = leaf(body, vocab)
    if is_ground(name):
        return GROUND
    return f"{name}({geom_type})"


def build(robot: str, sim: str):
    from rlworld.rl.runners import BaseRunner

    mod_path, cls_name = ROBOTS[robot]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=NUM_ENVS).build()
    return BaseRunner._create_env_from_config(cfgs)


def mjwarp_pairs(env, sim: str, vocab: set[str]) -> tuple[Counter, set[str]]:
    import mujoco

    manager = env.scene_manager
    if sim == "newton":
        mj_model, data = manager.solver.mj_model, manager.solver.mjw_data
    else:
        mj_model, data = manager.mj_model, manager.data

    body_of = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, int(mj_model.geom_bodyid[g]))
        or f"<body {int(mj_model.geom_bodyid[g])}>"
        for g in range(mj_model.ngeom)
    ]
    keys = [
        GROUND
        if int(mj_model.geom_bodyid[g]) == 0
        else key(body_of[g], MUJOCO_TYPE.get(int(mj_model.geom_type[g]), "?"), vocab)
        for g in range(mj_model.ngeom)
    ]
    seen = {leaf(name, vocab) for name in body_of}

    nacon = int(_np(data.nacon)[0])
    out: Counter = Counter()
    if nacon == 0:
        return out, seen
    geom = _np(data.contact.geom)[:nacon]
    dist = _np(data.contact.dist)[:nacon]
    for row in range(nacon):
        if dist[row] >= 0.0:
            continue
        out[tuple(sorted((keys[int(geom[row, 0])], keys[int(geom[row, 1])])))] += 1
    return out, seen


def genesis_pairs(env, vocab: set[str]) -> tuple[Counter, set[str]]:
    from genesis.utils.misc import qd_to_torch

    solver = env.scene_manager.scene.sim.rigid_solver
    keys = [f"<geom {i}>" for i in range(solver.n_geoms)]
    for entity in solver.entities:
        for link in entity.links:
            for i, geom in enumerate(link.geoms):
                # geom.idx, NOT link.geom_start + i. They are two claims
                # about the same number and they do not always agree:
                # KinematicLink.geom_start returns 0 unconditionally, so a
                # link of that kind sends its contacts to whatever occupies
                # the low indices and reads back as the wrong body. Contacts
                # attributed to a neighbour do not go missing, they appear
                # somewhere else, which is worse than an obvious zero.
                if link.geom_start + i != int(geom.idx):
                    raise RuntimeError(
                        f"genesis geom indexing disagrees on {link.name}: "
                        f"geom_start {link.geom_start} + {i} != geom.idx {int(geom.idx)}"
                    )
                keys[int(geom.idx)] = key(link.name, GENESIS_TYPE.get(int(geom.type), "?"), vocab)
    seen = {leaf(link.name, vocab) for entity in solver.entities for link in entity.links}

    state = solver.collider._collider_state
    n_con = _np(qd_to_torch(state.n_contacts, copy=True))
    geom_a = _np(qd_to_torch(state.contact_data.geom_a, transpose=True, copy=True))
    geom_b = _np(qd_to_torch(state.contact_data.geom_b, transpose=True, copy=True))
    out: Counter = Counter()
    for world in range(geom_a.shape[0]):
        for row in range(int(n_con[world])):
            out[tuple(sorted((keys[int(geom_a[world, row])], keys[int(geom_b[world, row])])))] += 1
    return out, seen


def inventory(env, sim: str, vocab: set[str]) -> dict[tuple[str, str], tuple[int, set]]:
    """Every collidable geom the backend built: {(body, type): (count, sizes)}.

    A stronger question than "do the same things touch". The contact
    comparison sees only what happens to be in contact in the one pose it
    settles into; this is the collision geometry itself, pose-independent,
    and it is the signal that exposed T1 -- 25 collidable geoms on mjlab
    against 12 on Newton.

    Counted by (body, type), NOT by size. The engines store primitive
    parameters their own way and the difference is uniform, not per geom:
    Genesis keeps a box's FULL extents where MuJoCo keeps half (K1's foot
    box reads 0.18/0.07/0.036 against 0.09/0.035/0.018, exactly twice),
    and each pads the slots a shape does not use with something else
    again. Comparing on size therefore marks every geom in the robot as a
    difference and buries the real ones. The sizes are still carried and
    printed, because a size that differs where the others do not is worth
    seeing -- but they do not decide the verdict.
    """
    out: dict[tuple[str, str], tuple[int, set]] = {}

    def add(body: str, geom_type: str, size) -> None:
        # A PLANE is the ground by construction -- it is infinite, so
        # nothing else can be one. Going by the body name instead misses
        # it on Newton, which hangs the plane off the world body and so
        # calls it `world` where mjlab says `terrain` and Genesis
        # `plane_baselink`; the ground then reads as a geom two backends
        # built and the third did not.
        name = leaf(body, vocab)
        label = GROUND if geom_type == "plane" or is_ground(name) else name
        entry = out.setdefault((label, geom_type), (0, set()))
        out[(label, geom_type)] = (entry[0] + 1, entry[1] | {size})

    if sim == "genesis":
        solver = env.scene_manager.scene.sim.rigid_solver
        for entity in solver.entities:
            for link in entity.links:
                for geom in link.geoms:
                    add(
                        link.name,
                        GENESIS_TYPE.get(int(geom.type), "?"),
                        tuple(round(float(v), 4) for v in np.asarray(geom.data)[:3]),
                    )
        return out

    import mujoco

    manager = env.scene_manager
    mj_model = manager.solver.mj_model if sim == "newton" else manager.mj_model
    for g in range(mj_model.ngeom):
        if not (int(mj_model.geom_contype[g]) or int(mj_model.geom_conaffinity[g])):
            continue
        body = (
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, int(mj_model.geom_bodyid[g]))
            or f"<body {int(mj_model.geom_bodyid[g])}>"
        )
        add(
            body,
            MUJOCO_TYPE.get(int(mj_model.geom_type[g]), "?"),
            tuple(round(float(v), 4) for v in np.asarray(mj_model.geom_size[g])[:3]),
        )
    return out


def measure(robot: str, sim: str, vocab: set[str], settle: int) -> tuple[Counter, set[str], dict, dict]:
    env = build(robot, sim)
    zero = torch.zeros(NUM_ENVS, env.act_manager.num_actions, device=env.device)
    for _ in range(settle):
        env.step(zero)
    env._invalidate_cache()
    pairs, seen = genesis_pairs(env, vocab) if sim == "genesis" else mjwarp_pairs(env, sim, vocab)
    stock = inventory(env, sim, vocab or seen)

    # Where the robot ENDED UP, and whether it is still moving. Without
    # this a contact table cannot distinguish the two ways it can differ:
    # the backends settled into genuinely different poses, or they are
    # all still falling and were read mid-flight. A pair that appears on
    # one backend only means nothing until the speed column says the
    # robots had stopped.
    data = env.robot_data
    pose = {
        "height": float(data.root_link_pos_w[:, 2].mean()),
        "upright": float(-data.projected_gravity_b[:, 2].mean()),
        "speed": float(data.root_link_lin_vel_w.norm(dim=-1).mean()),
        "spin": float(data.root_link_ang_vel_w.norm(dim=-1).mean()),
    }
    del env
    return pairs, seen, stock, pose


def run(robot: str, sims: list[str], settle: int) -> list[str]:
    failures: list[str] = []
    marginal: list[str] = []
    print("=" * 92)
    print(f"  {robot.upper()}")
    print("=" * 92)

    found: dict[str, Counter] = {}
    pose: dict[str, dict] = {}
    stock: dict[str, dict] = {}
    vocab: set[str] = set()
    for sim in sims:
        found[sim], seen, stock[sim], pose[sim] = measure(robot, sim, vocab, settle)
        if not vocab:
            vocab = seen

    print("    COLLISION GEOMETRY — what each backend built, before anything moves")
    print(f"    {'body / type':<40}" + "".join(f"{s:>10}" for s in sims) + "   sizes")
    for item in sorted({g for counts in stock.values() for g in counts}):
        row = [stock[sim].get(item, (0, set()))[0] for sim in sims]
        sizes = " | ".join(",".join(str(v) for v in sorted(stock[sim].get(item, (0, set()))[1])) for sim in sims)
        mark = "" if len(set(row)) == 1 else "   <-- DIFFERS"
        label = f"{item[0]} {item[1]}"
        print(f"    {label[:38]:<40}" + "".join(f"{c:>10}" for c in row) + f"   {sizes[:70]}{mark}")
        if len(set(row)) != 1:
            failures.append(f"{robot}: geometry {label} counts {dict(zip(sims, row))}")
    totals = {sim: sum(c for c, _ in counts.values()) for sim, counts in stock.items()}
    print(f"    {'TOTAL collidable geoms':<40}" + "".join(f"{totals[s]:>10}" for s in sims))
    if len(set(totals.values())) != 1:
        failures.append(f"{robot}: collidable geom totals differ {totals}")
    print()

    print(f"    RESTING POSE after {settle} steps of zero action")
    print(f"      {'':<20}" + "".join(f"{s:>12}" for s in sims))
    for field, label in (
        ("height", "root height m"),
        ("upright", "upright cos"),
        ("speed", "root speed m/s"),
        ("spin", "root spin r/s"),
    ):
        print(f"      {label:<20}" + "".join(f"{pose[s][field]:>12.4f}" for s in sims))
    moving = [s for s in sims if pose[s]["speed"] > 0.05 or pose[s]["spin"] > 0.5]
    if moving:
        print(f"      still moving on {', '.join(moving)} — the contact table below is a snapshot,")
        print("      not a resting state; re-run with a larger --settle before reading it")
    print()

    print("    CONTACT — what is actually touching after it settles")
    every = sorted({pair for counts in found.values() for pair in counts})
    print(f"    {'pair':<58}" + "".join(f"{s:>11}" for s in sims))
    for pair in every:
        row = [found[sim].get(pair, 0) for sim in sims]
        touched = [count > 0 for count in row]
        mark = "" if all(touched) or not any(touched) else "   <-- ONLY SOME"
        label = f"{pair[0]} + {pair[1]}"
        print(f"    {label[:56]:<58}" + "".join(f"{c:>11}" for c in row) + mark)
        if not all(touched):
            missing = [s for s, t in zip(sims, touched) if not t]
            line = f"{robot}: {label} does not exist on {', '.join(missing)}"
            (failures if max(row) >= STRUCTURAL else marginal).append(line)

    print(
        f"\n    {len(every)} distinct pairs; "
        f"{sum(1 for p in every if all(found[s].get(p, 0) for s in sims))} on every backend"
    )
    if marginal:
        print(f"    {len(marginal)} more differ by fewer than {STRUCTURAL} rows and are not counted:")
        for line in marginal:
            print(f"      {line.split(': ', 1)[1]}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=sorted(ROBOTS), choices=sorted(ROBOTS))
    ap.add_argument("--sims", nargs="+", default=list(SIMS), choices=list(SIMS))
    ap.add_argument("--settle", type=int, default=SETTLE, help="settling steps before reading")
    args = ap.parse_args()

    print("=" * 92)
    print("  DO THE THREE BACKENDS LET THE SAME GEOMS TOUCH")
    print("=" * 92)
    print(f"  {NUM_ENVS} envs, {args.settle} settling steps, zero action")
    print("  keyed by (body leaf, geom type) — a link alone cannot tell a foot's")
    print("  spheres from the box beside them, and that is where they differ")

    failures: list[str] = []
    for robot in args.robots:
        failures += run(robot, args.sims, args.settle)

    print("\n" + "=" * 92)
    if failures:
        print(f"  {len(failures)} DISAGREEMENTS")
        for line in failures:
            print(f"    {line}")
        print("  A geom one backend built and another did not, or a pair that")
        print("  exists on one and not another, is a different robot -- not a")
        print("  different parameter.")
    else:
        print("  same collision geometry, and the same pairs of it touching")
    print("=" * 92)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
