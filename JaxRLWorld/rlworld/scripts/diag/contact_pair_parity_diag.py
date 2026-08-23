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
                keys[link.geom_start + i] = key(link.name, GENESIS_TYPE.get(int(geom.type), "?"), vocab)
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


def measure(robot: str, sim: str, vocab: set[str]) -> tuple[Counter, set[str]]:
    env = build(robot, sim)
    zero = torch.zeros(NUM_ENVS, env.act_manager.num_actions, device=env.device)
    for _ in range(SETTLE):
        env.step(zero)
    env._invalidate_cache()
    pairs, seen = genesis_pairs(env, vocab) if sim == "genesis" else mjwarp_pairs(env, sim, vocab)
    del env
    return pairs, seen


def run(robot: str, sims: list[str]) -> list[str]:
    failures: list[str] = []
    print("=" * 92)
    print(f"  {robot.upper()}")
    print("=" * 92)

    found: dict[str, Counter] = {}
    vocab: set[str] = set()
    for sim in sims:
        found[sim], seen = measure(robot, sim, vocab)
        if not vocab:
            vocab = seen
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
            failures.append(f"{robot}: {label} does not exist on {', '.join(missing)}")

    print(
        f"\n    {len(every)} distinct pairs; "
        f"{sum(1 for p in every if all(found[s].get(p, 0) for s in sims))} on every backend"
    )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=sorted(ROBOTS), choices=sorted(ROBOTS))
    ap.add_argument("--sims", nargs="+", default=list(SIMS), choices=list(SIMS))
    args = ap.parse_args()

    print("=" * 92)
    print("  DO THE THREE BACKENDS LET THE SAME GEOMS TOUCH")
    print("=" * 92)
    print(f"  {NUM_ENVS} envs, {SETTLE} settling steps, zero action")
    print("  keyed by (body leaf, geom type) — a link alone cannot tell a foot's")
    print("  spheres from the box beside them, and that is where they differ")

    failures: list[str] = []
    for robot in args.robots:
        failures += run(robot, args.sims)

    print("\n" + "=" * 92)
    if failures:
        print(f"  {len(failures)} PAIRS DISAGREE")
        for line in failures:
            print(f"    {line}")
        print("  A pair that exists on one backend and not another is a different")
        print("  robot, not a different parameter.")
    else:
        print("  every contacting pair exists on every backend")
    print("=" * 92)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
