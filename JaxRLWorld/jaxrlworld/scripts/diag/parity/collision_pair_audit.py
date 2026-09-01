"""How many geom pairs does each backend actually test for contact?

K1 steps 2.0x slower on Genesis than on mjlab, while go2 — the same
kind of scene, tuned on both — is only 1.4x. The gap widens with the
robot, which points at something that scales with the number of links
rather than at a solver setting; the four solver and collision knobs
that differ between the two K1 builders all measured null.

Contact filtering is the candidate that scales that way. Both engines
claim to honour the MJCF's ``contype`` / ``conaffinity`` bitmasks, and
the Genesis builder's own docstring relies on it: it turns
``enable_self_collision`` on so the feet can touch each other, and
expects the masks to suppress every other intra-robot pair. If Genesis
ends up testing pairs mjlab excludes, that is not a speed setting to
turn off — it is the two backends simulating different robots, and the
speed is a symptom.

So both are counted, from the built model rather than from the config.
The MuJoCo side reimplements MuJoCo's own filter: same weld, welded
parent-child, an explicit ``<exclude>``, then the bitmask test.

    jaxpy -m jaxrlworld.scripts.diag.parity.collision_pair_audit --preset k1_joystick
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from jaxrlworld.rl.configs.presets.go2.base import Go2FlatConfig
from jaxrlworld.rl.configs.presets.k1_joystick.base import K1JoystickConfig
from jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
from jaxrlworld.rl.configs.presets.yam_lift.base import YamLiftConfig
from jaxrlworld.rl.runners import BaseRunner

_PRESETS = {
    "go2": Go2FlatConfig,
    "k1_joystick": K1JoystickConfig,
    # The config the K1 training scripts actually run. Its scene and
    # timing are inherited unchanged, so physics numbers match
    # k1_joystick; the policy, reward and DR cadence differ.
    "k1_g1_recipe": K1G1RecipeConfig,
    "yam_lift": YamLiftConfig,
}


def _mujoco_pairs(mj_model) -> tuple[int, int]:
    """(collision geoms, pairs surviving MuJoCo's filter) for one env."""
    contype = mj_model.geom_contype
    conaffinity = mj_model.geom_conaffinity
    bodyid = mj_model.geom_bodyid
    weldid = mj_model.body_weldid
    parentid = mj_model.body_parentid

    excluded = set()
    for i in range(mj_model.nexclude):
        signature = int(mj_model.exclude_signature[i])
        excluded.add((signature >> 16, signature & 0xFFFF))

    collidable = [g for g in range(mj_model.ngeom) if contype[g] or conaffinity[g]]
    pairs = 0
    for g1, g2 in itertools.combinations(collidable, 2):
        b1, b2 = int(bodyid[g1]), int(bodyid[g2])
        w1, w2 = int(weldid[b1]), int(weldid[b2])
        if w1 == w2:
            continue
        # A welded parent-child pair is filtered unless FILTERPARENT is off,
        # which no preset here disables.
        if int(parentid[w1]) == w2 or int(parentid[w2]) == w1:
            continue
        if (min(b1, b2), max(b1, b2)) in excluded:
            continue
        if (contype[g1] & conaffinity[g2]) or (contype[g2] & conaffinity[g1]):
            pairs += 1
    return len(collidable), pairs


def run_single(preset: str, sim: str, num_envs: int) -> dict:
    """Count the pairs one backend will test, from its built model."""
    cfgs = _PRESETS[preset](sim_type=sim, num_envs=num_envs).build()
    env = BaseRunner._create_env_from_config(cfgs)

    if sim == "mujoco":
        geoms, pairs = _mujoco_pairs(env.scene_manager.sim.mj_model)
    elif sim == "genesis":
        solver = env.scene_manager.scene.rigid_solver
        # Genesis resolves its own pair table at build time, after applying
        # the parsed masks — this is the set its broadphase walks.
        geoms = int(solver.n_geoms)
        pairs = int(solver.collider._n_possible_pairs)
    else:
        raise ValueError(f"{sim!r} exposes no pair table this diag can read.")

    print(f"  {sim:<9} geoms {geoms:5d}   pairs tested {pairs:6d}")
    return {"sim": sim, "geoms": geoms, "pairs": pairs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(_PRESETS), default="k1_joystick")
    ap.add_argument("--sim", choices=("mujoco", "genesis"), default=None)
    ap.add_argument("--result-json", default=None)
    # One environment's worth is enough: the pair table is a property of the
    # model, and building 8192 of them to count it would only cost memory.
    ap.add_argument("--num-envs", type=int, default=16)
    args = ap.parse_args()

    if args.sim is not None:
        result = run_single(args.preset, args.sim, args.num_envs)
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(result))
        return 0

    # One process per backend. Building two simulators in one process is
    # what the single-backend import guard exists to prevent.
    tmp = Path(tempfile.mkdtemp(prefix="pair_audit_"))
    out: dict[str, dict] = {}
    env_vars = dict(os.environ, JAXRLWORLD_ALLOW_MULTI_SIM="1")
    print("=" * 78)
    print(f"COLLISION PAIR AUDIT  [preset={args.preset}]")
    print("=" * 78)
    for sim in ("mujoco", "genesis"):
        path = tmp / f"{sim}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jaxrlworld.scripts.diag.parity.collision_pair_audit",
                "--preset",
                args.preset,
                "--sim",
                sim,
                "--num-envs",
                str(args.num_envs),
                "--result-json",
                str(path),
            ],
            env=env_vars,
            check=False,
        )
        if path.exists():
            out[sim] = json.loads(path.read_text())

    print("=" * 78)
    if len(out) == 2:
        for sim, r in out.items():
            print(f"  {sim:<9} collision geoms {r['geoms']:5d}   pairs tested {r['pairs']:6d}")
        if out["mujoco"]["pairs"]:
            print(f"  genesis / mujoco = {out['genesis']['pairs'] / out['mujoco']['pairs']:.2f}x pairs")
    print("  Equal counts mean the masks survived both parsers and the speed")
    print("  gap is elsewhere. A large ratio means the two backends are")
    print("  testing different contacts — a physics difference first, and")
    print("  only incidentally a cost.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
