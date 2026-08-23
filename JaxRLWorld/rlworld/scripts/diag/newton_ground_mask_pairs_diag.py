"""Does the ground collision-mask pass still say the same thing?

The pass that registers ground-vs-robot forbidden pairs used to build the
full upper triangle of every replicated shape and then throw away every
pair the ground was not in. That is quadratic in ``num_worlds *
shapes_per_world``: go2 at 4096 worlds carries 188416 colliding shapes,
whose triangle is 1.8e10 pairs, and numpy asks for 132 GiB before a
single mask is read. It now compares ground against everything else
directly, which is linear.

Same answer or not is the whole question, and it is asked three ways so
that no single mistake can pass all of them:

  [1] EQUIVALENCE -- the old code, run verbatim on the same builder, must
      return exactly the set the new code registered. Both call the same
      ``_forbidden_pairs``, so this catches a restructuring error and
      nothing else.

  [2] GROUND TRUTH -- MuJoCo's rule re-derived here from the builder's raw
      ``mujoco:contype`` / ``mujoco:conaffinity`` attributes, sharing no
      code with the thing under test. Every registered pair must be
      forbidden by it (soundness) and every unregistered ground pair must
      be allowed by it (completeness). This is the check that survives
      both implementations being wrong in the same way.

  [3] SCALING -- shape and pair counts at N worlds and at 2N, with the
      per-world slope and the constant term SOLVED FOR rather than
      assumed. Pairs must be purely per-world (no constant term), and
      the only shape that is not per-world must be the ground, which the
      scene adds once globally instead of into the replicated template.
      Linear with no constant means the per-world answer did not change
      with N -- what the old code could not demonstrate, having died
      first.

Usage:
    python -m rlworld.scripts.diag.newton_ground_mask_pairs_diag
    python -m rlworld.scripts.diag.newton_ground_mask_pairs_diag --robots k1
"""

from __future__ import annotations

import argparse
import importlib
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")


ROBOTS: dict[str, tuple[str, str]] = {
    "g1": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "go2": ("rlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "k1": ("rlworld.rl.configs.presets.k1_joystick.base", "K1JoystickConfig"),
    "t1": ("rlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
}
GROUND_HINTS = ("ground", "terrain", "plane")
TRAINING_ENVS = 4096


def capture(robot: str, num_envs: int) -> dict:
    """Build the scene, keeping the builder the ground pass ran on."""
    from rlworld.rl.envs.managers.newton.scene import NewtonSceneManager
    from rlworld.rl.runners import BaseRunner

    seen: dict = {}
    original = NewtonSceneManager._honour_mjcf_masks

    def spy(self, builder, what, ground_only=False):
        before = len(builder.shape_collision_filter_pairs)
        original(self, builder, what, ground_only=ground_only)
        if ground_only:
            seen["builder"] = builder
            seen["registered"] = list(builder.shape_collision_filter_pairs[before:])

    NewtonSceneManager._honour_mjcf_masks = spy
    try:
        mod_path, cls_name = ROBOTS[robot]
        cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
        env = BaseRunner._create_env_from_config(cfg_cls(sim_type="newton", num_envs=num_envs).build())
    finally:
        NewtonSceneManager._honour_mjcf_masks = original
    if "builder" not in seen:
        raise RuntimeError("the ground pass never ran -- _honour_mjcf_masks was not called with ground_only")
    seen["env"] = env
    return seen


def partition(builder) -> tuple[list[int], list[int]]:
    """The colliding shapes, and which of them are the world's ground."""
    from newton import ShapeFlags

    flags = builder.shape_flags
    colliding = [i for i in range(len(flags)) if int(flags[i]) & int(ShapeFlags.COLLIDE_SHAPES)]
    ground = [i for i in colliding if any(h in builder.shape_label[i].lower() for h in GROUND_HINTS)]
    return colliding, ground


def raw_masks(builder, shape: int) -> tuple[int, int]:
    """contype / conaffinity straight off the builder, no shared helper.

    Deliberately re-derived rather than imported: a check that reuses the
    code it is checking can only confirm that code is self-consistent.
    """
    from newton._src.solvers.mujoco.collision_masks import MUJOCO_COLLISION_MASK_UNSET

    attrs = builder.custom_attributes
    a = (attrs["mujoco:contype"].values or {}).get(shape, MUJOCO_COLLISION_MASK_UNSET)
    b = (attrs["mujoco:conaffinity"].values or {}).get(shape, MUJOCO_COLLISION_MASK_UNSET)
    if MUJOCO_COLLISION_MASK_UNSET in (a, b):
        return 1, 1  # MuJoCo's own default for an unauthored geom
    return int(a) & 0xFFFFFFFF, int(b) & 0xFFFFFFFF


def may_touch(builder, a: int, b: int) -> bool:
    """MuJoCo's rule, verbatim."""
    ct_a, ca_a = raw_masks(builder, a)
    ct_b, ca_b = raw_masks(builder, b)
    return bool((ct_a & ca_b) | (ct_b & ca_a))


def old_ground_pairs(builder, colliding: list[int], ground: list[int]) -> list[tuple[int, int]]:
    """The pre-fix implementation, verbatim, for the equivalence check."""
    from rlworld.rl.envs.managers.newton.scene import _forbidden_pairs

    pairs: list[tuple[int, int]] = []
    for g in ground:
        pairs += _forbidden_pairs(builder, [g, *(s for s in colliding if s != g)])
    return [p for p in pairs if p[0] in ground or p[1] in ground]


def canon(pairs) -> set[tuple[int, int]]:
    return {(min(a, b), max(a, b)) for a, b in pairs}


def describe(builder, pair: tuple[int, int]) -> str:
    a, b = pair
    return f"{builder.shape_label[a]} {raw_masks(builder, a)} <-> " f"{builder.shape_label[b]} {raw_masks(builder, b)}"


def run(robot: str, num_envs: int, failures: list[str]) -> tuple[int, int]:
    seen = capture(robot, num_envs)
    builder, registered = seen["builder"], canon(seen["registered"])
    colliding, ground = partition(builder)

    print(f"  {robot} @ {num_envs} worlds")
    print(f"    colliding shapes        {len(colliding)}")
    print(f"    ground shapes           {len(ground)}  {[builder.shape_label[g] for g in ground]}")
    print(f"    registered ground pairs {len(registered)}")

    # [1] EQUIVALENCE
    old = canon(old_ground_pairs(builder, colliding, ground))
    if old == registered:
        print(f"    [1] EQUIVALENCE         PASS  old implementation returns the same {len(old)} pairs")
    else:
        print(f"    [1] EQUIVALENCE         FAIL  old {len(old)} vs new {len(registered)}")
        for pair in sorted(old - registered)[:5]:
            print(f"          only old: {describe(builder, pair)}")
        for pair in sorted(registered - old)[:5]:
            print(f"          only new: {describe(builder, pair)}")
        failures.append(f"{robot}@{num_envs}: new ground pass differs from the old one")

    # [2] GROUND TRUTH, re-derived
    unsound = [p for p in registered if may_touch(builder, *p)]
    missing = [
        (g, s)
        for g in ground
        for s in colliding
        if s != g and not may_touch(builder, g, s) and (min(g, s), max(g, s)) not in registered
    ]
    if not unsound and not missing:
        print(f"    [2] GROUND TRUTH        PASS  {len(registered)} forbidden, {len(ground) * len(colliding)} checked")
    else:
        print(f"    [2] GROUND TRUTH        FAIL  {len(unsound)} not actually forbidden, {len(missing)} missed")
        for pair in unsound[:5]:
            print(f"          registered but allowed: {describe(builder, pair)}")
        for pair in missing[:5]:
            print(f"          forbidden but missed:   {describe(builder, pair)}")
        failures.append(f"{robot}@{num_envs}: ground pairs disagree with MuJoCo's rule")

    per_world = sorted(
        builder.shape_label[a].split("/")[-1] + " <-> " + builder.shape_label[b].split("/")[-1] for a, b in registered
    )
    for label in sorted(set(per_world)):
        print(f"          blocks {per_world.count(label):>4}x  {label}")

    triangle = len(colliding) * (len(colliding) - 1) // 2
    print(f"    old cost here           {triangle:,} pairs in the triangle vs {len(ground) * len(colliding):,} now")

    return len(registered), len(colliding), len(ground)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=sorted(ROBOTS), choices=sorted(ROBOTS))
    ap.add_argument("--worlds", type=int, default=4, help="world count for the first build")
    args = ap.parse_args()

    failures: list[str] = []
    for robot in args.robots:
        print(f"\n=== {robot} ===")
        worlds = args.worlds
        n_small, shapes_small, n_ground = run(robot, worlds, failures)
        n_big, shapes_big, _ = run(robot, worlds * 2, failures)

        # [3] SCALING -- fit shapes(N) = per_world * N + constant from the
        # two builds instead of assuming a doubling. The ground is added
        # once to the scene rather than into the replicated template, so
        # the constant is expected to be exactly the ground; anything
        # else in it is a shape that failed to replicate.
        shape_slope, shape_rest = divmod(shapes_big - shapes_small, worlds)
        pair_slope, pair_rest = divmod(n_big - n_small, worlds)
        shape_const = shapes_small - shape_slope * worlds
        pair_const = n_small - pair_slope * worlds

        problems = []
        if shape_rest or pair_rest:
            problems.append(f"not linear in the world count (remainders {shape_rest}, {pair_rest})")
        if shape_const != n_ground:
            problems.append(f"{shape_const} shapes are not per-world, expected {n_ground} (the ground)")
        if pair_const != 0:
            problems.append(f"{pair_const} pairs are not per-world; a pair must belong to a world")

        verdict = "PASS" if not problems else "FAIL"
        print(
            f"    [3] SCALING             {verdict}  shapes {shapes_small} -> {shapes_big}, pairs {n_small} -> {n_big}"
        )
        print(
            f"          shapes(N) = {shape_slope} * N + {shape_const}   ({n_ground} ground shape(s) added once, globally)"
        )
        print(f"          pairs(N)  = {pair_slope} * N + {pair_const}")
        for problem in problems:
            print(f"          {problem}")
            failures.append(f"{robot}: {problem}")

        at_training = shape_slope * TRAINING_ENVS + shape_const
        triangle = at_training * (at_training - 1) // 2
        print(
            f"    at {TRAINING_ENVS} worlds        {at_training:,} shapes -> old triangle {triangle:,} pairs "
            f"({triangle * 8 / 2**30:,.0f} GiB), new {n_ground * at_training:,} comparisons"
        )

    print("\n" + "=" * 70)
    if failures:
        print(f"OVERALL FAIL ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OVERALL PASS -- the ground pass registers exactly the pairs MuJoCo's rule forbids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
