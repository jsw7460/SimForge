"""Why Newton rebuilds the MuJoCo collision masks instead of reusing them.

The K1 foot stands on four spheres on mjlab and on its shell box as well
on Newton, because the two backends give the shell-ground pair different
``contype`` / ``conaffinity``. Newton's are not the MJCF's: ground
``1/1`` becomes ``2/3``, the shell ``4/4`` becomes ``3/2``, and the pair
turns from forbidden into allowed.

That is not Newton losing the values. ``SolverMuJoCo`` tries to reuse
them and says exactly when it will not
(``solver_mujoco.py``, "Reuse the original masks only when all shapes
came from the same ``add_mjcf()`` call and the masks already enforce
every Newton filter"). Four conditions have to hold at once:

  * every colliding shape carries a preserved ``contype``
  * every one carries a preserved ``conaffinity``
  * every one carries a ``collision_mask_domain``, and they are all the
    SAME domain -- one ``add_mjcf`` call for the lot
  * the preserved masks already forbid every pair Newton's own filters
    forbid

Fail any and Newton compiles fresh masks out of its allowed-pair set,
which cannot know what the MJCF meant. Our scenes put the robot in
through ``add_mjcf`` and the ground in through the terrain importer, so
the third condition is the one to look at first -- but this prints all
four rather than assuming.

Usage:
    python -m jaxrlworld.scripts.diag.engine.newton_collision_mask_domain_diag
    python -m jaxrlworld.scripts.diag.engine.newton_collision_mask_domain_diag --robots k1
"""

from __future__ import annotations

import argparse
import importlib
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np  # noqa: E402

ROBOTS: dict[str, tuple[str, str]] = {
    "g1": ("jaxrlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "go2": ("jaxrlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "k1": ("jaxrlworld.rl.configs.presets.k1_joystick.base", "K1JoystickConfig"),
    "t1": ("jaxrlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
}


def build(robot: str):
    from jaxrlworld.rl.runners import BaseRunner

    mod_path, cls_name = ROBOTS[robot]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type="newton", num_envs=2).build()
    return BaseRunner._create_env_from_config(cfgs)


def preserved(model, name: str):
    """One preserved MuJoCo attribute off the Newton model, or None."""
    attrs = getattr(model, "mujoco", None)
    if attrs is None:
        return None
    attr = getattr(attrs, name, None)
    return None if attr is None else attr.numpy()


def run(robot: str) -> list[str]:
    from newton._src.solvers.mujoco.solver_mujoco import (
        MUJOCO_COLLISION_MASK_DOMAIN_UNSET,
        MUJOCO_COLLISION_MASK_UNSET,
    )

    failures: list[str] = []
    print("=" * 88)
    print(f"  {robot.upper()}")
    print("=" * 88)

    env = build(robot)
    solver = env.scene_manager.solver
    model = solver.model
    labels = list(model.shape_label)
    contype = preserved(model, "contype")
    conaffinity = preserved(model, "conaffinity")
    domain = preserved(model, "collision_mask_domain")

    if contype is None or conaffinity is None or domain is None:
        print("    the model carries no preserved MuJoCo masks at all")
        failures.append(f"{robot}: preserved masks absent")
        del env
        return failures

    flags = np.asarray(model.shape_flags.numpy())
    from newton import ShapeFlags

    colliding = np.flatnonzero(flags & int(ShapeFlags.COLLIDE_SHAPES))

    unset_type = int((contype[colliding] == MUJOCO_COLLISION_MASK_UNSET).sum())
    unset_aff = int((conaffinity[colliding] == MUJOCO_COLLISION_MASK_UNSET).sum())
    unset_dom = int((domain[colliding] == MUJOCO_COLLISION_MASK_DOMAIN_UNSET).sum())
    domains = np.unique(domain[colliding])

    print(f"    {len(colliding)} colliding shapes")
    print(f"    contype unset on      {unset_type}")
    print(f"    conaffinity unset on  {unset_aff}")
    print(f"    domain unset on       {unset_dom}")
    print(f"    distinct domains      {domains.tolist()}")

    ok = unset_type == 0 and unset_aff == 0 and unset_dom == 0 and domains.size <= 1
    if ok:
        print("    -> the first three conditions HOLD; if Newton still rebuilt the")
        print("       masks it is the fourth (preserved masks vs Newton's filters)")
    else:
        print("    -> Newton CANNOT reuse the preserved masks, so it compiles new")
        print("       ones from its own allowed-pair set, and the MJCF's intent")
        print("       is gone")
        failures.append(f"{robot}: preserved masks unusable ({domains.size} domains)")

    print(f"\n    {'shape':<52}{'preserved':>14}{'domain':>9}{'final':>12}")
    mjw = solver.mjw_model
    final_type = np.asarray(mjw.geom_contype.numpy())
    final_aff = np.asarray(mjw.geom_conaffinity.numpy())
    if final_type.ndim > 1:
        final_type, final_aff = final_type[0], final_aff[0]
    for i, gid in enumerate(colliding):
        label = labels[gid].rsplit("/", 1)[-1][-50:]
        pres = f"{int(contype[gid])}/{int(conaffinity[gid])}"
        fin = f"{int(final_type[i])}/{int(final_aff[i])}" if i < final_type.size else "-"
        print(f"    {label:<52}{pres:>14}{int(domain[gid]):>9}{fin:>12}")

    del env
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=sorted(ROBOTS), choices=sorted(ROBOTS))
    args = ap.parse_args()

    print("=" * 88)
    print("  CAN NEWTON REUSE THE MJCF's COLLISION MASKS")
    print("=" * 88)

    failures: list[str] = []
    for robot in args.robots:
        failures += run(robot)

    print("\n" + "=" * 88)
    if failures:
        for line in failures:
            print(f"  {line}")
        print("  Where a scene mixes an add_mjcf robot with a ground plane from")
        print("  somewhere else, the domains differ and the reuse is refused. The")
        print("  fix is on our side: give the terrain the same domain and masks.")
    else:
        print("  every scene keeps its authored masks")
    print("=" * 88)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
