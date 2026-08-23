"""The friction each backend actually uses where a foot meets the ground.

Not the coefficient stored on a geom -- the one on the live contact row,
after each engine has combined the foot's value with the ground's by its
own rule. Those rules differ, and the difference does not appear anywhere
in a config:

  * MuJoCo (mjlab, and Newton through SolverMuJoCo) gives the geom with
    the HIGHER ``priority`` its values outright, and takes the
    element-wise max only when the priorities tie.
  * Genesis has no priority concept and combines with max() throughout.
  * ``NewtonSceneManager`` paints priority 1 onto every collision shape,
    which makes every mjwarp pair tie, which forces max() there too.

Four robot assets author ``priority="1"`` with an explicit friction on
their feet precisely so the foot wins. Whether that intent survives is a
per-backend question, and a robot that grips at 1.0 on one simulator and
0.6 on another is not a policy difference -- it is a task that trains on
one and not the other, with nothing in the logs to say why.

So: settle each robot on each backend, find the rows where a foot geom
touches the ground, and read the friction off them.

Usage:
    python -m rlworld.scripts.diag.foot_friction_effective_diag
    python -m rlworld.scripts.diag.foot_friction_effective_diag --robots g1
"""

from __future__ import annotations

import argparse
import importlib
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROBOTS: dict[str, tuple[str, str]] = {
    "g1": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "go2": ("rlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "k1": ("rlworld.rl.configs.presets.k1_joystick.base", "K1JoystickConfig"),
}
SIMS = ("mujoco", "newton", "genesis")

FOOT_HINTS = ("foot", "ankle_roll")
"""Substrings that name a foot, whichever way a backend spells the shape.
Genesis loses MJCF geom names and labels by link, so the link name is what
survives on that side -- and every one of these robots puts its foot geoms
on a link whose name says so."""

GROUND_HINTS = ("ground", "plane", "terrain")

NUM_ENVS = 16
SETTLE = 120
"""Long enough for the PD to settle the default pose onto its feet."""


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


def is_foot(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in FOOT_HINTS)


def is_ground(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in GROUND_HINTS)


def build(robot: str, sim: str):
    from rlworld.rl.runners import BaseRunner

    mod_path, cls_name = ROBOTS[robot]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=NUM_ENVS).build()
    return BaseRunner._create_env_from_config(cfgs)


def mjwarp_rows(env, sim: str) -> np.ndarray:
    """Friction on every touching foot-ground row."""
    import mujoco

    manager = env.scene_manager
    if sim == "newton":
        mj_model, data = manager.solver.mj_model, manager.solver.mjw_data
    else:
        mj_model, data = manager.mj_model, manager.data
    names = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"<geom {gid}>" for gid in range(mj_model.ngeom)
    ]
    world_geom = {gid for gid in range(mj_model.ngeom) if int(mj_model.geom_bodyid[gid]) == 0}

    nacon = int(_np(data.nacon)[0])
    if nacon == 0:
        return np.array([])
    geom = _np(data.contact.geom)[:nacon]
    dist = _np(data.contact.dist)[:nacon]
    mu = _np(data.contact.friction)[:nacon, 0]
    keep = []
    for row in range(nacon):
        if dist[row] >= 0.0:
            continue
        a, b = int(geom[row, 0]), int(geom[row, 1])
        ground = (a in world_geom or is_ground(names[a]), b in world_geom or is_ground(names[b]))
        foot = (is_foot(names[a]), is_foot(names[b]))
        if (foot[0] and ground[1]) or (foot[1] and ground[0]):
            keep.append(mu[row])
    return np.asarray(keep)


def genesis_rows(env) -> np.ndarray:
    from genesis.utils.misc import qd_to_torch

    solver = env.scene_manager.scene.sim.rigid_solver
    owner = [f"<geom {i}>" for i in range(solver.n_geoms)]
    for entity in solver.entities:
        for link in entity.links:
            for gid in range(link.geom_start, link.geom_end):
                owner[gid] = link.name

    state = solver.collider._collider_state
    n_con = _np(qd_to_torch(state.n_contacts, copy=True))
    mu = _np(qd_to_torch(state.contact_data.friction, transpose=True, copy=True))
    geom_a = _np(qd_to_torch(state.contact_data.geom_a, transpose=True, copy=True))
    geom_b = _np(qd_to_torch(state.contact_data.geom_b, transpose=True, copy=True))
    keep = []
    for world in range(mu.shape[0]):
        for row in range(int(n_con[world])):
            a, b = owner[int(geom_a[world, row])], owner[int(geom_b[world, row])]
            if (is_foot(a) and is_ground(b)) or (is_foot(b) and is_ground(a)):
                keep.append(mu[world, row])
    return np.asarray(keep)


def measure(robot: str, sim: str) -> np.ndarray:
    env = build(robot, sim)
    zero = torch.zeros(NUM_ENVS, env.act_manager.num_actions, device=env.device)
    for _ in range(SETTLE):
        env.step(zero)
    env._invalidate_cache()
    rows = genesis_rows(env) if sim == "genesis" else mjwarp_rows(env, sim)
    del env
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=sorted(ROBOTS), choices=sorted(ROBOTS))
    ap.add_argument("--sims", nargs="+", default=list(SIMS), choices=list(SIMS))
    args = ap.parse_args()

    print("=" * 84)
    print("  FRICTION ON THE LIVE FOOT-GROUND CONTACT, PER BACKEND")
    print("=" * 84)
    print(f"  {NUM_ENVS} envs, {SETTLE} settling steps, zero action (the default pose)")

    failures: list[str] = []
    for robot in args.robots:
        print("\n" + "-" * 84)
        print(f"  {robot.upper()}")
        print("-" * 84)
        print(f"    {'backend':<12}{'rows':>8}{'min':>10}{'mean':>10}{'max':>10}")
        summary: dict[str, float] = {}
        for sim in args.sims:
            rows = measure(robot, sim)
            if rows.size == 0:
                print(f"    {sim:<12}{0:>8}   no foot-ground contact found")
                failures.append(f"{robot}/{sim}: no foot-ground contact after {SETTLE} steps")
                continue
            print(f"    {sim:<12}{rows.size:>8}{rows.min():>10.4f}" f"{rows.mean():>10.4f}{rows.max():>10.4f}")
            summary[sim] = float(rows.mean())
        if len(summary) > 1:
            spread = max(summary.values()) - min(summary.values())
            if spread > 1e-3:
                print(f"    -> the backends DISAGREE by {spread:.4f}")
                failures.append(f"{robot}: foot friction differs by {spread:.4f} across backends")
            else:
                print("    -> the backends agree")

    print("\n" + "=" * 84)
    if failures:
        print(f"  {len(failures)} FINDINGS")
        for line in failures:
            print(f"    {line}")
    else:
        print("  every backend grips the ground with the same friction")
    print("=" * 84)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
