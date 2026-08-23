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
    "t1": ("rlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
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


def mjwarp_rows(env, sim: str, report: bool = False) -> np.ndarray:
    """Friction on every touching foot-ground row, and which geom made it."""
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
    # Keyed by the foot side, because "the foot touches the ground" and
    # "the foot's SHELL touches the ground" are different facts and the
    # totals cannot tell them apart. The K1 asset masks the shell away
    # from the ground; whether a backend honours that is the question.
    seen: dict[str, list] = {}
    for row in range(nacon):
        if dist[row] >= 0.0:
            continue
        a, b = int(geom[row, 0]), int(geom[row, 1])
        ground = (
            a in world_geom or is_ground(names[a]),
            b in world_geom or is_ground(names[b]),
        )
        foot = (is_foot(names[a]), is_foot(names[b]))
        if (foot[0] and ground[1]) or (foot[1] and ground[0]):
            keep.append(mu[row])
            key = names[a] if foot[0] else names[b]
            seen.setdefault(key.rsplit("/", 1)[-1], []).append(float(mu[row]))
    if report:
        print(f"      which foot geom is touching the ground ({sim}):")
        for key, values in sorted(seen.items()):
            arr = np.asarray(values)
            print(f"        {key:<28}{arr.size:>6} rows  " f"{arr.min():.4f} .. {arr.max():.4f}")
    return np.asarray(keep)


def genesis_rows(env, report: bool = False) -> np.ndarray:
    from genesis.utils.misc import qd_to_torch

    solver = env.scene_manager.scene.sim.rigid_solver
    owner = [f"<geom {i}>" for i in range(solver.n_geoms)]
    for entity in solver.entities:
        for link in entity.links:
            for gid in range(link.geom_start, link.geom_end):
                # Link name AND the index within the link: Genesis discards
                # MJCF geom names, and a foot link here carries a shell box
                # plus four spheres whose contact masks differ. Reporting
                # only the link cannot tell which of them is touching, and
                # that is the question.
                owner[gid] = f"{link.name}/g{gid - link.geom_start}"

    state = solver.collider._collider_state
    n_con = _np(qd_to_torch(state.n_contacts, copy=True))
    mu = _np(qd_to_torch(state.contact_data.friction, transpose=True, copy=True))
    geom_a = _np(qd_to_torch(state.contact_data.geom_a, transpose=True, copy=True))
    geom_b = _np(qd_to_torch(state.contact_data.geom_b, transpose=True, copy=True))
    keep = []
    seen: dict[str, list] = {}
    for world in range(mu.shape[0]):
        for row in range(int(n_con[world])):
            a, b = owner[int(geom_a[world, row])], owner[int(geom_b[world, row])]
            if (is_foot(a) and is_ground(b)) or (is_foot(b) and is_ground(a)):
                keep.append(mu[world, row])
                key = a if is_foot(a) else b
                seen.setdefault(key, []).append(float(mu[world, row]))
    if report:
        print("      which foot geom is touching the ground (genesis):")
        for key, values in sorted(seen.items()):
            arr = np.asarray(values)
            print(f"        {key:<28}{arr.size:>6} rows  " f"{arr.min():.4f} .. {arr.max():.4f}")
    return np.asarray(keep)


def geom_mu(env, sim: str) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]:
    """Every foot-region and ground geom by NAME, with its range over worlds.

    Ranges alone cannot separate "the same geom carries a different
    coefficient" from "the two backends are looking at different geoms" --
    and the second is easy to do by accident here, because mjwarp keeps
    the MJCF geom names while Genesis keeps only the link's, so a match on
    "foot" resolves to four named spheres on one side and everything
    hanging off the foot link on the other. Listed per geom so the two
    cases read differently.
    """
    if sim == "genesis":
        from genesis.utils.misc import qd_to_torch

        solver = env.scene_manager.scene.sim.rigid_solver
        owner = [f"<geom {i}>" for i in range(solver.n_geoms)]
        for entity in solver.entities:
            for link in entity.links:
                for gid in range(link.geom_start, link.geom_end):
                    owner[gid] = f"{link.name}/g{gid - link.geom_start}"
        base = _np(qd_to_torch(solver.dyn_info.geoms.friction, copy=True))

        def rows(ids):
            if not ids:
                return []
            # Genesis applies friction DR as a RATIO on the stored base, so
            # the base alone is not what the solve sees.
            ratio = _np(solver.get_geoms_friction_ratio(geoms_idx=ids))
            mu = base[ids][None, :] * ratio
            return [(owner[gid], float(mu[:, i].min()), float(mu[:, i].max())) for i, gid in enumerate(ids)]

        feet = [g for g, n in enumerate(owner) if is_foot(n)]
        ground = [g for g, n in enumerate(owner) if is_ground(n)]
        return rows(feet), rows(ground)

    import mujoco

    manager = env.scene_manager
    if sim == "newton":
        mj_model, wp_model = manager.solver.mj_model, manager.solver.mjw_model
    else:
        mj_model, wp_model = manager.mj_model, manager.model
    names = [
        (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"<geom {gid}>").rsplit("/", 1)[-1]
        for gid in range(mj_model.ngeom)
    ]
    mu = _np(wp_model.geom_friction)[..., 0]
    if mu.ndim == 1:
        mu = mu[None, :]

    def rows(ids):
        return [(names[g], float(mu[:, g].min()), float(mu[:, g].max())) for g in ids]

    feet = [g for g, n in enumerate(names) if is_foot(n)]
    ground = [g for g, n in enumerate(names) if int(mj_model.geom_bodyid[g]) == 0 or is_ground(n)]
    return rows(feet), rows(ground)


def masks(env, sim: str) -> list[tuple[str, int, int]]:
    """``contype`` / ``conaffinity`` for the foot and ground geoms.

    A pair collides when ``contype_a & conaffinity_b`` or
    ``contype_b & conaffinity_a`` is non-zero -- the same test in MuJoCo
    and in Genesis (``collider.py:406``). So a foot shell the asset masks
    away from the ground on one backend and not on another is a
    disagreement about these four numbers, and nothing else.
    """
    if sim == "genesis":
        solver = env.scene_manager.scene.sim.rigid_solver
        out = []
        for entity in solver.entities:
            for link in entity.links:
                for i, geom in enumerate(link.geoms):
                    label = f"{link.name}/g{i}"
                    if is_foot(label) or is_ground(label):
                        out.append((label, int(geom.contype), int(geom.conaffinity)))
        return out

    import mujoco

    manager = env.scene_manager
    mj_model = manager.solver.mj_model if sim == "newton" else manager.mj_model
    out = []
    for gid in range(mj_model.ngeom):
        name = (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"<geom {gid}>").rsplit("/", 1)[-1]
        if is_foot(name) or is_ground(name) or int(mj_model.geom_bodyid[gid]) == 0:
            out.append((name, int(mj_model.geom_contype[gid]), int(mj_model.geom_conaffinity[gid])))
    return out


def priority_census(env, sim: str) -> str:
    """How many collidable geoms carry a non-zero priority, and out of how many.

    The scene manager no longer paints priority onto collision shapes, so
    this reads what the MJCF authors. For a robot whose MJCF authors 1 on
    ALL of them the forced value and the authored value were the same, and
    removing that workaround provably changed nothing -- which is the whole
    check for T1, whose ``<default>`` class carries ``priority="1"``.
    """

    manager = env.scene_manager
    if sim == "newton":
        mj_model, wp_model = manager.solver.mj_model, manager.solver.mjw_model
    else:
        mj_model, wp_model = manager.mj_model, manager.model
    priority = _np(wp_model.geom_priority)
    if priority.ndim > 1:
        priority = priority[0]
    collidable = [
        gid for gid in range(mj_model.ngeom) if int(mj_model.geom_contype[gid]) or int(mj_model.geom_conaffinity[gid])
    ]
    nonzero = sum(1 for gid in collidable if int(priority[gid]) != 0)
    return f"{nonzero}/{len(collidable)} collidable geoms at priority != 0"


def measure(robot: str, sim: str) -> np.ndarray:
    env = build(robot, sim)
    zero = torch.zeros(NUM_ENVS, env.act_manager.num_actions, device=env.device)
    for _ in range(SETTLE):
        env.step(zero)
    env._invalidate_cache()
    rows = genesis_rows(env, report=True) if sim == "genesis" else mjwarp_rows(env, sim, report=True)
    feet, ground = geom_mu(env, sim)
    print(f"      stored mu per geom ({sim}):")
    for label, lo, hi in feet + ground:
        print(f"        {label:<34}{lo:8.4f} .. {hi:8.4f}")
    print(f"      contype / conaffinity ({sim}):")
    for label, contype, conaffinity in masks(env, sim):
        print(f"        {label:<34}{contype:>6} / {conaffinity:<6}")
    if sim != "genesis":
        print(f"    {'':<12}{'':>8}   {priority_census(env, sim)}")
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
