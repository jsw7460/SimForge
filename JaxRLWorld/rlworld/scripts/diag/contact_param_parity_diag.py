"""Do the three backends give the same geom the same contact parameters?

The third question in the series, and the one nothing was asking.
``contact_pair_parity_diag`` establishes that the backends built the same
collision geometry and that the same pairs of it touch. Two robots can
pass both and still behave differently, because what a contact DOES is
set by ``friction``, ``solref``, ``solimp``, ``condim`` and ``priority``,
and those travel by a different route on every backend.

That route is where they diverge. mjlab can edit the compiled spec, so a
preset may hand it a ``CollisionCfg`` that rewrites every collision
geom's parameters after the XML is read -- and Newton and Genesis, which
only ever see the XML, get none of it. It has happened twice:
``b803250`` (the yam arm carried mjlab-only per-geom overrides, and the
tong pad's friction differed by 11%) and T1 getup, whose
``FULL_COLLISION`` sets friction 0.6 and solref 0.01 on the mjlab path
while the other two take the MJCF defaults of 1.0 and 0.02.

Both were invisible to every existing check. A parameter divergence has
no symptom a pair table can show: the same geoms touch, in the same
poses, and merely grip differently.

``contype`` and ``conaffinity`` are compared too. They decide whether a
pair is considered AT ALL, so a difference there does not shade a
contact, it deletes one, and Genesis is the backend whose mask handling
this repo has had to change. Read off the live model on all three, not
off the MJCF, since the whole question is what survived the import.

Read from each backend's own live model, never from the asset -- the
point is what the solver will use, after any spec edit and after the
parser's own conversions:

* ``mujoco`` and ``newton`` both hold a compiled ``mj_model``.
* ``genesis`` keeps the same quantities on ``dyn_info.geoms``:
  ``friction``, ``friction_torsional``, ``friction_rolling``, and a
  7-vector ``sol_params`` that is MuJoCo's ``solref`` followed by its
  ``solimp``.

**Spin and roll are compared as EFFECTIVE values.** MuJoCo carries a
full friction triple on every geom and then ignores the spin term below
``condim`` 4 and the roll term below 6; Genesis applies that rule at
parse time instead, storing zero (``genesis/utils/mjcf.py``, "the
coefficients of a lower-condim geom must parse as inert"). The same rule
is applied to the MuJoCo side here, so the columns mean the same thing.
Comparing the raw triples would flag every low-condim geom of every
robot and say nothing.

**The ground's stored friction is SUPPOSED to differ on Genesis** and is
not a finding. MuJoCo and Newton give the robot's collision geoms
``priority`` 1, so at a robot-ground contact the robot's coefficient wins
outright and the ground's value never enters. Genesis has no priority and
combines a pair by ``max``, so the ground is anchored low (0.01) to let
the robot's win there too (``fc201cb``). All three end up applying the
robot's number; only the unused input differs.

**A stored value can agree and still be unused.** Genesis parses a geom's
torsional and rolling coefficients but applies them only when the SCENE
turns them on (``RigidOptions.enable_torsional_friction`` /
``enable_rolling_friction``, both False by default), so a robot whose
feet are authored ``condim`` 6 reads identical spin and roll numbers on
all three backends while spinning freely on one of them. The per-geom
table cannot see that, because it is a solver option and not a geom
parameter. So the flags are checked separately, against the highest
condim any geom in the scene actually declares.

``condim`` and ``priority`` have no Genesis equivalent -- it has no
per-geom condim, and it resolves a pair's friction by ``max`` rather
than by priority -- so those two columns are left blank for it rather
than filled with an invented value.

Usage:
    python -m rlworld.scripts.diag.contact_param_parity_diag
    python -m rlworld.scripts.diag.contact_param_parity_diag --robots t1
    python -m rlworld.scripts.diag.contact_param_parity_diag --robots t1 --dump
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rlworld.scripts.diag.contact_pair_parity_diag import (  # noqa: E402
    GENESIS_TYPE,
    GROUND,
    MUJOCO_TYPE,
    NUM_ENVS,
    ROBOTS,
    SETTLE,
    build,
    is_ground,
    leaf,
)

SIMS = ("mujoco", "newton", "genesis")

SHARED = (
    "contype",
    "conaffinity",
    "friction_slide",
    "friction_spin",
    "friction_roll",
    "solref_t",
    "solref_d",
    "solimp",
)
MUJOCO_ONLY = ("condim", "priority")


def key_for(body: str, geom_type: str, vocab: set[str]) -> tuple[str, str]:
    name = leaf(body, vocab)
    return (GROUND if geom_type == "plane" or is_ground(name) else name, geom_type)


def read_mujoco(env, sim: str, vocab: set[str]) -> tuple[dict[tuple[str, str], list[dict]], set[str]]:
    import mujoco

    manager = env.scene_manager
    mj_model = manager.solver.mj_model if sim == "newton" else manager.mj_model

    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b) or f"<body {b}>" for b in range(mj_model.nbody)]
    seen = {leaf(name, vocab) for name in names}

    out: dict[tuple[str, str], list[dict]] = {}
    for g in range(mj_model.ngeom):
        if not (int(mj_model.geom_contype[g]) or int(mj_model.geom_conaffinity[g])):
            continue
        body = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, int(mj_model.geom_bodyid[g])) or "?"
        geom_type = MUJOCO_TYPE.get(int(mj_model.geom_type[g]), "?")
        friction = np.asarray(mj_model.geom_friction[g], dtype=float)
        solref = np.asarray(mj_model.geom_solref[g], dtype=float)
        solimp = np.asarray(mj_model.geom_solimp[g], dtype=float)
        condim = int(mj_model.geom_condim[g])
        out.setdefault(key_for(body, geom_type, vocab), []).append(
            {
                "contype": int(mj_model.geom_contype[g]),
                "conaffinity": int(mj_model.geom_conaffinity[g]),
                "friction_slide": round(float(friction[0]), 6),
                # Effective, matching how Genesis parses them -- see the module docstring.
                "friction_spin": round(float(friction[1]), 6) if condim >= 4 else 0.0,
                "friction_roll": round(float(friction[2]), 6) if condim >= 6 else 0.0,
                "solref_t": round(float(solref[0]), 6),
                "solref_d": round(float(solref[1]), 6),
                "solimp": ",".join(f"{v:g}" for v in np.round(solimp, 6)),
                "condim": condim,
                "priority": int(mj_model.geom_priority[g]),
            }
        )
    return out, seen


def read_genesis(env, vocab: set[str]) -> tuple[dict[tuple[str, str], list[dict]], set[str]]:
    from genesis.utils.misc import qd_to_torch

    solver = env.scene_manager.scene.sim.rigid_solver
    geoms = solver.dyn_info.geoms
    contype = qd_to_torch(geoms.contype, copy=True).cpu().numpy().reshape(-1)
    conaffinity = qd_to_torch(geoms.conaffinity, copy=True).cpu().numpy().reshape(-1)
    slide = qd_to_torch(geoms.friction, copy=True).cpu().numpy().reshape(-1)
    spin = qd_to_torch(geoms.friction_torsional, copy=True).cpu().numpy().reshape(-1)
    roll = qd_to_torch(geoms.friction_rolling, copy=True).cpu().numpy().reshape(-1)
    sol = qd_to_torch(geoms.sol_params, copy=True).cpu().numpy().reshape(-1, 7)

    seen = {leaf(link.name, vocab) for entity in solver.entities for link in entity.links}

    out: dict[tuple[str, str], list[dict]] = {}
    for entity in solver.entities:
        for link in entity.links:
            for geom in link.geoms:
                i = int(geom.idx)
                key = key_for(link.name, GENESIS_TYPE.get(int(geom.type), "?"), vocab)
                out.setdefault(key, []).append(
                    {
                        "contype": int(contype[i]),
                        "conaffinity": int(conaffinity[i]),
                        "friction_slide": round(float(slide[i]), 6),
                        "friction_spin": round(float(spin[i]), 6),
                        "friction_roll": round(float(roll[i]), 6),
                        "solref_t": round(float(sol[i, 0]), 6),
                        "solref_d": round(float(sol[i, 1]), 6),
                        "solimp": ",".join(f"{v:g}" for v in np.round(sol[i, 2:7], 6)),
                    }
                )
    return out, seen


def measure(robot: str, sim: str, vocab: set[str]) -> tuple[dict, set[str], dict]:
    env = build(robot, sim)
    zero = torch.zeros(NUM_ENVS, env.act_manager.num_actions, device=env.device)
    for _ in range(SETTLE):
        env.step(zero)
    values, seen = read_genesis(env, vocab) if sim == "genesis" else read_mujoco(env, sim, vocab)
    flags = {}
    if sim == "genesis":
        options = env.scene_manager.scene.sim.rigid_solver._options
        flags = {
            "torsional": bool(options.enable_torsional_friction),
            "rolling": bool(options.enable_rolling_friction),
        }
    del env
    return values, seen, flags


def is_ground_anchor(key: tuple[str, str], field: str, cells: dict[str, str]) -> bool:
    """The one difference that is supposed to be there.

    MuJoCo and Newton resolve a contact pair by ``priority``, so a robot
    geom with priority 1 supplies the whole friction triple and the
    ground's is never read. Genesis has no priority and takes the ``max``
    of the pair, so the ground is anchored BELOW anything a robot could
    carry to reach the same answer (`fc201cb`, and the terrain importer
    says so beside the value). All three end up applying the robot's
    number; only the unused input differs.

    Recognised narrowly, so a real regression still fails: the ground
    only, a friction component only, and only when Genesis is the LOW
    one. A Genesis ground that crept above the others stops being an
    anchor and is reported.
    """
    if key[0] != GROUND or not field.startswith("friction"):
        return False
    try:
        values = {sim: float(text) for sim, text in cells.items() if text != "-"}
    except ValueError:
        return False
    genesis = values.pop("genesis", None)
    return genesis is not None and bool(values) and genesis < min(values.values())


def cell(rows: list[dict] | None, field: str) -> str:
    """Every distinct value this backend gave the key, replicas collapsed."""
    if rows is None:
        return "-"
    seen = sorted({row[field] for row in rows if field in row}, key=str)
    if not seen:
        return "-"
    return ",".join(f"{v:g}" if isinstance(v, int | float) else str(v) for v in seen)


def run(robot: str, sims: list[str], dump: bool) -> list[str]:
    failures: list[str] = []
    print("=" * 104)
    print(f"  {robot.upper()}")
    print("=" * 104)

    # The first backend measured supplies the body-name vocabulary and the
    # rest match into it by longest suffix, exactly as the pair diag does.
    # Without it Newton's flattened names (T1_worldbody_Trunk_Waist_...)
    # key their own rows, every backend looks like it has geoms the others
    # lack, and not one value gets compared with its counterpart.
    read: dict[str, dict] = {}
    flags: dict[str, bool] = {}
    vocab: set[str] = set()
    for sim in sims:
        read[sim], seen, sim_flags = measure(robot, sim, vocab)
        flags.update(sim_flags)
        if not vocab:
            vocab = seen
    keys = sorted({key for values in read.values() for key in values})

    for field in SHARED + MUJOCO_ONLY:
        speaks = [s for s in sims if s != "genesis" or field in SHARED]
        note = "" if field in SHARED else "   (no genesis equivalent)"
        print(f"\n    {field}{note}")
        print(f"      {'body / type':<46}" + "".join(f"{s:>16}" for s in speaks))
        for key in keys:
            cells = [cell(read[sim].get(key), field) for sim in speaks]
            present = [c for c in cells if c != "-"]
            differs = len(set(present)) > 1
            expected = differs and is_ground_anchor(key, field, dict(zip(speaks, cells)))
            label = f"{key[0]} {key[1]}"
            mark = "   <-- anchored on genesis, by design" if expected else ("   <-- DIFFERS" if differs else "")
            print(f"      {label[:44]:<46}" + "".join(f"{c[:15]:>16}" for c in cells) + mark)
            if differs and not expected:
                failures.append(f"{robot}: {field} on {label} = {dict(zip(speaks, cells))}")

    # condim is read off mjlab, which is where it is authoritative; genesis
    # has no per-geom condim to compare against, only the scene switches.
    if "genesis" in sims and "mujoco" in sims:
        declared = max((row["condim"] for rows in read["mujoco"].values() for row in rows), default=3)
        print("\n    genesis solver flags   (a parameter agreeing does not mean it is used)")
        print(f"      {'highest condim declared in the scene':<44}{declared:>8}")
        for name, needed_at in (("torsional", 4), ("rolling", 6)):
            on = flags.get(name, False)
            label = f"enable_{name}_friction   (used from condim {needed_at})"
            missing = declared >= needed_at and not on
            print(f"      {label:<44}{str(on):>8}" + ("   <-- DECLARED BUT IGNORED" if missing else ""))
            if missing:
                failures.append(
                    f"{robot}: geoms declare condim {declared} but genesis "
                    f"enable_{name}_friction is off, so it ignores them"
                )

    if dump:
        print("\n    RAW")
        for key in keys:
            for sim in sims:
                for row in read[sim].get(key, []):
                    print(f"      {sim:<9}{key[0]} {key[1]:<12}{row}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=sorted(ROBOTS), choices=sorted(ROBOTS))
    ap.add_argument("--sims", nargs="+", default=list(SIMS), choices=list(SIMS))
    ap.add_argument("--dump", action="store_true", help="print every raw parameter row")
    args = ap.parse_args()

    print("=" * 104)
    print("  DO THE THREE BACKENDS GIVE THE SAME GEOM THE SAME CONTACT PARAMETERS")
    print("=" * 104)
    print("  the same geoms touching in the same poses can still grip differently")

    failures: list[str] = []
    for robot in args.robots:
        failures += run(robot, args.sims, args.dump)

    print("\n" + "=" * 104)
    if failures:
        print(f"  {len(failures)} PARAMETERS DISAGREE")
        for line in failures:
            print(f"    {line}")
        print("  A parameter divergence has no symptom a pair table can show.")
    else:
        print("  every collision geom carries the same contact parameters")
    print("=" * 104)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
