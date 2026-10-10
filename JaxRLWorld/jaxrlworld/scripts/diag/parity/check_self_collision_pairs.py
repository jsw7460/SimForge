"""Does the compiled MuJoCo model of every preset collide exactly the body pairs its config asks for?

``EntityCfg.enable_self_collisions`` reaches mjlab as a ``<contact><exclude>`` per
body pair of the entity, added by ``_spec_fn_without_self_collisions`` in the
mujoco scene manager. This diag rebuilds every mujoco preset's scene the way
mjlab builds it (entity ``spec_fn`` -> mocap wrapping -> prefixed attach), with
the flags as configured and once more with every flag forced on, compiles both,
and checks on the compiled models:

1. the candidate geom-pair table, computed with mujoco-warp's own filter
   (contype/conaffinity mask, same-weld, parent-child, exclude): an entity
   with the flag off has ZERO intra-entity candidates, one with the flag on
   has the same candidates as the forced-on model;
2. the exclude list of a flag-off entity names every pair of its bodies;
3. candidates between two different entities are identical in both models
   (the exclude touches nothing across entities);
4. random joint configurations within the joint ranges, the robot lifted off
   the ground: a flag-off entity reports zero self contacts in every sample,
   and the forced-on model reports how many samples DID self-collide, which is
   what makes the sample set a real test for that robot;
5. the same configurations with the robot pushed into the ground plane: the set
   of robot-vs-ground contacts is identical in both models.

Needs mujoco and mjlab only (no GPU, no warp).

    python -m jaxrlworld.scripts.diag.parity.check_self_collision_pairs
    python -m jaxrlworld.scripts.diag.parity.check_self_collision_pairs --only yam --samples 500
    python -m jaxrlworld.scripts.diag.parity.check_self_collision_pairs --specs jaxrlworld_private.scripts.diag.preset_sweep_specs
"""

from __future__ import annotations

import argparse
import itertools
import math

import mujoco
import numpy as np
from mjlab.utils.spec import auto_wrap_fixed_base_mocap

from jaxrlworld.rl.configs.scene.unified_entity_config import MujocoEntityCfg
from jaxrlworld.rl.envs.managers.mujoco.scene import _spec_fn_without_self_collisions
from jaxrlworld.rl.utils.resolve import resolve_callable
from jaxrlworld.scripts.diag.gates.check_all_presets import _PUBLIC, _extra_specs, _load

_LIFT_Z = 1.5  # m, free-floating robots are held this high for the self-contact samples
_GROUND_Z = 0.0  # m, and pushed to here for the external-contact samples


def _entity_spec_fn(cfg: MujocoEntityCfg, honor_flag: bool):
    spec_fn = cfg.spec_fn
    if isinstance(spec_fn, str):
        spec_fn = resolve_callable(spec_fn)
    if honor_flag and not cfg.enable_self_collisions:
        spec_fn = _spec_fn_without_self_collisions(spec_fn)
    return spec_fn


def build_scene_model(entities: dict[str, MujocoEntityCfg], honor_flag: bool) -> mujoco.MjModel:
    """The mjlab scene build: every entity through the mocap wrapper, attached under ``name/``, plus a ground plane."""
    scene = mujoco.MjSpec()
    scene.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 1.0])
    for name, cfg in entities.items():
        spec = auto_wrap_fixed_base_mocap(_entity_spec_fn(cfg, honor_flag))()
        for key in list(spec.keys):
            spec.delete(key)
        scene.attach(spec, prefix=f"{name}/", frame=scene.worldbody.add_frame())
    return scene.compile()


def body_name(m: mujoco.MjModel, body: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or ""


def entity_bodies(m: mujoco.MjModel, name: str) -> np.ndarray:
    return np.array([b for b in range(m.nbody) if body_name(m, b).startswith(f"{name}/")], dtype=int)


def candidate_pairs(m: mujoco.MjModel) -> np.ndarray:
    """Geom pairs mujoco-warp hands to its broadphase, by its own filter (``io.py``, ``nxn_pairid != -2``)."""
    geom1, geom2 = np.triu_indices(m.ngeom, k=1)
    body1, body2 = m.geom_bodyid[geom1], m.geom_bodyid[geom2]
    weld1, weld2 = m.body_weldid[body1], m.body_weldid[body2]
    weld_parent1 = m.body_weldid[m.body_parentid[weld1]]
    weld_parent2 = m.body_weldid[m.body_parentid[weld2]]
    same_weld = weld1 == weld2
    filterparent = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    parent_child = filterparent & (weld1 != 0) & (weld2 != 0) & ((weld1 == weld_parent2) | (weld2 == weld_parent1))
    mask = (
        (m.geom_contype[geom1] & m.geom_conaffinity[geom2]) | (m.geom_contype[geom2] & m.geom_conaffinity[geom1])
    ) != 0
    excluded = np.isin((body1 << 16) + body2, m.exclude_signature)
    keep = mask & ~same_weld & ~parent_child & ~excluded
    return np.stack([geom1[keep], geom2[keep]], axis=1)


def pair_bodies(m: mujoco.MjModel, pairs: np.ndarray) -> np.ndarray:
    return m.geom_bodyid[pairs]


def split_pairs(m: mujoco.MjModel, pairs: np.ndarray, bodies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(intra, external)``: pairs with both geoms on the entity, and pairs with exactly one."""
    on = np.isin(pair_bodies(m, pairs), bodies)
    return pairs[on.all(axis=1)], pairs[on.sum(axis=1) == 1]


def body_pair_set(m: mujoco.MjModel, pairs: np.ndarray) -> set[tuple[str, str]]:
    out = set()
    for b1, b2 in pair_bodies(m, pairs):
        n1, n2 = body_name(m, b1), body_name(m, b2)
        out.add((n1, n2) if n1 <= n2 else (n2, n1))
    return out


def exclude_pairs(m: mujoco.MjModel) -> set[tuple[int, int]]:
    out = set()
    for sig in m.exclude_signature:
        b1, b2 = int(sig) >> 16, int(sig) & 0xFFFF
        out.add((min(b1, b2), max(b1, b2)))
    return out


def entity_joints(m: mujoco.MjModel, bodies: np.ndarray) -> list[int]:
    return [j for j in range(m.njnt) if m.jnt_bodyid[j] in set(bodies.tolist())]


def sample_qpos(m: mujoco.MjModel, joints: list[int], rng: np.random.Generator, mode: str, root_z: float) -> np.ndarray:
    """One joint configuration: ``mode`` is ``random`` (uniform in range), ``lower``, ``upper`` or ``alternate``."""
    qpos = m.qpos0.copy()
    sign = 1.0
    for j in joints:
        adr, jtype = m.jnt_qposadr[j], m.jnt_type[j]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            qpos[adr : adr + 7] = [0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0]
            continue
        if jtype == mujoco.mjtJoint.mjJNT_BALL:
            qpos[adr : adr + 4] = [1.0, 0.0, 0.0, 0.0]
            continue
        if m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
        elif jtype == mujoco.mjtJoint.mjJNT_HINGE:
            lo, hi = -math.pi, math.pi
        else:
            lo, hi = -0.1, 0.1
        if mode == "random":
            qpos[adr] = rng.uniform(lo, hi)
        elif mode == "lower":
            qpos[adr] = lo
        elif mode == "upper":
            qpos[adr] = hi
        else:
            qpos[adr] = hi if sign > 0 else lo
            sign = -sign
    return qpos


def contacts_at(m: mujoco.MjModel, d: mujoco.MjData, qpos: np.ndarray) -> np.ndarray:
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return d.contact.geom[: d.ncon].copy()


def check_preset(label: str, entities: dict[str, MujocoEntityCfg], samples: int, seed: int) -> bool:
    m_cfg = build_scene_model(entities, honor_flag=True)
    m_on = build_scene_model(entities, honor_flag=False)
    if m_cfg.ngeom != m_on.ngeom or m_cfg.nbody != m_on.nbody:
        raise RuntimeError(f"{label}: the exclude wrapper changed the geom/body count ({m_cfg.ngeom} vs {m_on.ngeom})")
    cand_cfg, cand_on = candidate_pairs(m_cfg), candidate_pairs(m_on)
    d_cfg, d_on = mujoco.MjData(m_cfg), mujoco.MjData(m_on)
    ok = True
    print(
        f"\n== {label}: nbody={m_cfg.nbody} ngeom={m_cfg.ngeom} nexclude={m_cfg.nexclude} (forced-on {m_on.nexclude})"
    )

    for name, cfg in entities.items():
        bodies = entity_bodies(m_cfg, name)
        real = np.array([b for b in bodies if body_name(m_cfg, b) != f"{name}/mocap_base"])
        intra_cfg, ext_cfg = split_pairs(m_cfg, cand_cfg, bodies)
        intra_on, ext_on = split_pairs(m_on, cand_on, bodies)
        flag = cfg.enable_self_collisions
        print(f"  [{name}] enable_self_collisions={flag} bodies={len(real)}")
        print(f"    intra-entity candidate geom pairs: configured={len(intra_cfg)} forced-on={len(intra_on)}")
        print(f"    external candidate geom pairs:     configured={len(ext_cfg)} forced-on={len(ext_on)}")
        if len(ext_cfg) != len(ext_on) or body_pair_set(m_cfg, ext_cfg) != body_pair_set(m_on, ext_on):
            print("    FAIL external candidate pairs differ between configured and forced-on")
            ok = False
        if flag:
            if len(intra_cfg) != len(intra_on):
                print("    FAIL flag on but intra-entity candidates differ from the forced-on model")
                ok = False
            if len(intra_on) == 0:
                print("    note: this entity has no self pairs even with the flag on (single body or asset masks)")
        else:
            if len(intra_cfg) != 0:
                print(f"    FAIL flag off but {len(intra_cfg)} intra-entity candidate pairs remain:")
                for n1, n2 in sorted(body_pair_set(m_cfg, intra_cfg))[:10]:
                    print(f"      {n1} <-> {n2}")
                ok = False
            want = {(min(a, b), max(a, b)) for a, b in itertools.combinations(real.tolist(), 2)}
            have = exclude_pairs(m_cfg)
            missing = want - have
            print(f"    exclude pairs covering this entity: {len(want & have)}/{len(want)} (missing {len(missing)})")
            if missing:
                print("    FAIL exclude list does not name every body pair")
                ok = False
            if len(intra_on) == 0:
                print("    note: forced-on model has no self pairs either, so the pair-table check is vacuous here")

        joints = entity_joints(m_cfg, bodies)
        rng = np.random.default_rng(seed)
        modes = ["lower", "upper", "alternate"] + ["random"] * samples
        self_cfg = self_on = 0
        worst_on = 0
        ext_mismatch = 0
        ext_hits = 0
        for mode in modes:
            q = sample_qpos(m_cfg, joints, rng, mode, _LIFT_Z)
            c_cfg = contacts_at(m_cfg, d_cfg, q)
            c_on = contacts_at(m_on, d_on, q)
            n_cfg = int(np.isin(pair_bodies(m_cfg, c_cfg), bodies).all(axis=1).sum()) if len(c_cfg) else 0
            n_on = int(np.isin(pair_bodies(m_on, c_on), bodies).all(axis=1).sum()) if len(c_on) else 0
            self_cfg += n_cfg > 0
            self_on += n_on > 0
            worst_on = max(worst_on, n_on)
            qg = sample_qpos(m_cfg, joints, rng, mode, _GROUND_Z)
            e_cfg = contacts_at(m_cfg, d_cfg, qg)
            e_on = contacts_at(m_on, d_on, qg)
            e_cfg = e_cfg[np.isin(pair_bodies(m_cfg, e_cfg), bodies).sum(axis=1) == 1] if len(e_cfg) else e_cfg
            e_on = e_on[np.isin(pair_bodies(m_on, e_on), bodies).sum(axis=1) == 1] if len(e_on) else e_on
            ext_hits += len(e_cfg) > 0
            if body_pair_set(m_cfg, e_cfg) != body_pair_set(m_on, e_on):
                ext_mismatch += 1
        n = len(modes)
        print(
            f"    lifted samples with self contacts: configured={self_cfg}/{n} forced-on={self_on}/{n} (max {worst_on} per sample)"
        )
        print(
            f"    grounded samples: external contact set equal in {n - ext_mismatch}/{n}, with contacts in {ext_hits}/{n}"
        )
        if not flag and self_cfg:
            print("    FAIL flag off but self contacts were generated")
            ok = False
        if flag and self_cfg != self_on:
            print("    FAIL flag on but self-contact samples differ from the forced-on model")
            ok = False
        if not flag and self_on == 0:
            print("    note: no sample self-collides even with the flag on; the dynamic check cannot discriminate here")
        if ext_mismatch:
            print("    FAIL external contacts differ between configured and forced-on")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=None, help="substring filter on the preset label")
    ap.add_argument("--samples", type=int, default=200, help="random joint configurations per entity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--specs", default=None, help="module exposing PRESETS for presets outside this package")
    args = ap.parse_args()

    extra = _extra_specs(args.specs)
    table = list(_PUBLIC) + (list(extra.PRESETS) if extra else [])
    seen: set[str] = set()
    failed: list[str] = []
    for label, loader, sims in table:
        if "mujoco" not in sims or loader in seen or (args.only and args.only not in label):
            continue
        seen.add(loader)
        cfgs = _load(loader, "mujoco", 2)
        entities = {k: v for k, v in cfgs.scene.entities.items() if isinstance(v, MujocoEntityCfg)}
        if not check_preset(label, entities, args.samples, args.seed):
            failed.append(label)
    print("\nRESULT:", "FAIL " + ", ".join(failed) if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
