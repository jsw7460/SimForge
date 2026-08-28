"""Exhaustive asset/config parity diag for the Booster K1 port (Phase 1).

Compares the vendored K1 asset + ``K1Config`` against the upstream
mujoco_playground K1 source tree and verifies:

 1. mesh inventory — every referenced mesh exists, byte-identical, no extras
 2. compiler options — timestep / integrator / iterations / eulerdamp
 3. model structure — joints, actuators, bodies (inertials), sites, sensors
 4. geom table — the ONLY intended delta is the 8 foot spheres
    (priority 0→1, sliding friction 1.0→0.6, condim 1→3); everything
    else identical. The triplet moves the upstream floor's winning
    contact parameters onto the feet so foot-side friction DR works
    against a default (priority-0) plane.
 5. K1Config values — kp/kd/effort per joint, default joint angles vs
    the upstream home keyframe, base height, pattern coverage,
    referenced geom/body names, collision-filter masks.  Armature is
    the one intentional config-over-XML override: it is checked
    against the Booster reference model ``assets/K1/K1_22dof.xml``
    instead (the feetonly MJCF flattens armature to a uniform 0.005)
 6. physics behaviour — PD hold at the home pose, run twice: the
    upstream scene (ground truth) and our robot on a default
    (priority-0, friction-1.0, condim-3) plane. Both runs must produce
    the same effective contact (mu=0.6, condim=3, spheres only) and the
    same trajectory (tight tolerance at 0.5 s, loose at 3 s)

Everything measured is printed (verbose-first); any mismatch is
collected and the script exits non-zero at the end.

This diag needs the upstream tree (``third_party/rl_frameworks/
pal-playground``), so it runs on the machine that vendors the asset —
NOT on the training server. It gates every change to the vendored
asset; the server-side counterpart with the upstream truths pinned as
constants is ``check_k1_asset_selfcheck.py``.

Run from anywhere (paths resolve from this file's location):

    PYTHONPATH=JaxRLWorld:JaxRLWorld-private \
        python -m rlworld.scripts.diag.k1.check_k1_asset_parity
"""

import hashlib
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

from rlworld.rl.configs.robots.k1 import K1Config

_SIMFORGE_ROOT = Path(__file__).resolve().parents[4]
_OURS_DIR = _SIMFORGE_ROOT / "JaxRLWorld/rlworld/assets/K1"
_PAL_DIR = _SIMFORGE_ROOT / "third_party/rl_frameworks/pal-playground/mujoco_playground/_src/locomotion/k1/xmls"

_FOOT_SPHERES = tuple(f"{side}_foot_{i}" for side in ("left", "right") for i in range(1, 5))

if not _PAL_DIR.exists():
    sys.exit(
        f"Upstream pal-playground tree not found at {_PAL_DIR}.\n"
        "This parity diag compares the vendored asset against the upstream "
        "source and therefore only runs on the machine that has third_party/. "
        "On the training server run check_k1_asset_selfcheck.py instead."
    )

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


# ──────────────────────────────────────────────────────────────────────
section("1. Mesh inventory")
pal_xml_text = (_PAL_DIR / "k1_mjx_feetonly.xml").read_text()
referenced = re.findall(r'<mesh name="[^"]+" file="([^"]+)"/>', pal_xml_text)
print(f"  upstream references {len(referenced)} meshes")
ours_meshes = sorted(p.name for p in (_OURS_DIR / "meshes").iterdir())
check(
    "mesh count matches references",
    len(ours_meshes) == len(set(referenced)),
    f"{len(ours_meshes)} files vs {len(set(referenced))} referenced",
)
check(
    "no extra meshes",
    set(ours_meshes) == set(referenced),
    f"extra={set(ours_meshes) - set(referenced)} missing={set(referenced) - set(ours_meshes)}",
)
for name in sorted(set(referenced)):
    ours_f = _OURS_DIR / "meshes" / name
    pal_f = _PAL_DIR / "meshes" / name
    same = (
        ours_f.exists() and hashlib.sha256(ours_f.read_bytes()).digest() == hashlib.sha256(pal_f.read_bytes()).digest()
    )
    if not same:
        check(f"mesh bytes: {name}", False, "missing or differs")
check("all mesh files byte-identical", not any("mesh bytes" in f for f in failures))

# ──────────────────────────────────────────────────────────────────────
section("2. Model load + compiler options")
ours = mujoco.MjModel.from_xml_path(str(_OURS_DIR / "k1_mjx_feetonly.xml"))
pal = mujoco.MjModel.from_xml_path(str(_PAL_DIR / "k1_mjx_feetonly.xml"))
pal_scene = mujoco.MjModel.from_xml_path(str(_PAL_DIR / "scene_mjx_feetonly_flat_terrain.xml"))
print(
    f"  ours: nq={ours.nq} nv={ours.nv} nu={ours.nu} nbody={ours.nbody} "
    f"ngeom={ours.ngeom} nsite={ours.nsite} nsensor={ours.nsensor}"
)
for attr in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsite", "nsensor", "nmesh"):
    check(
        f"size {attr}",
        getattr(ours, attr) == getattr(pal, attr),
        f"{getattr(ours, attr)} vs {getattr(pal, attr)}",
    )
check("timestep", ours.opt.timestep == pal.opt.timestep == 0.002)
check("integrator (Euler)", ours.opt.integrator == pal.opt.integrator == 0)
check("iterations", ours.opt.iterations == pal.opt.iterations == 3)
check("ls_iterations", ours.opt.ls_iterations == pal.opt.ls_iterations == 5)
check(
    "eulerdamp disabled",
    bool(ours.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    and ours.opt.disableflags == pal.opt.disableflags,
)

# ──────────────────────────────────────────────────────────────────────
section("3. Joints / actuators / bodies / sites / sensors")
print(f"  {'joint':<24}{'damping':>8}{'armature':>9}{'frictionloss':>13}" f"{'actfrc':>8}  range")
for i in range(ours.njnt):
    jo, jp = ours.joint(i), pal.joint(i)
    check(f"joint[{i}] name", jo.name == jp.name, f"{jo.name} vs {jp.name}")
    if ours.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
        dof = ours.jnt_dofadr[i]
        row = (
            f"  {jo.name:<24}{ours.dof_damping[dof]:>8.3f}{ours.dof_armature[dof]:>9.4f}"
            f"{ours.dof_frictionloss[dof]:>13.3f}{ours.jnt_actfrcrange[i][1]:>8.1f}"
            f"  [{ours.jnt_range[i][0]:.3f}, {ours.jnt_range[i][1]:.3f}]"
        )
        print(row)
        same = (
            ours.jnt_type[i] == pal.jnt_type[i]
            and np.array_equal(ours.jnt_axis[i], pal.jnt_axis[i])
            and np.array_equal(ours.jnt_range[i], pal.jnt_range[i])
            and np.array_equal(ours.jnt_actfrcrange[i], pal.jnt_actfrcrange[i])
            and ours.dof_damping[dof] == pal.dof_damping[pal.jnt_dofadr[i]]
            and ours.dof_armature[dof] == pal.dof_armature[pal.jnt_dofadr[i]]
            and ours.dof_frictionloss[dof] == pal.dof_frictionloss[pal.jnt_dofadr[i]]
        )
        check(f"joint[{i}] {jo.name} attrs", same)

for i in range(ours.nu):
    ao, ap = ours.actuator(i), pal.actuator(i)
    same = (
        ao.name == ap.name
        and np.array_equal(ao.gainprm, ap.gainprm)
        and np.array_equal(ao.biasprm, ap.biasprm)
        and np.array_equal(ao.ctrlrange, ap.ctrlrange)
        and ours.actuator_trnid[i, 0] == pal.actuator_trnid[i, 0]
    )
    check(f"actuator[{i}] {ao.name} (kp={ao.gainprm[0]:.0f})", same)

for i in range(ours.nbody):
    bo, bp = ours.body(i), pal.body(i)
    same = (
        bo.name == bp.name
        and np.array_equal(bo.mass, bp.mass)
        and np.array_equal(bo.ipos, bp.ipos)
        and np.array_equal(bo.iquat, bp.iquat)
        and np.array_equal(bo.inertia, bp.inertia)
        and np.array_equal(bo.pos, bp.pos)
        and np.array_equal(bo.quat, bp.quat)
    )
    check(f"body[{i}] {bo.name} (mass={float(bo.mass[0]):.3f})", same)
total_mass = float(ours.body_subtreemass[ours.body("Trunk").id])
print(f"  total robot mass = {total_mass:.4f} kg")
check(
    "total mass matches upstream",
    total_mass == float(pal.body_subtreemass[pal.body("Trunk").id]),
)

for i in range(ours.nsite):
    so, sp = ours.site(i), pal.site(i)
    check(
        f"site[{i}] {so.name}",
        so.name == sp.name and np.array_equal(so.pos, sp.pos) and ours.site_bodyid[i] == pal.site_bodyid[i],
    )

for i in range(ours.nsensor):
    check(
        f"sensor[{i}] {ours.sensor(i).name}",
        ours.sensor(i).name == pal.sensor(i).name
        and ours.sensor_type[i] == pal.sensor_type[i]
        and ours.sensor_objid[i] == pal.sensor_objid[i],
    )
# Contact sensors live in the upstream SCENE xml (not the robot xml) and
# are intentionally absent here — JaxRLWorld declares them via
# ContactSensorCfg at scene-composition time.
n_contact_sensors_pal_scene = sum(
    1 for i in range(pal_scene.nsensor) if pal_scene.sensor_type[i] == mujoco.mjtSensor.mjSENS_CONTACT
)
print(
    f"  upstream scene declares {n_contact_sensors_pal_scene} contact sensors " "(ported as ContactSensorCfg, not XML)"
)
check("upstream contact sensor count is 9", n_contact_sensors_pal_scene == 9)

# ──────────────────────────────────────────────────────────────────────
section("4. Geom table — intended delta only")
_GEOM_FIELDS = (
    "geom_type",
    "geom_contype",
    "geom_conaffinity",
    "geom_condim",
    "geom_priority",
    "geom_group",
)
_GEOM_VEC_FIELDS = (
    "geom_size",
    "geom_pos",
    "geom_quat",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
)
delta: dict[str, list[str]] = {}
for i in range(ours.ngeom):
    name = ours.geom(i).name or f"<unnamed #{i} in {ours.body(ours.geom_bodyid[i]).name}>"
    diffs = []
    for f in _GEOM_FIELDS:
        a, b = getattr(ours, f)[i], getattr(pal, f)[i]
        if a != b:
            diffs.append(f"{f}: {b} -> {a}")
    for f in _GEOM_VEC_FIELDS:
        a, b = getattr(ours, f)[i], getattr(pal, f)[i]
        if not np.array_equal(a, b):
            diffs.append(f"{f}: {b} -> {a}")
    if diffs:
        delta[name] = diffs
        for d in diffs:
            print(f"  delta {name}: {d}")
check(
    "delta geom set == 8 foot spheres",
    set(delta) == set(_FOOT_SPHERES),
    f"unexpected: {set(delta) ^ set(_FOOT_SPHERES)}",
)
for name in _FOOT_SPHERES:
    d = delta.get(name, [])
    ok = (
        len(d) == 3
        and any(x.startswith("geom_priority") for x in d)
        and any(x.startswith("geom_friction") for x in d)
        and any(x.startswith("geom_condim") for x in d)
    )
    check(f"{name} delta is exactly priority+friction+condim", ok, str(d))
    gid = ours.geom(name).id
    check(
        f"{name} priority=1, sliding friction=0.6, condim=3",
        ours.geom_priority[gid] == 1
        and ours.geom_friction[gid][0] == 0.6
        and ours.geom_condim[gid] == 3
        and np.array_equal(ours.geom_friction[gid][1:], pal.geom_friction[pal.geom(name).id][1:]),
    )
# The upstream floor's winning contact parameters, which the sphere
# triplet must reproduce exactly (friction vector, condim, solref/solimp).
pal_floor = pal_scene.geom("floor").id
check(
    "upstream floor: priority=1, friction=0.6, condim=3 (the values we moved)",
    pal_scene.geom_priority[pal_floor] == 1
    and pal_scene.geom_friction[pal_floor][0] == 0.6
    and pal_scene.geom_condim[pal_floor] == 3,
)
gid0 = ours.geom(_FOOT_SPHERES[0]).id
check(
    "sphere friction vector == upstream floor friction vector",
    np.array_equal(ours.geom_friction[gid0], pal_scene.geom_friction[pal_floor]),
)
check(
    "sphere solref/solimp == upstream floor solref/solimp",
    np.array_equal(ours.geom_solref[gid0], pal_scene.geom_solref[pal_floor])
    and np.array_equal(ours.geom_solimp[gid0], pal_scene.geom_solimp[pal_floor]),
)

# ──────────────────────────────────────────────────────────────────────
section("5. K1Config cross-check")
cfg = K1Config()

# Booster reference model: source of truth for per-joint armature (the
# vendored feetonly XML carries a flattened uniform 0.005 instead).
ref = mujoco.MjModel.from_xml_path(str(_OURS_DIR / "K1_22dof.xml"))
check(
    "mjcf_path resolves to the vendored asset",
    (_SIMFORGE_ROOT / cfg.mjcf_path.lstrip("./")).resolve() == (_OURS_DIR / "k1_mjx_feetonly.xml").resolve(),
)


def resolve(d: dict, name: str, default: float = 0.0) -> float:
    hits = {v for p, v in d.items() if re.fullmatch(p, name)}
    if len(hits) > 1:
        check(f"unambiguous match for {name}", False, f"{hits}")
    return hits.pop() if hits else default


print(f"  {'joint':<24}{'kp':>6}{'kd':>6}{'armature':>9}{'effort':>7}{'default':>9}")
key_qpos = pal_scene.keyframe("home").qpos
for i in range(1, ours.njnt):
    jname = ours.joint(i).name
    dof = ours.jnt_dofadr[i]
    act = next(a for a in range(ours.nu) if ours.actuator_trnid[a, 0] == i)
    kp_c = resolve(cfg.p_gains, jname)
    kd_c = resolve(cfg.d_gains, jname)
    arm_c = resolve(cfg.armature, jname)
    eff_c = resolve(cfg.effort_limits, jname)
    q0_c = resolve(cfg.default_joint_angles, jname)
    q0_x = key_qpos[pal_scene.jnt_qposadr[pal_scene.joint(jname).id]]
    print(f"  {jname:<24}{kp_c:>6.0f}{kd_c:>6.0f}{arm_c:>9.4f}{eff_c:>7.0f}{q0_c:>9.2f}")
    check(f"{jname} kp", kp_c == ours.actuator(act).gainprm[0])
    check(f"{jname} kd (joint damping)", kd_c == ours.dof_damping[dof])
    check(
        f"{jname} armature (Booster reference model)",
        arm_c == ref.dof_armature[ref.jnt_dofadr[ref.joint(jname).id]],
    )
    check(f"{jname} effort (jnt_actfrcrange)", eff_c == ours.jnt_actfrcrange[i][1])
    check(f"{jname} default angle vs home keyframe", q0_c == q0_x)
    n_act = sum(bool(re.fullmatch(p, jname)) for p in cfg.actuated_dof_patterns)
    check(
        f"{jname} matched exactly once by actuated_dof_patterns",
        n_act == 1,
        f"n={n_act}",
    )

check(
    "base_init_height vs keyframe",
    cfg.base_init_height == key_qpos[2],
    f"{cfg.base_init_height} vs {key_qpos[2]}",
)
check("keyframe base orientation is identity", np.array_equal(key_qpos[3:7], [1, 0, 0, 0]))
check(
    "get_action_offset == default_joint_angles",
    cfg.get_action_offset() == cfg.default_joint_angles,
)
check(
    "base_link_name exists",
    cfg.base_link_name == "Trunk" and ours.body("Trunk").id >= 0,
)
for b in cfg.foot_names + [cfg.trunk_body_name]:
    check(f"body '{b}' exists", ours.body(b).id >= 0)
for g in cfg.foot_geom_names:
    gid = ours.geom(g).id
    check(
        f"foot sphere '{g}' contype/conaffinity = 2/1",
        ours.geom_contype[gid] == 2 and ours.geom_conaffinity[gid] == 1,
    )
for g in cfg.foot_box_geom_names:
    gid = ours.geom(g).id
    check(
        f"foot box '{g}' contype/conaffinity = 4/4",
        ours.geom_contype[gid] == 4 and ours.geom_conaffinity[gid] == 4,
    )
for group, pats in (
    ("hip", cfg.hip_joint_patterns),
    ("knee", cfg.knee_joint_patterns),
    ("ankle", cfg.ankle_joint_patterns),
):
    n = sum(1 for i in range(1, ours.njnt) for p in pats if re.fullmatch(p, ours.joint(i).name))
    expected = {"hip": 4, "knee": 2, "ankle": 4}[group]
    check(f"{group}_joint_patterns match {expected} joints", n == expected, f"n={n}")


def collides(m: mujoco.MjModel, g1: int, c1: int, g2: int, c2: int) -> bool:
    return bool(g1 & c2) or bool(g2 & c1)


# Collision-filter truth table against a default floor (contype/conaffinity
# 1/1 — what a JaxRLWorld terrain plane gets) and the upstream floor (1/2).
for floor_ct, floor_ca, label in ((1, 1, "default floor"), (1, 2, "upstream floor")):
    check(f"sphere collides with {label}", collides(ours, 2, 1, floor_ct, floor_ca))
    check(
        f"foot box does NOT collide with {label}",
        not collides(ours, 4, 4, floor_ct, floor_ca),
    )
check("foot boxes collide with each other", collides(ours, 4, 4, 4, 4))
check("spheres do NOT collide with each other", not collides(ours, 2, 1, 2, 1))
check("visual/body geoms (0/0) collide with nothing", not collides(ours, 0, 0, 1, 1))

# ──────────────────────────────────────────────────────────────────────
section("6. Physics: PD hold — upstream scene vs our robot on a default plane")
default_pose = np.array([resolve(cfg.default_joint_angles, ours.joint(i).name) for i in range(1, ours.njnt)])
check(
    "config-derived qpos == upstream home keyframe",
    np.array_equal(
        np.concatenate([[0, 0, cfg.base_init_height, 1, 0, 0, 0], default_pose]),
        key_qpos,
    ),
)

spec = mujoco.MjSpec.from_file(str(_OURS_DIR / "k1_mjx_feetonly.xml"))
floor_g = spec.worldbody.add_geom()
floor_g.name = "floor"
floor_g.type = mujoco.mjtGeom.mjGEOM_PLANE
floor_g.size = [0.0, 0.0, 0.01]
# Deliberately a JaxRLWorld-style default plane: contype=1 conaffinity=1,
# priority=0, friction=(1.0, ...), condim=3. The foot spheres must win.
ours_scene = spec.compile()

_N_STEPS = 1500  # 3 s at 2 ms


def rollout(m: mujoco.MjModel) -> dict:
    d = mujoco.MjData(m)
    d.qpos[:] = key_qpos
    d.ctrl[:] = default_pose
    mujoco.mj_forward(m, d)
    qpos_hist = np.empty((_N_STEPS, m.nq))
    touching, mus, dims = set(), set(), set()
    for step in range(_N_STEPS):
        mujoco.mj_step(m, d)
        qpos_hist[step] = d.qpos
        for c in range(d.ncon):
            g1, g2 = d.contact.geom[c]
            n1, n2 = m.geom(g1).name, m.geom(g2).name
            if "floor" in (n1, n2):
                touching.add(n2 if n1 == "floor" else n1)
                mus.add(float(d.contact.friction[c][0]))
                dims.add(int(d.contact.dim[c]))
    trunk_zz = d.xmat[m.body("Trunk").id].reshape(3, 3)[2, 2]
    return {
        "qpos": qpos_hist,
        "touching": touching,
        "mus": mus,
        "dims": dims,
        "upz": trunk_zz,
    }


ref = rollout(pal_scene)
got = rollout(ours_scene)
for label, r in (("upstream", ref), ("ours", got)):
    h = r["qpos"][:, 2]
    print(
        f"  {label:>8}: height end {h[-1]:.4f} (min {h.min():.4f} max {h.max():.4f}) "
        f"upvector_z {r['upz']:.5f} contact mu {sorted(r['mus'])} "
        f"condim {sorted(r['dims'])}"
    )
    print(f"           floor-touching geoms: {sorted(r['touching'])}")

for label, r in (("upstream", ref), ("ours", got)):
    check(
        f"{label}: ground contact only through the 8 foot spheres",
        r["touching"] <= set(_FOOT_SPHERES) and len(r["touching"]) >= 4,
        f"{sorted(r['touching'])}",
    )
    check(
        f"{label}: effective contact friction == 0.6",
        r["mus"] == {0.6},
        f"{sorted(r['mus'])}",
    )
    check(
        f"{label}: effective contact condim == 3",
        r["dims"] == {3},
        f"{sorted(r['dims'])}",
    )

# Trajectory parity. Contact-rich dynamics amplify float noise, so the
# tolerance is tight early and loose at the horizon; the point is that
# the two models are the same mechanism, not bit-identical solvers.
for t_s, atol in ((0.1, 1e-6), (0.5, 1e-3), (3.0, 5e-2)):
    idx = int(t_s / 0.002) - 1
    err = float(np.abs(ref["qpos"][idx] - got["qpos"][idx]).max())
    check(
        f"trajectory parity at {t_s:.1f} s (max |dqpos| < {atol:g})",
        err < atol,
        f"err={err:.3e}",
    )
check(
    "same final height (|dz| < 0.01)",
    abs(ref["qpos"][-1, 2] - got["qpos"][-1, 2]) < 0.01,
    f"{ref['qpos'][-1, 2]:.4f} vs {got['qpos'][-1, 2]:.4f}",
)
check(
    "same uprightness (|d upvector_z| < 0.02)",
    abs(ref["upz"] - got["upz"]) < 0.02,
    f"{ref['upz']:.5f} vs {got['upz']:.5f}",
)

# ──────────────────────────────────────────────────────────────────────
section("Result")
if failures:
    print(f"  {len(failures)} FAILURE(S):")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)
print("  ALL CHECKS PASSED")
