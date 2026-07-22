"""Server-side self-check for the vendored Booster K1 asset + K1Config.

Standalone counterpart of ``check_k1_asset_parity.py``: instead of
comparing against the upstream pal-playground tree (which only exists on
the vendoring machine), every upstream truth is PINNED here as a
constant, captured from the parity diag run that gated the asset:

 - model sizes, total mass, compiler options
 - the home keyframe qpos (upstream scene keyframe)
 - the effective ground-contact parameters (mu=0.6, condim=3) that the
   foot-sphere priority/friction/condim triplet must produce against a
   default (priority-0) plane

Verifies, using only the vendored files:

 1. mesh inventory — referenced set == shipped set, all loadable
 2. model sizes + compiler options vs pinned constants
 3. K1Config vs the vendored XML — kp/kd/armature/effort per joint,
    default joint angles vs the pinned home keyframe, pattern coverage,
    referenced geom/body names
 4. collision-filter masks + truth table (spheres↔floor only,
    boxes↔boxes only)
 5. physics — PD hold at the home pose on a default plane: ground
    contact exclusively through the 8 foot spheres with mu=0.6 and
    condim=3, joints track the hold target, state stays finite

Everything measured is printed (verbose-first); any mismatch is
collected and the script exits non-zero at the end.

Run:

    PYTHONPATH=JaxRLWorld:JaxRLWorld-private \
        python -m rlworld.scripts.diag.check_k1_asset_selfcheck
"""

import re
import sys
from pathlib import Path

import mujoco
import numpy as np

from rlworld.rl.configs.robots.k1 import K1Config

_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets/K1"
_XML = _ASSET_DIR / "k1_mjx_feetonly.xml"

_FOOT_SPHERES = tuple(f"{side}_foot_{i}" for side in ("left", "right") for i in range(1, 5))
_FOOT_BOXES = ("left_foot", "right_foot")

# ── Upstream truths pinned by check_k1_asset_parity.py ────────────────
_EXPECTED_SIZES = {
    "nq": 29,
    "nv": 28,
    "nu": 22,
    "nbody": 24,
    "njnt": 23,
    "ngeom": 51,
    "nsite": 5,
    "nsensor": 13,
    "nmesh": 24,
}
_EXPECTED_TOTAL_MASS = 19.666  # kg (subtree mass of Trunk)
# Upstream scene "home" keyframe: base (xyz wxyz) + 22 joint angles.
_HOME_QPOS = np.array(
    [
        0.0,
        0.0,
        0.545,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # head
        0.0,
        -1.4,
        0.0,
        -0.4,  # left arm
        0.0,
        1.4,
        0.0,
        0.4,  # right arm
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,  # left leg
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,  # right leg
    ]
)
# Effective ground contact (winner = foot sphere) on a priority-0 plane.
_EXPECTED_CONTACT_MU = 0.6
_EXPECTED_CONTACT_CONDIM = 3

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


# ──────────────────────────────────────────────────────────────────────
section("1. Mesh inventory")
xml_text = _XML.read_text()
referenced = set(re.findall(r'<mesh name="[^"]+" file="([^"]+)"/>', xml_text))
shipped = {p.name for p in (_ASSET_DIR / "meshes").iterdir()}
print(f"  referenced {len(referenced)}, shipped {len(shipped)}")
check(
    "referenced mesh set == shipped mesh set",
    referenced == shipped,
    f"missing={referenced - shipped} extra={shipped - referenced}",
)

# ──────────────────────────────────────────────────────────────────────
section("2. Model sizes + compiler options (pinned)")
m = mujoco.MjModel.from_xml_path(str(_XML))
for attr, expected in _EXPECTED_SIZES.items():
    check(
        f"size {attr} == {expected}",
        getattr(m, attr) == expected,
        f"got {getattr(m, attr)}",
    )
total_mass = float(m.body_subtreemass[m.body("Trunk").id])
print(f"  total robot mass = {total_mass:.4f} kg")
check(
    "total mass (pinned)",
    abs(total_mass - _EXPECTED_TOTAL_MASS) < 1e-3,
    f"{total_mass:.4f} vs {_EXPECTED_TOTAL_MASS}",
)
check("timestep == 0.002", m.opt.timestep == 0.002)
check("integrator == Euler", m.opt.integrator == 0)
check("iterations == 3", m.opt.iterations == 3)
check("ls_iterations == 5", m.opt.ls_iterations == 5)
check(
    "eulerdamp disabled",
    bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
)

# ──────────────────────────────────────────────────────────────────────
section("3. K1Config vs vendored XML + pinned keyframe")
cfg = K1Config()


def resolve(d: dict, name: str, default: float = 0.0) -> float:
    hits = {v for p, v in d.items() if re.fullmatch(p, name)}
    if len(hits) > 1:
        check(f"unambiguous match for {name}", False, f"{hits}")
    return hits.pop() if hits else default


print(f"  {'joint':<24}{'kp':>6}{'kd':>6}{'armature':>9}{'effort':>7}{'default':>9}")
for i in range(1, m.njnt):
    jname = m.joint(i).name
    dof = m.jnt_dofadr[i]
    act = next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == i)
    kp_c = resolve(cfg.p_gains, jname)
    kd_c = resolve(cfg.d_gains, jname)
    arm_c = resolve(cfg.armature, jname)
    eff_c = resolve(cfg.effort_limits, jname)
    q0_c = resolve(cfg.default_joint_angles, jname)
    print(f"  {jname:<24}{kp_c:>6.0f}{kd_c:>6.0f}{arm_c:>9.4f}{eff_c:>7.0f}{q0_c:>9.2f}")
    check(f"{jname} kp", kp_c == m.actuator(act).gainprm[0])
    check(f"{jname} kd (joint damping)", kd_c == m.dof_damping[dof])
    check(f"{jname} armature", arm_c == m.dof_armature[dof])
    check(f"{jname} effort (jnt_actfrcrange)", eff_c == m.jnt_actfrcrange[i][1])
    check(
        f"{jname} default angle vs pinned keyframe",
        q0_c == _HOME_QPOS[m.jnt_qposadr[i]],
    )
    n_act = sum(bool(re.fullmatch(p, jname)) for p in cfg.actuated_dof_patterns)
    check(
        f"{jname} matched exactly once by actuated_dof_patterns",
        n_act == 1,
        f"n={n_act}",
    )

check("base_init_height vs pinned keyframe", cfg.base_init_height == _HOME_QPOS[2])
check(
    "get_action_offset == default_joint_angles",
    cfg.get_action_offset() == cfg.default_joint_angles,
)
check(
    "mjcf_path basename matches vendored XML",
    Path(cfg.mjcf_path).name == _XML.name and "assets/K1" in cfg.mjcf_path,
)
for b in cfg.foot_names + [cfg.trunk_body_name, cfg.base_link_name]:
    check(f"body '{b}' exists", m.body(b).id >= 0)
check("foot_geom_names == the 8 spheres", cfg.foot_geom_names == _FOOT_SPHERES)
check("foot_box_geom_names == the 2 boxes", cfg.foot_box_geom_names == _FOOT_BOXES)
for group, pats, expected in (
    ("hip", cfg.hip_joint_patterns, 4),
    ("knee", cfg.knee_joint_patterns, 2),
    ("ankle", cfg.ankle_joint_patterns, 4),
):
    n = sum(1 for i in range(1, m.njnt) for p in pats if re.fullmatch(p, m.joint(i).name))
    check(f"{group}_joint_patterns match {expected} joints", n == expected, f"n={n}")

# ──────────────────────────────────────────────────────────────────────
section("4. Collision filtering")
for g in _FOOT_SPHERES:
    gid = m.geom(g).id
    check(
        f"{g}: contype/conaffinity=2/1, priority=1, mu=0.6, condim=3",
        m.geom_contype[gid] == 2
        and m.geom_conaffinity[gid] == 1
        and m.geom_priority[gid] == 1
        and m.geom_friction[gid][0] == _EXPECTED_CONTACT_MU
        and m.geom_condim[gid] == _EXPECTED_CONTACT_CONDIM,
    )
for g in _FOOT_BOXES:
    gid = m.geom(g).id
    check(
        f"{g}: contype/conaffinity = 4/4",
        m.geom_contype[gid] == 4 and m.geom_conaffinity[gid] == 4,
    )


def collides(ct1: int, ca1: int, ct2: int, ca2: int) -> bool:
    return bool(ct1 & ca2) or bool(ct2 & ca1)


check("sphere collides with default floor (1/1)", collides(2, 1, 1, 1))
check("foot box does NOT collide with default floor", not collides(4, 4, 1, 1))
check("foot boxes collide with each other", collides(4, 4, 4, 4))
check("spheres do NOT collide with each other", not collides(2, 1, 2, 1))
check("visual/body geoms (0/0) collide with nothing", not collides(0, 0, 1, 1))

# ──────────────────────────────────────────────────────────────────────
section("5. Physics: PD hold at home pose on a default plane")
spec = mujoco.MjSpec.from_file(str(_XML))
floor_g = spec.worldbody.add_geom()
floor_g.name = "floor"
floor_g.type = mujoco.mjtGeom.mjGEOM_PLANE
floor_g.size = [0.0, 0.0, 0.01]
# MuJoCo defaults: contype=1 conaffinity=1 priority=0 friction=(1.0, ...),
# condim=3 — the foot-sphere triplet must win the combination.
scene = spec.compile()
d = mujoco.MjData(scene)
d.qpos[:] = _HOME_QPOS
default_pose = _HOME_QPOS[7:]
d.ctrl[:] = default_pose
mujoco.mj_forward(scene, d)

heights = []
touching, mus, dims = set(), set(), set()
for step in range(1500):  # 3 s at 2 ms
    mujoco.mj_step(scene, d)
    heights.append(float(d.qpos[2]))
    for c in range(d.ncon):
        g1, g2 = d.contact.geom[c]
        n1, n2 = scene.geom(g1).name, scene.geom(g2).name
        if "floor" in (n1, n2):
            touching.add(n2 if n1 == "floor" else n1)
            mus.add(float(d.contact.friction[c][0]))
            dims.add(int(d.contact.dim[c]))

trunk_zz = d.xmat[scene.body("Trunk").id].reshape(3, 3)[2, 2]
joint_err = np.abs(d.qpos[7:] - default_pose)
print(
    f"  base height: start {_HOME_QPOS[2]:.3f} → end {heights[-1]:.4f} "
    f"(min {min(heights):.4f}, max {max(heights):.4f})"
)
print(
    f"  trunk upvector_z after 3 s: {trunk_zz:.5f} " "(the source model cannot passively stand — falling is expected)"
)
print(f"  max |q - q_hold|: {joint_err.max():.4f} rad " f"(argmax {scene.joint(1 + int(joint_err.argmax())).name})")
print(f"  floor-touching geoms: {sorted(touching)}")
print(f"  contact mu set: {sorted(mus)}, condim set: {sorted(dims)}")
check(
    "ground contact only through the 8 foot spheres",
    touching <= set(_FOOT_SPHERES) and len(touching) >= 4,
    f"{sorted(touching)}",
)
check(
    "effective contact friction == 0.6 (foot priority wins)",
    mus == {_EXPECTED_CONTACT_MU},
    f"{sorted(mus)}",
)
check(
    "effective contact condim == 3",
    dims == {_EXPECTED_CONTACT_CONDIM},
    f"{sorted(dims)}",
)
# The source model cannot passively stand (soft ankle kp=10; the policy
# does the balancing — verified against upstream in the parity diag), so
# the pinned expected signature after 3 s is a FALLEN robot whose trunk
# hangs below the plane (only the feet collide with the floor).
check(
    "ends fallen, matching pinned upstream behaviour (upvector_z < 0)",
    trunk_zz < 0.0,
    f"{trunk_zz:.5f}",
)
# 0.25 rad allows the ground-loaded ankle of the fallen pose (upstream
# shows ~0.18); a limp or exploding PD would blow far past this.
check(
    "joints track the hold target (max err < 0.25 rad)",
    joint_err.max() < 0.25,
    f"{joint_err.max():.4f}",
)
check("state stays finite", bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()))

# ──────────────────────────────────────────────────────────────────────
section("Result")
if failures:
    print(f"  {len(failures)} FAILURE(S):")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)
print("  ALL CHECKS PASSED")
