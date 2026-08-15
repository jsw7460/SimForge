"""Multi-entity rigid-object (RigidObjectCfg) verification across the three backends.

Builds a scene with the Go2 robot plus two passive rigid objects — a
free-floating cube and a FIXED table — and checks every API a manipulation
task will lean on, at three layers.

**Env layer.** Both objects load into ``scene.rigid_objects``, read back
through ``env.get_rigid_object_data``, and expose no joint API. The free cube
is placed by a reset event.

**MDP layer.** Terms address entities by NAME (``asset_cfg``) and cannot know
which registry a name lives in, so they read through the polymorphic
``World.get_entity_data``. Without that a task cannot observe or reward the
object it manipulates. These checks also pin the failure modes: a joint read
on a passive body must surface as a missing attribute (the RigidObjectData /
RobotData split working), and ``get_robot_data`` must stay
articulation-only on all three backends.

**Fixed-base entity.** ``floating=False`` — the shape every workbench, tank
and machine frame in a manipulation scene takes. A welded body has no root
joint, so its pose is not in ``qpos``; each backend reaches it differently:

* mjlab wraps the entity in a ``mocap_base`` body, whose pose lives in
  ``data.mocap_pos`` — per-environment state.
* Newton loads it as a *kinematic* free body (``BodyFlags.KINEMATIC``): a real
  root joint, pinned by a huge armature, so it is placed like any free body.
  Same modelling IsaacLab uses for its manipulation tables
  (``RigidObjectCfg`` + ``kinematic_enabled=True``).
* Genesis writes a fixed link's pose straight into ``dyn_state.links.pos``.

The contract the checks below pin down, identical on all three:

1. ``init_state.pos`` places the entity at build time — one pose shared by
   every environment.
2. A reset event places it **per environment**, which is the only correct path
   once ``env_origins`` is non-zero, and the only way to randomize it.
3. Writing a root *velocity* raises: a welded body has no velocity state.

Everything else a passive body must do — load, read back finite and
well-formed, hold still under stepping, catch a falling object with its
collision geometry, agree across its body/CoM/selector reads — is checked too.

**Contact matching.** A manipulation task asks "did the tool touch the
workpiece", which no existing sensor has ever expressed: every one of them
watches robot-vs-terrain. The scene therefore carries a three-body gripper
fixture, and the same primary is matched against its counterpart three ways —
``mode="entity"`` (any part of the tool), one named jaw, and both jaws as
primaries. With the cube resting on one jaw the answers must be
``entity=True, left=True, right=False``; anything else means a backend widened
a named body into its entity or narrowed an entity to its first body. A final
probe, in its own process, builds a scene whose secondary names several bodies:
Genesis and Newton must accept it, MuJoCo must refuse at build time, because
its contact sensor carries a single reference and would otherwise watch one jaw
and call the other untouched.

**Protocol sweep.** Reading back *a* number proves nothing about whether it
is the RIGHT number, so the last section walks the entire
:class:`RigidObjectData` surface — every root property, both body-frame
conversions, projected gravity, heading, the CoM/link-origin split, all six
``body_*_all`` arrays, and the four ways to address a body — and cross-checks
each read against an independently computed value. It runs twice: on the cube
thrown into free flight, tilted and spinning, so the rotations and velocity
transforms cannot pass by being zero; and on the welded table at rest.

Run (GPU box) — one command covers all three backends and cross-compares::

    jaxpy -m rlworld.scripts.diag.rigid_object_smoke

Single backend (no cross-sim comparison)::

    jaxpy -m rlworld.scripts.diag.rigid_object_smoke --sim newton
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

# Diags never log to wandb; set before any rlworld import so the logger's
# wandb.init() becomes a no-op. (The manager tables, policy parameter dump and
# warp module-load lines are off framework-wide — see
# rlworld.rl.utils.verbosity.)
os.environ.setdefault("WANDB_MODE", "disabled")

import torch

from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.configs.scene import RigidObjectCfg
from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from rlworld.rl.configs.scene.unified_entity_config import InitialStateCfg
from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from rlworld.rl.envs.mdp.events import common as common_ef
from rlworld.rl.envs.mdp.observations.common import proprioception as obs_common
from rlworld.rl.runners import BaseRunner
from rlworld.rl.utils.quat_utils import (
    quat_error_magnitude_wxyz,
    quat_from_euler_xyz_wxyz,
    quat_rotate_inverse_wxyz,
    quat_rotate_wxyz,
)

_MODULE = "rlworld.scripts.diag.rigid_object_smoke"
_SIMS = ("genesis", "newton", "mujoco")

CUBE_SPAWN = (0.3, 0.0, 0.6)
CUBE_HALF = 0.03  # half the 0.06 box
CUBE_MASS = 0.2  # matches CUBE_URDF; the contact checks predict its weight

# Far from the robot (pinned to the origin below) so the two never interact and
# the table's own numbers stay clean.
TABLE_SPAWN = (2.0, 0.0, 0.2)
# Declared explicitly (not left to the default) so the spawn-orientation
# check below compares against something the scene really stated.
TABLE_ROT = (1.0, 0.0, 0.0, 0.0)
TABLE_HALF_Z = 0.2  # half the 0.4 box height
TABLE_HALF_XY = 0.6  # half the 1.2 box footprint

# A three-body fixture: a base plate carrying two jaws. Everything the
# contact-matching rules need is here — an entity with several bodies, of which
# only some touch a given object — and it is the shape of the real task (a tool
# whose two jaws must BOTH touch the workpiece to count as a grasp).
#
# The jaws are on HINGES, not welds, for the same reason the real tool has
# them: a welded part is one rigid body, and Newton's MuJoCo conversion folds
# such a part into its parent, so a contact on a jaw gets reported against the
# base and the jaw's name stops meaning anything. A joint keeps the bodies
# distinct on every backend. The hinges are clamped to +-0.001 rad so the
# fixture's geometry stays exact while remaining articulated. A non-zero
# ``effort`` is required: Newton maps it to MuJoCo's actuator force range, and
# zero would make that range empty, which MuJoCo rejects at compile time.
#
# Geometry, in the gripper's frame: the base spans z in [-0.01, 0.01]; each jaw
# sits on it, spanning z in [0.01, 0.07]. Jaw centres are at y = +-0.048 with
# half-width 0.02, so the inner faces are at y = +-0.028 and the gap between
# them is 0.056 — narrower than the 0.06 cube, so a cube placed in the gap
# interferes with BOTH jaws by 2 mm.
GRIPPER_SPAWN = (-2.0, -2.0, 0.5)
GRIPPER_BASE_HALF_Z = 0.01
GRIPPER_JAW_HALF_Z = 0.03
GRIPPER_JAW_Y = 0.048
# Top of a jaw, in the gripper's frame.
GRIPPER_JAW_TOP = GRIPPER_BASE_HALF_Z + 2 * GRIPPER_JAW_HALF_Z

GRIPPER_URDF = """<?xml version="1.0"?>
<robot name="gripper">
  <link name="grip_base">
    <inertial><origin xyz="0 0 0"/><mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    <visual><origin xyz="0 0 0"/><geometry><box size="0.10 0.16 0.02"/></geometry></visual>
    <collision><origin xyz="0 0 0"/><geometry><box size="0.10 0.16 0.02"/></geometry></collision>
  </link>
  <link name="jaw_left">
    <inertial><origin xyz="0 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><origin xyz="0 0 0"/><geometry><box size="0.04 0.04 0.06"/></geometry></visual>
    <collision><origin xyz="0 0 0"/><geometry><box size="0.04 0.04 0.06"/></geometry></collision>
  </link>
  <link name="jaw_right">
    <inertial><origin xyz="0 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><origin xyz="0 0 0"/><geometry><box size="0.04 0.04 0.06"/></geometry></visual>
    <collision><origin xyz="0 0 0"/><geometry><box size="0.04 0.04 0.06"/></geometry></collision>
  </link>
  <joint name="joint_jaw_left" type="revolute">
    <parent link="grip_base"/><child link="jaw_left"/>
    <origin xyz="0 0.048 0.04"/><axis xyz="1 0 0"/>
    <limit lower="-0.001" upper="0.001" effort="10" velocity="1"/>
  </joint>
  <joint name="joint_jaw_right" type="revolute">
    <parent link="grip_base"/><child link="jaw_right"/>
    <origin xyz="0 -0.048 0.04"/><axis xyz="1 0 0"/>
    <limit lower="-0.001" upper="0.001" effort="10" velocity="1"/>
  </joint>
</robot>
"""

CUBE_URDF = """<?xml version="1.0"?>
<robot name="cube">
  <link name="cube">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.2"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual><origin xyz="0 0 0"/><geometry><box size="0.06 0.06 0.06"/></geometry></visual>
    <collision><origin xyz="0 0 0"/><geometry><box size="0.06 0.06 0.06"/></geometry></collision>
  </link>
</robot>
"""

TABLE_URDF = """<?xml version="1.0"?>
<robot name="table">
  <link name="table">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="50.0"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
    <visual><origin xyz="0 0 0"/><geometry><box size="1.2 1.2 0.4"/></geometry></visual>
    <collision><origin xyz="0 0 0"/><geometry><box size="1.2 1.2 0.4"/></geometry></collision>
  </link>
</robot>
"""


def _fmt3(v) -> str:
    return "[" + ", ".join(f"{float(x):+.5f}" for x in v) + "]"


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def _gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _step_n(env, zeros: torch.Tensor, n: int) -> bool:
    """Step ``n`` times; return True if any environment reset along the way.

    The robot is driven with a zero action and may eventually topple. That
    fires a termination, and the reset teleports the cube back to CUBE_SPAWN
    via the reset_cube event — which would silently invalidate any measurement
    taken after the loop. Callers check the flag instead of trusting the
    numbers.
    """
    any_reset = False
    for _ in range(n):
        _obs, _rew, terminated, truncated, _extras = env.step(zeros)
        any_reset |= bool((terminated | truncated).any())
    return any_reset


# ══════════════════════════════════════════════════════════════════════════
# Full RigidObjectData protocol sweep
# ══════════════════════════════════════════════════════════════════════════

# Direct tensor comparisons (same quantity read two ways).
TOL = 1e-4
# Comparisons that route a ~1 m/s vector through a quaternion rotation in
# float32, so they carry a few more ulps of error.
ROT_TOL = 1e-3


def _protocol_sweep(env, name: str, results: dict, measured: dict, tag: str) -> None:
    """Exercise every ``RigidObjectData`` member of *name* and cross-check it.

    Every assertion here is a convention that must hold on all three backends
    regardless of pose or motion:

    * declared shapes and finiteness,
    * unit-norm quaternion,
    * body-frame velocity == ``R^T`` (world-frame velocity),
    * ``projected_gravity_b`` == ``R^T [0,0,-1]``, unit norm,
    * ``heading_w`` == the yaw of the rotated forward axis (exact while
      ``|pitch| < 90 deg``, which the caller guarantees),
    * root reads == the named body's reads for a single-link entity,
    * CoM reads == link-origin reads for a box whose inertial origin is its
      geometric centre,
    * the four addressing routes (``*_all`` / by-name / by-id / single-index)
      returning the same numbers.

    Results are keyed ``<tag>_*`` so the free cube in flight and the welded
    table at rest occupy separate rows in the verdict.

    Args:
        env: The world under test.
        name: Scene-entity name to read.
        results: Hard PASS/FAIL sink.
        measured: REPORTED sink (unsupported/optional API surface).
        tag: Prefix for the result keys.
    """
    d = env.get_entity_data(name)
    n = env.num_envs

    pos = d.root_link_pos_w
    quat = d.root_link_quat_w
    lin_w = d.root_link_lin_vel_w
    ang_w = d.root_link_ang_vel_w
    lin_b = d.root_link_lin_vel_b
    ang_b = d.root_link_ang_vel_b
    com_pos = d.root_com_pos_w
    com_lin_w = d.root_com_lin_vel_w
    com_lin_b = d.root_com_lin_vel_b
    grav_b = d.projected_gravity_b
    heading = d.heading_w

    print(f"[{tag}] root_link_pos_w  = {_fmt3(pos[0])}")
    print(f"[{tag}] root_link_quat_w = {_fmt3(quat[0])}")
    print(f"[{tag}] root_link_lin_vel_w = {_fmt3(lin_w[0])}   _b = {_fmt3(lin_b[0])}")
    print(f"[{tag}] root_link_ang_vel_w = {_fmt3(ang_w[0])}   _b = {_fmt3(ang_b[0])}")
    print(f"[{tag}] root_com_pos_w      = {_fmt3(com_pos[0])}")
    print(f"[{tag}] root_com_lin_vel_w  = {_fmt3(com_lin_w[0])}   _b = {_fmt3(com_lin_b[0])}")
    print(f"[{tag}] projected_gravity_b = {_fmt3(grav_b[0])}   heading_w = {float(heading[0]):+.5f}")

    # ── shapes / finiteness ──────────────────────────────────────────────
    vec3 = (pos, lin_w, ang_w, lin_b, ang_b, com_pos, com_lin_w, com_lin_b, grav_b)
    shapes_ok = all(t.shape == (n, 3) for t in vec3) and quat.shape == (n, 4) and heading.shape == (n,)
    finite_ok = all(_finite(t) for t in vec3) and _finite(quat) and _finite(heading)
    print(f"[{tag}] shapes ok: {shapes_ok}   all finite: {finite_ok}")
    results[f"{tag}_root_shapes"] = shapes_ok
    results[f"{tag}_root_finite"] = finite_ok

    qnorm = float(torch.linalg.norm(quat, dim=-1).sub(1.0).abs().max())
    print(f"[{tag}] max||q| - 1| = {qnorm:.3e}")
    results[f"{tag}_quat_unit"] = qnorm < 1e-4

    # ── frame transforms ─────────────────────────────────────────────────
    lin_b_gap = _gap(lin_b, quat_rotate_inverse_wxyz(quat, lin_w))
    ang_b_gap = _gap(ang_b, quat_rotate_inverse_wxyz(quat, ang_w))
    com_b_gap = _gap(com_lin_b, quat_rotate_inverse_wxyz(quat, com_lin_w))
    print(f"[{tag}] |lin_vel_b - R^T lin_vel_w| = {lin_b_gap:.3e}")
    print(f"[{tag}] |ang_vel_b - R^T ang_vel_w| = {ang_b_gap:.3e}")
    print(f"[{tag}] |com_lin_vel_b - R^T com_lin_vel_w| = {com_b_gap:.3e}")
    results[f"{tag}_lin_vel_b_transform"] = lin_b_gap < ROT_TOL
    results[f"{tag}_ang_vel_b_transform"] = ang_b_gap < ROT_TOL
    results[f"{tag}_com_lin_vel_b_transform"] = com_b_gap < ROT_TOL

    g_world = torch.tensor([[0.0, 0.0, -1.0]], device=env.device).expand(n, 3)
    grav_gap = _gap(grav_b, quat_rotate_inverse_wxyz(quat, g_world))
    grav_norm = float(torch.linalg.norm(grav_b, dim=-1).sub(1.0).abs().max())
    print(f"[{tag}] |projected_gravity_b - R^T [0,0,-1]| = {grav_gap:.3e}   max||g_b|-1| = {grav_norm:.3e}")
    results[f"{tag}_projected_gravity"] = grav_gap < ROT_TOL and grav_norm < 1e-3

    # Yaw of the rotated forward axis. Identical to the ZYX yaw whenever
    # |pitch| < 90 deg, and derived without reusing the euler helper the
    # backends themselves call.
    fwd = quat_rotate_wxyz(quat, torch.tensor([[1.0, 0.0, 0.0]], device=env.device).expand(n, 3))
    heading_alt = torch.atan2(fwd[:, 1], fwd[:, 0])
    heading_gap = float((torch.remainder(heading - heading_alt + torch.pi, 2 * torch.pi) - torch.pi).abs().max())
    print(f"[{tag}] |heading_w - atan2(fwd_y, fwd_x)| = {heading_gap:.3e}")
    results[f"{tag}_heading"] = heading_gap < ROT_TOL

    # ── CoM vs link origin (inertial origin is the box centre) ───────────
    com_pos_gap = _gap(com_pos, pos)
    com_vel_gap = _gap(com_lin_w, lin_w)
    print(f"[{tag}] |root_com_pos_w - root_link_pos_w| = {com_pos_gap:.3e} (CoM at box centre -> ~0)")
    print(f"[{tag}] |root_com_lin_vel_w - root_link_lin_vel_w| = {com_vel_gap:.3e} (zero CoM offset -> ~0)")
    results[f"{tag}_com_pos_matches_link"] = com_pos_gap < TOL
    results[f"{tag}_com_vel_matches_link"] = com_vel_gap < ROT_TOL

    # ── body-level reads ─────────────────────────────────────────────────
    bpos = d.body_pos_w_all
    bquat = d.body_quat_w_all
    blin = d.body_lin_vel_w_all
    bang = d.body_ang_vel_w_all
    bcom = d.body_com_pos_w_all
    bcomv = d.body_com_lin_vel_w_all
    nb = int(bpos.shape[1])
    body_shapes_ok = (
        bpos.shape == (n, nb, 3)
        and bquat.shape == (n, nb, 4)
        and blin.shape == (n, nb, 3)
        and bang.shape == (n, nb, 3)
        and bcom.shape == (n, nb, 3)
        and bcomv.shape == (n, nb, 3)
    )
    body_finite_ok = all(_finite(t) for t in (bpos, bquat, blin, bang, bcom, bcomv))
    print(f"[{tag}] body count = {nb}   body shapes ok: {body_shapes_ok}   finite: {body_finite_ok}")
    results[f"{tag}_body_shapes"] = body_shapes_ok
    results[f"{tag}_body_finite"] = body_finite_ok
    measured[f"{tag}_body_count"] = nb

    # Index by NAME, never by position. Newton's ``body_*_all`` spans every
    # body in the world (the robot's included) because it views the flat
    # ``state.body_q``, whereas Genesis and mjlab return only this entity's
    # bodies. ``find_body_index`` is the per-backend bridge, and
    # ``body_count`` below is reported precisely because it differs.
    idx = d.find_body_index(name)
    quat_err = float(quat_error_magnitude_wxyz(bquat[:, idx, :], quat).abs().max())
    gaps = {
        "pos": _gap(bpos[:, idx, :], pos),
        "lin_vel": _gap(blin[:, idx, :], lin_w),
        "ang_vel": _gap(bang[:, idx, :], ang_w),
        "com_pos": _gap(bcom[:, idx, :], com_pos),
        "com_lin_vel": _gap(bcomv[:, idx, :], com_lin_w),
    }
    print(f"[{tag}] find_body_index({name!r}) = {idx}")
    print(f"[{tag}] body[{idx}] vs root:  " + "  ".join(f"{k}={v:.3e}" for k, v in gaps.items()))
    print(f"[{tag}] body[{idx}] quat vs root quat: angular error = {quat_err:.3e} rad")
    results[f"{tag}_named_body_matches_root"] = max(gaps.values()) < ROT_TOL and quat_err < 1e-3

    # ── the four addressing routes must agree ────────────────────────────
    ids = torch.tensor([idx], device=env.device, dtype=torch.long)
    by_name_pos = d.body_pos_w([name])
    by_ids_pos = d.body_pos_w_by_ids(ids)
    by_name_vel = d.body_lin_vel_w([name])
    by_ids_vel = d.body_lin_vel_w_by_ids(ids)
    single_ang = d.body_ang_vel_w(idx)
    route_gaps = {
        "body_pos_w(names)": _gap(by_name_pos[:, 0, :], bpos[:, idx, :]),
        "body_pos_w_by_ids": _gap(by_ids_pos[:, 0, :], bpos[:, idx, :]),
        "body_lin_vel_w(names)": _gap(by_name_vel[:, 0, :], blin[:, idx, :]),
        "body_lin_vel_w_by_ids": _gap(by_ids_vel[:, 0, :], blin[:, idx, :]),
        "body_ang_vel_w(index)": _gap(single_ang.reshape(n, 3), bang[:, idx, :]),
    }
    for k, v in route_gaps.items():
        print(f"[{tag}] {k:<24} vs *_all: {v:.3e}")
    results[f"{tag}_addressing_routes_agree"] = max(route_gaps.values()) < TOL

    # ── optional / backend-dependent surface: REPORTED ───────────────────
    # Our URDFs declare no sites, and only mjlab has a site concept at all;
    # Newton/Genesis raise NotImplementedError. Recorded so the cross-sim
    # table shows exactly which backends a site-based task could use.
    try:
        sites = d.site_pos_w([name])
        measured[f"{tag}_site_pos_w"] = f"returned {tuple(sites.shape)}"
    except Exception as e:  # noqa: BLE001
        measured[f"{tag}_site_pos_w"] = type(e).__name__
    try:
        h = d.angular_momentum_w()
        measured[f"{tag}_angular_momentum_w"] = f"returned {tuple(h.shape)}"
    except Exception as e:  # noqa: BLE001
        measured[f"{tag}_angular_momentum_w"] = type(e).__name__
    print(f"[{tag}] site_pos_w -> {measured[f'{tag}_site_pos_w']}")
    print(f"[{tag}] angular_momentum_w -> {measured[f'{tag}_angular_momentum_w']}")


def _mjlab_body_evidence(env, name: str) -> None:
    """Dump the mjlab body indices behind an entity's root/body reads.

    ``root_com_pos_w`` (``xipos[root_body_id]``) and ``body_com_pos_w_all``
    (``xipos[body_ids]``) disagreed for the fixed table on the MuJoCo backend,
    which can only happen if ``root_body_id`` is not the body the name lookup
    resolves to. mjlab derives it from ``indexing.bodies[0].id`` — a spec
    handle — while ``body_ids`` is a compiled-model index tensor, so the two
    can drift apart. Printing both, next to the raw MuJoCo arrays they index,
    says which one is wrong instead of leaving it to inference.
    """
    entity = env.get_entity_data(name)._entity
    ed = entity.data
    idx = ed.indexing
    rb = idx.root_body_id
    ids = [int(i) for i in idx.body_ids]
    mj_model = env.scene_manager.sim.mj_model

    print(f"[{name}/mjlab] is_fixed_base = {entity.is_fixed_base}")
    print(f"[{name}/mjlab] indexing.root_body_id = {rb}   indexing.body_ids = {ids}")
    print(f"[{name}/mjlab] indexing.bodies names = {[b.name for b in idx.bodies]}")
    print(f"[{name}/mjlab] entity.body_names     = {list(entity.body_names)}")
    for i in dict.fromkeys([rb, *ids]):
        mj_name = mj_model.body(i).name
        xpos = [float(v) for v in ed.data.xpos[0, i]]
        xipos = [float(v) for v in ed.data.xipos[0, i]]
        ipos = [float(v) for v in ed.model.body_ipos[0, i]]
        mass = float(ed.model.body_mass[0, i])
        tag = "  <-- root_body_id" if i == rb else ""
        print(
            f"[{name}/mjlab] body {i:3d} {mj_name!r:22} mass={mass:8.3f} "
            f"xpos={_fmt3(xpos)} xipos={_fmt3(xipos)} body_ipos={_fmt3(ipos)}{tag}"
        )


def _newton_raw_contacts(env, name: str) -> None:
    """Dump every raw rigid contact that touches one entity's shapes.

    The sensor answers "was body X touched", so a disagreement with the other
    backends is either the sensor asking about the wrong body or the solver
    reporting the contact against a different body than the geometry implies.
    Printing the raw ``(shape0, shape1)`` pairs and the body each shape hangs
    off separates the two: the pair is what the collision pipeline actually
    produced, before any sensor filtering.
    """
    sm = env.scene_manager
    contacts = sm.sensor_contacts if sm.sensor_contacts is not None else sm.contacts
    if contacts is None:
        print(f"[{name}/newton] no contact buffer to dump")
        return
    leaf = lambda label: label.rsplit("/", 1)[-1]  # noqa: E731
    shape_body = sm.model.shape_body.numpy()
    body_labels = list(sm.model.body_label)
    shape_labels = list(sm.model.shape_label)
    view = sm.articulation_views[name]
    wanted = set(view.link_names)
    # shape_body == -1 is the static world (the ground plane); a Python
    # negative index would silently label it as the last body in the model.
    ours = {
        s
        for s in range(len(shape_labels))
        if int(shape_body[s]) >= 0 and leaf(body_labels[int(shape_body[s])]) in wanted
    }

    # Where the bodies actually ARE, and whether their shapes collide at all.
    # "the sensor says the base was touched" is only surprising if the jaw is
    # where the geometry says it is and its shape is collidable.
    body_q = sm.state_0.body_q.numpy()
    shape_flags = sm.model.shape_flags.numpy()
    cube_body = env.get_entity_data("cube").find_body_index("cube")
    for b in sorted({int(shape_body[s]) for s in ours})[: view.link_count] + [cube_body]:
        pos = body_q[b][:3]
        flags = [
            f"{leaf(shape_labels[s])}:flags={int(shape_flags[s])}"
            f"{'/COLLIDE' if int(shape_flags[s]) & 2 else '/NO-COLLIDE'}"
            for s in range(len(shape_labels))
            if int(shape_body[s]) == b
        ]
        print(
            f"[{name}/newton]   body {b} ({leaf(body_labels[b])}) at "
            f"[{pos[0]:+.5f}, {pos[1]:+.5f}, {pos[2]:+.5f}]  {' '.join(flags)}"
        )

    n = int(contacts.rigid_contact_count.numpy()[0])
    s0 = contacts.rigid_contact_shape0.numpy()
    s1 = contacts.rigid_contact_shape1.numpy()
    print(f"[{name}/newton] raw rigid contacts = {n} total; ones touching this entity:")
    shown = 0
    for c in range(min(n, len(s0))):
        a, b = int(s0[c]), int(s1[c])
        if a not in ours and b not in ours:
            continue
        if shown >= 12:
            print(f"[{name}/newton]   ... (truncated)")
            break
        pair = []
        for s in (a, b):
            if s < 0:
                pair.append("(-1)")
            else:
                pair.append(f"{leaf(shape_labels[s])}[shape {s}] on body {int(shape_body[s])} ")
        print(f"[{name}/newton]   {pair[0]} <-> {pair[1]}")
        shown += 1
    if shown == 0:
        print(f"[{name}/newton]   (none — the pipeline produced no contact on this entity's shapes)")


def _newton_view_evidence(env, name: str) -> None:
    """Dump the ArticulationView layout behind a Newton entity's root writes.

    A fixed-base entity's root pose is not in ``joint_q``; Newton redirects it
    to ``model.joint_X_p``, and the buffer a write must supply has to match
    that attribute's shape exactly — Newton asserts on it with no message. The
    shape is not guessable (selecting one joint by integer index drops an axis
    inside ``_get_attribute_array``), so print the array the read side returns,
    plus the layout numbers a manual write would need.
    """
    view = env.scene_manager.articulation_views[name]
    state = env.scene_manager.state
    arr = view.get_root_transforms(state)
    print(f"[{name}/newton] is_floating_base = {view.is_floating_base}")
    print(
        f"[{name}/newton] world_count={view.world_count} count={view.count} "
        f"count_per_world={view.count_per_world} joint_count={view.joint_count} link_count={view.link_count}"
    )
    print(f"[{name}/newton] get_root_transforms -> shape={arr.shape} ndim={arr.ndim} dtype={arr.dtype}")
    print(f"[{name}/newton] joint_names={list(view.joint_names)}  link_names={list(view.link_names)}")
    # What the contact-sensor resolver actually sees for this entity: the label
    # pool it matches patterns against, and the global body indices a per-body
    # pattern lands on. A primary that resolves to the wrong index reports no
    # contact while the entity-wide sensor still fires.
    idx = env.scene_manager.label_indexing.get(name)
    if idx is not None:
        n_per_world = idx.bodies.n_per_world
        print(f"[{name}/newton] label pool leaves (world 0) = {list(idx.bodies.leaves[:n_per_world])}")
        for leaf in list(idx.bodies.leaves[:n_per_world]):
            print(f"[{name}/newton]   find_bodies({leaf!r}) -> {idx.find_bodies((leaf,))}")

    # Which body each collision shape is PARENTED to. A contact is reported
    # against the shape's parent body, so a multi-body tool whose child shapes
    # were folded onto the root at import time reports every contact on the
    # root while still exposing the child bodies and their joints.
    leaf = lambda label: label.rsplit("/", 1)[-1]  # noqa: E731
    model = env.scene_manager.model
    shape_body = model.shape_body.numpy()
    body_flags = model.body_flags.numpy()
    body_labels = list(model.body_label)
    wanted = set(view.link_names)
    world0 = [b for b in range(len(body_labels)) if leaf(body_labels[b]) in wanted][: view.link_count]
    for b in world0:
        shapes = [i for i in range(len(model.shape_label)) if int(shape_body[i]) == b]
        names = [leaf(model.shape_label[i]) for i in shapes]
        print(
            f"[{name}/newton]   body {b} ({leaf(body_labels[b])}): flags={int(body_flags[b])} shapes={shapes} {names}"
        )


# ══════════════════════════════════════════════════════════════════════════
# Scene construction
# ══════════════════════════════════════════════════════════════════════════


def _build_env(sim: str, num_envs: int, extra_sensors: tuple = ()):
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(CUBE_URDF)
        cube_urdf = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(TABLE_URDF)
        table_urdf = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(GRIPPER_URDF)
        gripper_urdf = f.name

    cfgs = Go2FlatConfig(sim_type=sim, num_envs=num_envs).build()

    # Two passive rigid objects in the (separate) rigid_objects registry: one
    # free-floating, one welded to the world.
    cfgs.scene.rigid_objects = {
        "cube": RigidObjectCfg(
            urdf_path=cube_urdf,
            floating=True,
            init_state=InitialStateCfg(pos=CUBE_SPAWN),
        ),
        "table": RigidObjectCfg(
            urdf_path=table_urdf,
            floating=False,
            init_state=InitialStateCfg(pos=TABLE_SPAWN, rot=TABLE_ROT),
        ),
        # Passive: the hinges carry no actuator. Genesis's default link merging
        # only folds FIXED joints, so hinged jaws survive without
        # ``links_to_keep``; ``grip_primary_expands`` below fails loudly if any
        # backend collapses them anyway.
        "gripper": RigidObjectCfg(
            urdf_path=gripper_urdf,
            floating=False,
            init_state=InitialStateCfg(pos=GRIPPER_SPAWN),
        ),
    }

    # Pin the robot's spawn. The preset randomizes it over +-0.5 m in x/y and a
    # full yaw turn, which would make every cross-backend number below noise.
    cfgs.event.reset_root.params["pose_range"] = {}

    # Contact sensors whose entities are NOT the robot and NOT the terrain —
    # the shape a manipulation task needs and the one no preset has ever used.
    # "cube_table" is object<->object (the tool<->workpiece case); "cube_robot"
    # is object<->robot. Both name a passive rigid object as the PRIMARY, which
    # is the side every existing sensor fills with the robot.
    #
    # The field these live in is NOT the same on every backend: Genesis and
    # Newton keep contact sensors in `contact_sensors` (their `sensors` field
    # holds IMU-style sensors), while the mjlab scene keeps them in `sensors`.
    # Appending to the wrong one registers nothing, silently.
    field = "sensors" if sim == "mujoco" else "contact_sensors"
    existing = tuple(getattr(cfgs.scene, field) or ())
    # history_length must equal the decimation on the Genesis backend, so it is
    # taken from a sensor the preset already built rather than hardcoded.
    history_length = existing[0].history_length if existing else 0
    setattr(
        cfgs.scene,
        field,
        list(existing)
        + [
            # Both sides name ONE body, never ".*". MuJoCo's contact sensor
            # takes a single reference object, so mjlab silently keeps the
            # first match of a multi-body pattern — which for the table is its
            # massless "mocap_base" wrapper, a body that can never touch
            # anything. Genesis (whole entity) and Newton (all matches) would
            # have read the same config as "any body of that entity".
            ContactSensorCfg(
                name="cube_table_contact",
                primary=ContactMatch(mode="body", pattern="cube", entity="cube"),
                secondary=ContactMatch(mode="body", pattern="table", entity="table"),
                history_length=history_length,
            ),
            ContactSensorCfg(
                name="cube_robot_contact",
                primary=ContactMatch(mode="body", pattern="cube", entity="cube"),
                secondary=ContactMatch(mode="body", pattern="trunk", entity="robot"),
                history_length=history_length,
            ),
            # ── the matching-rule set ─────────────────────────────────────
            # With the cube touching ONE jaw, a correct backend answers
            # entity=True, left=True, right=False. A backend that widens a
            # named body to its whole entity answers right=True; one that
            # narrows an entity to its first body answers entity=False.
            ContactSensorCfg(
                name="cube_gripper_entity",
                primary=ContactMatch(mode="body", pattern="cube", entity="cube"),
                secondary=ContactMatch(mode="entity", entity="gripper"),
                history_length=history_length,
            ),
            # Mirror of cube_gripper_left with the two sides swapped. Contact
            # is symmetric, so these must agree; if one reports a touch and the
            # other does not, the fault is on the side that changed.
            ContactSensorCfg(
                name="mirror_cube_vs_jaw_left",
                primary=ContactMatch(mode="body", pattern="cube", entity="cube"),
                secondary=ContactMatch(mode="body", pattern="jaw_left", entity="gripper"),
                history_length=history_length,
            ),
            # Where did the contact actually land? A backend that folded the
            # jaws into their parent reports the touch here instead of on a
            # jaw. That is what a welded fixture does on Newton; with hinges it
            # must not happen anywhere.
            ContactSensorCfg(
                name="cube_gripper_base",
                primary=ContactMatch(mode="body", pattern="cube", entity="cube"),
                secondary=ContactMatch(mode="body", pattern="grip_base", entity="gripper"),
                history_length=history_length,
            ),
            ContactSensorCfg(
                name="cube_gripper_left",
                primary=ContactMatch(mode="body", pattern="jaw_left", entity="gripper"),
                secondary=ContactMatch(mode="entity", entity="cube"),
                history_length=history_length,
            ),
            ContactSensorCfg(
                name="cube_gripper_right",
                primary=ContactMatch(mode="body", pattern="jaw_right", entity="gripper"),
                secondary=ContactMatch(mode="entity", entity="cube"),
                history_length=history_length,
            ),
            # The grasp shape: BOTH jaws as primaries (two output columns) vs
            # the workpiece as a whole. "grasped" = both columns True.
            ContactSensorCfg(
                name="jaws_vs_cube",
                primary=ContactMatch(mode="body", pattern="jaw_.*", entity="gripper"),
                secondary=ContactMatch(mode="entity", entity="cube"),
                history_length=history_length,
            ),
            # Genesis requires history_length == decimation on every contact
            # sensor, so probe sensors inherit it rather than declaring one.
            *[replace(x, history_length=history_length) for x in extra_sensors],
        ],
    )

    # Place the cube at reset via the SAME event the robot uses, targeting it
    # through a SceneEntitySelector. No perturbation, so it should land exactly
    # at CUBE_SPAWN on every backend (this is what exercises the rigid-object
    # state writer and the polymorphic root-writer lookup).
    cfgs.event.reset_cube = EventTermConfig(
        func=common_ef.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "default_pos": CUBE_SPAWN,
            "default_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
            "asset_cfg": SceneEntitySelector(name="cube"),
        },
    )

    # Env only — no runner. Nothing here needs a policy, and building one
    # costs a JAX model plus its parameter-summary dump. Same entry point the
    # cross-sim comparison scripts use.
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return env


# ══════════════════════════════════════════════════════════════════════════
# Per-backend run
# ══════════════════════════════════════════════════════════════════════════


def run_one_sim(sim: str, num_envs: int, settle_steps: int) -> dict:
    env = _build_env(sim, num_envs)
    zeros = torch.zeros((env.num_envs, env.num_actions), device=env.device)

    print("=" * 78)
    print(f"RIGID-OBJECT / FIXED-ENTITY DIAG  [sim={sim}]")
    print("=" * 78)

    results: dict[str, bool] = {}
    measured: dict[str, object] = {}

    # ── A. free object + robot (original checks, unchanged) ──────────────
    try:
        rd = env.get_robot_data("robot")
        rpos = rd.root_link_pos_w[0].tolist()
        rjoints = tuple(rd.joint_pos.shape)
        print(f"[robot]  root_link_pos_w = {_fmt3(rpos)}  joint_pos.shape = {rjoints}")
        results["robot_read"] = len(rpos) == 3 and rd.joint_pos.shape[-1] > 0
    except Exception as e:  # noqa: BLE001
        print(f"[robot]  ERROR: {type(e).__name__}: {e}")
        results["robot_read"] = False

    try:
        od = env.get_rigid_object_data("cube")
        opos = od.root_link_pos_w[0].tolist()
        oquat = od.root_link_quat_w[0].tolist()
        ovel = od.root_link_lin_vel_w[0].tolist()
        print(f"[cube]   root_link_pos_w  = {_fmt3(opos)}")
        print(f"[cube]   root_link_quat_w = {_fmt3(oquat)}")
        print(f"[cube]   root_link_lin_vel_w = {_fmt3(ovel)}")
        results["object_read"] = (
            len(opos) == 3
            and len(oquat) == 4
            and len(ovel) == 3
            and _finite(od.root_link_pos_w)
            and _finite(od.root_link_quat_w)
            and _finite(od.root_link_lin_vel_w)
        )
    except Exception as e:  # noqa: BLE001
        print(f"[cube]   ERROR: {type(e).__name__}: {e}")
        results["object_read"] = False

    try:
        # The reset event adds env_origins, so the target is per-env. mjlab
        # lays environments out on an env_spacing grid in one world (origin[0]
        # is non-zero); Genesis and Newton give each env its own world, so
        # theirs are zero on flat terrain. Comparing against the bare spawn
        # would fail on mjlab for a correct placement.
        origins = env.scene_manager.env_origins
        expected = torch.tensor(CUBE_SPAWN, device=env.device).unsqueeze(0) + origins
        got = env.get_rigid_object_data("cube").root_link_pos_w
        err = float((got - expected).abs().max())
        print(f"[cube]   post-reset pos = {_fmt3(got[0])}  target = {_fmt3(expected[0])}  max_err = {err:.4f}")
        print(f"[cube]   env_origins[0] = {_fmt3(origins[0])}   (max err is over all {env.num_envs} envs)")
        results["object_placed_by_event"] = err < 1e-3
    except Exception as e:  # noqa: BLE001
        print(f"[cube]   PLACEMENT ERROR: {type(e).__name__}: {e}")
        results["object_placed_by_event"] = False

    has_joint_attr = hasattr(env.get_rigid_object_data("cube"), "joint_pos")
    print(f"[cube]   exposes joint_pos? {has_joint_attr} (expect False — RigidObjectData)")
    results["object_is_rigid_only"] = not has_joint_attr

    # ── B. MDP-layer reach (unchanged) ──────────────────────────────────
    print("-" * 78)
    robot_sel = env.resolve_selector(SceneEntitySelector(name="robot"))
    cube_sel = env.resolve_selector(SceneEntitySelector(name="cube"))

    try:
        h_robot = float(obs_common.base_height(env, robot_sel)[0, 0])
        print(f"[mdp]    base_height(robot) = {h_robot:.4f}")
        results["mdp_reads_robot"] = h_robot == h_robot
    except Exception as e:  # noqa: BLE001
        print(f"[mdp]    base_height(robot) ERROR: {type(e).__name__}: {e}")
        results["mdp_reads_robot"] = False

    try:
        h_cube = float(obs_common.base_height(env, cube_sel)[0, 0])
        target_h = CUBE_SPAWN[2] + float(env.scene_manager.env_origins[0, 2])
        print(f"[mdp]    base_height(cube)  = {h_cube:.4f}  (target {target_h:.4f})")
        results["mdp_reads_object"] = abs(h_cube - target_h) < 1e-3
    except Exception as e:  # noqa: BLE001
        print(f"[mdp]    base_height(cube)  ERROR: {type(e).__name__}: {e}")
        results["mdp_reads_object"] = False

    same = env.get_entity_data("robot") is env.get_robot_data("robot")
    print(f"[mdp]    get_entity_data('robot') is get_robot_data('robot')? {same} (expect True)")
    results["entity_data_identity"] = same

    try:
        obs_common.dof_pos(env, cube_sel)
        print("[mdp]    dof_pos(cube) returned a value (expected AttributeError)")
        results["joint_read_on_object_rejected"] = False
    except AttributeError as e:
        print(f"[mdp]    dof_pos(cube) -> AttributeError: {e}")
        results["joint_read_on_object_rejected"] = True
    except Exception as e:  # noqa: BLE001
        print(f"[mdp]    dof_pos(cube) -> {type(e).__name__} (expected AttributeError): {e}")
        results["joint_read_on_object_rejected"] = False

    try:
        env.get_robot_data("cube")
        print("[mdp]    get_robot_data('cube') SUCCEEDED (expected KeyError)")
        results["robot_accessor_is_strict"] = False
    except KeyError:
        print("[mdp]    get_robot_data('cube') -> KeyError (correct: not an articulation)")
        results["robot_accessor_is_strict"] = True
    except Exception as e:  # noqa: BLE001
        print(f"[mdp]    get_robot_data('cube') -> {type(e).__name__} (expected KeyError): {e}")
        results["robot_accessor_is_strict"] = False

    # ══════════════════════════════════════════════════════════════════
    # C. FIXED-BASE ENTITY
    # ══════════════════════════════════════════════════════════════════
    print("-" * 78)
    print("FIXED-BASE ENTITY (floating=False)")
    print("-" * 78)

    # C1 — registered in the rigid-object registry, not the articulation one.
    in_rigid = "table" in env.scene_manager.rigid_objects
    in_arti = "table" in env.scene_manager.entities
    print(f"[table]  in scene.rigid_objects: {in_rigid}   in scene.entities: {in_arti} (expect True / False)")
    results["fixed_in_rigid_registry"] = in_rigid and not in_arti

    # C2 — data object type and surface.
    td = env.get_entity_data("table")
    type_name = type(td).__name__
    fixed_has_joint = hasattr(td, "joint_pos")
    print(f"[table]  get_entity_data -> {type_name}   exposes joint_pos? {fixed_has_joint} (expect False)")
    results["fixed_is_rigid_only"] = not fixed_has_joint
    measured["fixed_data_type"] = type_name

    # C3 — root reads well-formed. A quaternion that is not unit-norm means the
    #      wxyz/xyzw plumbing is wrong for this entity kind.
    tpos = td.root_link_pos_w
    tquat = td.root_link_quat_w
    tlin = td.root_link_lin_vel_w
    tang = td.root_link_ang_vel_w
    qnorm = float(torch.linalg.norm(tquat[0]))
    print(f"[table]  root_link_pos_w  = {_fmt3(tpos[0])}   shape={tuple(tpos.shape)}")
    print(f"[table]  root_link_quat_w = {_fmt3(tquat[0])}  |q| = {qnorm:.6f} (expect 1.0)")
    print(f"[table]  root_lin_vel = {_fmt3(tlin[0])}   root_ang_vel = {_fmt3(tang[0])}")
    results["fixed_root_read_wellformed"] = (
        tpos.shape == (env.num_envs, 3)
        and tquat.shape == (env.num_envs, 4)
        and _finite(tpos)
        and _finite(tquat)
        and _finite(tlin)
        and _finite(tang)
        and abs(qnorm - 1.0) < 1e-4
    )
    results["fixed_at_rest"] = float(tlin.abs().max()) < 1e-6 and float(tang.abs().max()) < 1e-6

    # C3b — the spawn ORIENTATION, against what the config declared. A unit norm
    #       is not enough: the two quaternion layouts in play (config wxyz vs
    #       Newton/warp xyzw) map each other's identity onto a 180-degree flip
    #       about X, which is still unit-norm and invisible on a symmetric box.
    #       It is only visible on a multi-body entity, where the children end up
    #       mirrored through the root.
    decl_rot = list(TABLE_ROT)
    got_rot = [float(v) for v in tquat[0]]
    # q and -q are the same rotation.
    rot_err = min(
        max(abs(got_rot[i] - decl_rot[i]) for i in range(4)),
        max(abs(got_rot[i] + decl_rot[i]) for i in range(4)),
    )
    print(f"[table]  declared init_state.rot = {_fmt3(decl_rot)}   actual = {_fmt3(got_rot)}   err = {rot_err:.2e}")
    results["fixed_spawn_rot_matches_decl"] = rot_err < 1e-5

    # C4 — WHERE it landed. Reported, not asserted: the three backends take
    #      different routes to placing a welded body and no design decision has
    #      been made yet. The cross-sim table at the end is what this feeds.
    declared = list(TABLE_SPAWN)
    actual = [float(x) for x in tpos[0]]
    place_err = max(abs(actual[i] - declared[i]) for i in range(3))
    at_origin = max(abs(v) for v in actual) < 1e-6
    print(f"[table]  declared init_state.pos = {_fmt3(declared)}")
    print(f"[table]  actual                  = {_fmt3(actual)}   max_err = {place_err:.5f}")
    print(f"[table]  -> honours init_state.pos: {place_err < 1e-3}    sits at origin: {at_origin}   (REPORTED)")
    measured["fixed_pos"] = actual
    measured["fixed_honours_init_state"] = place_err < 1e-3
    measured["fixed_at_origin"] = at_origin

    # C5 — body-level reads. Indexed by NAME: on Newton these arrays span every
    #      body in the world (see _protocol_sweep), so a whole-array comparison
    #      would be measuring the robot's links, not the table's.
    bpos = td.body_pos_w_all
    bquat = td.body_quat_w_all
    bcom = td.body_com_pos_w_all
    table_idx = td.find_body_index("table")
    # For a single-link box whose inertial origin is the geometric centre, the
    # CoM must coincide with the link origin; a mismatch means the com /
    # link-frame split is mis-plumbed for this entity kind.
    com_gap = float((bcom[:, table_idx, :] - bpos[:, table_idx, :]).abs().max())
    print(f"[table]  body_pos_w_all  shape={tuple(bpos.shape)}  quat shape={tuple(bquat.shape)}")
    print(f"[table]  find_body_index('table') = {table_idx}")
    print(f"[table]  |body_com_pos_w_all[idx] - body_pos_w_all[idx]| = {com_gap:.3e} (expect ~0, CoM at box centre)")
    results["fixed_body_reads"] = (
        bpos.ndim == 3 and bpos.shape[0] == env.num_envs and bpos.shape[-1] == 3 and _finite(bpos) and _finite(bquat)
    )
    results["fixed_com_equals_link_origin"] = com_gap < 1e-4
    measured["fixed_body_count"] = int(bpos.shape[1])

    # C6 — name-addressed body reads agree with the batched ones.
    try:
        by_name = td.body_pos_w(["table"])
        gap = float((by_name[:, 0, :] - bpos[:, table_idx, :]).abs().max())
        print(f"[table]  max|body_pos_w(names) - body_pos_w_all[idx]| = {gap:.3e}")
        results["fixed_named_body_read"] = gap < 1e-6
    except Exception as e:  # noqa: BLE001
        print(f"[table]  named body read ERROR: {type(e).__name__}: {e}")
        results["fixed_named_body_read"] = False

    # C7 — the selector path MDP terms go through resolves a fixed entity.
    try:
        table_sel = env.resolve_selector(SceneEntitySelector(name="table"))
        h_table = float(obs_common.base_height(env, table_sel)[0, 0])
        print(f"[table]  resolve_selector ok; base_height(table) = {h_table:.4f}")
        results["fixed_mdp_reads"] = abs(h_table - actual[2]) < 1e-5
    except Exception as e:  # noqa: BLE001
        print(f"[table]  selector/MDP ERROR: {type(e).__name__}: {e}")
        results["fixed_mdp_reads"] = False

    try:
        env.resolve_selector(SceneEntitySelector(name="table", body_names=("table",)))
        print("[table]  resolve_selector(body_names=('table',)) ok")
        results["fixed_selector_body_names"] = True
    except Exception as e:  # noqa: BLE001
        print(f"[table]  selector body_names ERROR: {type(e).__name__}: {e}")
        results["fixed_selector_body_names"] = False

    # C8 — immovability. This is the definition of a fixed body and holds
    #      whatever the placement design turns out to be.
    before = td.root_link_pos_w.clone()
    reset_during_settle = _step_n(env, zeros, settle_steps)
    after = env.get_entity_data("table").root_link_pos_w
    drift = float((after - before).abs().max())
    print(f"[table]  after {settle_steps} steps: max|Δpos| = {drift:.3e} (expect ~0 — welded to world)")
    print(f"[table]  an env reset during those steps: {reset_during_settle}")
    # A reset does not move a welded body on any backend, so the immovability
    # claim holds either way; the flag is printed for context only.
    results["fixed_immovable"] = drift < 1e-6

    # C9 — collision geometry. Drop the cube from directly above wherever the
    #      table ACTUALLY is, so this is independent of the placement question:
    #      it should come to rest on the table top, not fall through to ground.
    table_top = actual[2] + TABLE_HALF_Z
    drop_from = table_top + 0.25
    writer = env.get_root_state_writer("cube")
    pos = torch.tensor([[actual[0], actual[1], drop_from]], device=env.device).expand(env.num_envs, 3).contiguous()
    quat = torch.zeros(env.num_envs, 4, device=env.device)
    quat[:, 0] = 1.0
    z3 = torch.zeros(env.num_envs, 3, device=env.device)
    writer.set_root_pose(pos, quat)
    writer.set_root_velocity(z3, z3)
    writer.eval_fk()
    env._invalidate_cache()

    print(f"[table]  dropping cube from z={drop_from:.4f} onto table top z={table_top:.4f}")
    reset_during_drop = _step_n(env, zeros, settle_steps * 4)
    rest_all = env.get_entity_data("cube").root_link_pos_w[:, 2]
    cube_rest = float(rest_all[0])
    expected_rest = table_top + CUBE_HALF
    ground_rest = CUBE_HALF
    print(
        f"[table]  cube rest z = {cube_rest:.4f}   on-table expects {expected_rest:.4f}, ground would be {ground_rest:.4f}"
    )
    print(f"[table]  rest z over all {env.num_envs} envs = {[round(float(v), 4) for v in rest_all]}")
    if reset_during_drop:
        # The reset_cube event teleported the cube back to CUBE_SPAWN mid-drop,
        # so this resting height says nothing about the table's collision
        # geometry. Report a void measurement instead of failing on one that
        # never actually ran.
        print("[table]  an env reset fired during the drop -> cube was teleported away; MEASUREMENT VOID")
        print("[table]  re-run with a smaller --settle-steps so the robot cannot topple first")
        measured["fixed_collision_check"] = "VOID (reset during drop)"
    else:
        results["fixed_collision_supports_object"] = float((rest_all - expected_rest).abs().max()) < 0.03
        measured["fixed_collision_check"] = "measured"
    measured["cube_rest_on_table"] = cube_rest
    measured["table_top"] = table_top

    # C10 — self-description. MDP terms branch on this to skip the velocity
    #       write for a welded entity, so it has to be right per backend.
    is_fixed = env.get_entity_data("table").is_fixed_base
    cube_is_fixed = env.get_entity_data("cube").is_fixed_base
    print(f"[table]  is_fixed_base = {is_fixed} (expect True)   cube.is_fixed_base = {cube_is_fixed} (expect False)")
    results["fixed_self_describes"] = is_fixed and not cube_is_fixed

    # C11 — a reset event must place a welded entity, PER ENVIRONMENT. Once
    #       env_origins is non-zero this is the only correct placement path:
    #       the build-time pose is a single value every environment shares.
    #       Each backend gets there differently (mjlab: mocap base; Newton:
    #       joint_X_p -> the solver's mocap bodies; Genesis: fixed-link pose
    #       write), so this is the check that they actually agree.
    target = (actual[0] + 0.5, actual[1], actual[2])
    all_ids = torch.arange(env.num_envs, device=env.device)
    try:
        common_ef.reset_root_state_uniform(
            env=env,
            env_ids=all_ids,
            pose_range={},
            velocity_range={},
            default_pos=target,
            default_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            asset_cfg=env.resolve_selector(SceneEntitySelector(name="table")),
        )
        env._post_reset_forward()
        env._invalidate_cache()
        origins = env.scene_manager.env_origins
        expected = torch.tensor(target, device=env.device).unsqueeze(0) + origins
        moved_to = env.get_entity_data("table").root_link_pos_w
        err = float((moved_to - expected).abs().max())
        print(f"[table]  reset-event write -> env0 {_fmt3(moved_to[0])}  expected {_fmt3(expected[0])}")
        print(
            f"[table]  env_origins[0] = {_fmt3(origins[0])}   max|actual - expected| over {env.num_envs} envs = {err:.3e}"
        )
        results["fixed_reset_places_entity"] = err < 1e-3
    except Exception as e:  # noqa: BLE001
        print(f"[table]  reset-event write raised {type(e).__name__}: {e}")
        results["fixed_reset_places_entity"] = False

    # C12 — per-env placement, independent of env_origins: write a different
    #       pose to every environment and read each one back. A backend that
    #       stores the pose in the shared model instead of per-env state
    #       collapses these to one value — silently, and only at num_envs > 1.
    try:
        table_writer = env.get_root_state_writer("table")
        per_env = torch.tensor(target, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
        per_env[:, 0] += torch.arange(env.num_envs, device=env.device, dtype=torch.float32)
        quat = torch.zeros(env.num_envs, 4, device=env.device)
        quat[:, 0] = 1.0
        table_writer.set_root_pose(per_env, quat, env_ids=all_ids)
        table_writer.eval_fk(env_ids=all_ids)
        env._post_reset_forward()
        env._invalidate_cache()
        td_after = env.get_entity_data("table")
        got = td_after.root_link_pos_w
        err = float((got - per_env).abs().max())
        # Also read it as a BODY, not as a root. On Newton the root read comes
        # straight back out of the model attribute that was just written, so it
        # would agree even if the physical body never moved; the body read goes
        # through the simulator's own kinematics.
        body_got = td_after.body_pos_w(["table"])[:, 0, :]
        body_err = float((body_got - per_env).abs().max())
        print(f"[table]  per-env write x = {[round(float(v), 3) for v in per_env[:, 0]]}")
        print(f"[table]  read back     x = {[round(float(v), 3) for v in got[:, 0]]}   max err = {err:.3e}")
        print(f"[table]  body_pos_w    x = {[round(float(v), 3) for v in body_got[:, 0]]}   max err = {body_err:.3e}")
        results["fixed_pose_write_is_per_env"] = err < 1e-3 and body_err < 1e-3
        # And it must survive stepping: a welded body holds the pose it was given.
        _step_n(env, zeros, 3)
        td_held = env.get_entity_data("table")
        drift = float((td_held.root_link_pos_w - per_env).abs().max())
        body_drift = float((td_held.body_pos_w(["table"])[:, 0, :] - per_env).abs().max())
        print(f"[table]  after 3 steps: max|Δ| root = {drift:.3e}   body = {body_drift:.3e}")
        results["fixed_pose_write_persists"] = drift < 1e-4 and body_drift < 1e-4
    except Exception as e:  # noqa: BLE001
        print(f"[table]  per-env pose write raised {type(e).__name__}: {e}")
        results["fixed_pose_write_is_per_env"] = False
        results["fixed_pose_write_persists"] = False

    # C14 — the check the earlier drop cannot make: does the COLLISION geometry
    #       follow a table that has been moved per environment? C9 dropped the
    #       cube before any move, into env 0 only. Reading a moved pose back
    #       proves the kinematics followed; it says nothing about whether the
    #       simulator's collision representation did. On Newton the fixture is
    #       now a kinematic body and on mjlab a mocap body — both are exactly
    #       the cases where a stale collider would go unnoticed. So: drop a cube
    #       onto EACH environment's table, at its own new x, and check every
    #       environment's resting height.
    try:
        td_moved = env.get_entity_data("table")
        tops = td_moved.root_link_pos_w[:, 2] + TABLE_HALF_Z
        drop_pos = td_moved.root_link_pos_w.clone()
        drop_pos[:, 2] = tops + 0.25
        cube_writer = env.get_root_state_writer("cube")
        z3 = torch.zeros(env.num_envs, 3, device=env.device)
        cube_quat = torch.zeros(env.num_envs, 4, device=env.device)
        cube_quat[:, 0] = 1.0
        cube_writer.set_root_pose(drop_pos, cube_quat, env_ids=all_ids)
        cube_writer.set_root_velocity(z3, z3, env_ids=all_ids)
        cube_writer.eval_fk(env_ids=all_ids)
        env._invalidate_cache()
        print(f"[table]  moved tables at x = {[round(float(v), 3) for v in td_moved.root_link_pos_w[:, 0]]}")
        reset_during_redrop = _step_n(env, zeros, settle_steps * 4)
        rest_moved = env.get_entity_data("cube").root_link_pos_w[:, 2]
        expected_moved = tops + CUBE_HALF
        err_moved = float((rest_moved - expected_moved).abs().max())
        print(f"[table]  cube rest z per env = {[round(float(v), 4) for v in rest_moved]}")
        print(
            f"[table]  expected            = {[round(float(v), 4) for v in expected_moved]}   max err = {err_moved:.4f}"
        )
        print(f"[table]  (falling through would land at z = {CUBE_HALF})")
        if reset_during_redrop:
            print("[table]  an env reset fired during the drop -> MEASUREMENT VOID")
            measured["moved_collision_check"] = "VOID (reset during drop)"
        else:
            results["fixed_collision_follows_move"] = err_moved < 0.03
            measured["moved_collision_check"] = "measured"
    except Exception as e:  # noqa: BLE001
        print(f"[table]  moved-table drop raised {type(e).__name__}: {e}")
        results["fixed_collision_follows_move"] = False

    # C13 — a welded body has no root velocity state; every backend must say so
    #       the same way rather than one of them silently doing nothing.
    try:
        z3 = torch.zeros(env.num_envs, 3, device=env.device)
        env.get_root_state_writer("table").set_root_velocity(z3, z3, env_ids=all_ids)
        print("[table]  set_root_velocity SUCCEEDED (expected ValueError)")
        results["fixed_velocity_write_rejected"] = False
    except ValueError as e:
        print(f"[table]  set_root_velocity -> ValueError: {e}")
        results["fixed_velocity_write_rejected"] = True
    except Exception as e:  # noqa: BLE001
        print(f"[table]  set_root_velocity -> {type(e).__name__} (expected ValueError): {e}")
        results["fixed_velocity_write_rejected"] = False

    # ══════════════════════════════════════════════════════════════════
    # D. FULL PROTOCOL SWEEP
    # ══════════════════════════════════════════════════════════════════
    # The table sweeps at rest (every velocity zero, identity rotation), so
    # the frame-transform checks there are degenerate. The cube is therefore
    # thrown into free flight first — tilted and spinning — so the same
    # checks have to distinguish a correct rotation from a wrong one.
    print("-" * 78)
    print("PROTOCOL SWEEP: cube in flight (tilted + spinning)")
    print("-" * 78)

    rpy = torch.tensor([0.35, 0.20, 0.70], device=env.device)
    flight_quat = quat_from_euler_xyz_wxyz(rpy[0], rpy[1], rpy[2]).reshape(1, 4).expand(env.num_envs, 4).contiguous()
    flight_pos = torch.tensor([[-2.0, 2.0, 1.5]], device=env.device).expand(env.num_envs, 3).contiguous()
    flight_lin = torch.tensor([[0.5, -0.3, 0.2]], device=env.device).expand(env.num_envs, 3).contiguous()
    flight_ang = torch.tensor([[0.4, 0.7, -0.2]], device=env.device).expand(env.num_envs, 3).contiguous()
    writer.set_root_pose(flight_pos, flight_quat)
    writer.set_root_velocity(flight_lin, flight_ang)
    writer.eval_fk()
    env._invalidate_cache()
    # Two steps of unobstructed free flight, so the state the sweep reads is
    # produced by the integrator rather than by the write above.
    reset_during_flight = _step_n(env, zeros, 2)
    speed = float(env.get_entity_data("cube").root_link_lin_vel_w.norm(dim=-1).max())
    spin = float(env.get_entity_data("cube").root_link_ang_vel_w.norm(dim=-1).max())
    print(
        f"[cube]  in-flight |v| = {speed:.4f} m/s   |w| = {spin:.4f} rad/s   (reset during flight: {reset_during_flight})"
    )
    measured["cube_sweep_speed"] = round(speed, 4)
    measured["cube_sweep_spin"] = round(spin, 4)
    _protocol_sweep(env, "cube", results, measured, tag="cube")

    print("-" * 78)
    print("PROTOCOL SWEEP: table (welded, at rest)")
    print("-" * 78)
    _protocol_sweep(env, "table", results, measured, tag="table")

    # ══════════════════════════════════════════════════════════════════
    # E. CONTACT SENSING BETWEEN NON-TERRAIN ENTITIES
    # ══════════════════════════════════════════════════════════════════
    # Every existing sensor watches robot-vs-terrain. A manipulation task
    # needs robot-vs-object and object-vs-object — tool touching workpiece —
    # and that path has never been exercised. The config expresses it
    # (ContactMatch.entity), and all three backends resolve a rigid-object
    # name, but resolving a name is not the same as reporting a contact.
    print("-" * 78)
    print("CONTACT SENSING (object<->object, object<->robot)")
    print("-" * 78)

    groups = env.contact_manager.group_names()
    have_both = env.contact_manager.has_group("cube_table_contact") and env.contact_manager.has_group(
        "cube_robot_contact"
    )
    print(f"[contact] groups = {groups}")
    if have_both:
        print(f"[contact] cube_table tracked = {env.contact_manager.tracked_names('cube_table_contact')}")
        print(f"[contact] cube_robot tracked = {env.contact_manager.tracked_names('cube_robot_contact')}")
    results["contact_groups_registered"] = have_both

    def _place_cube(pos: torch.Tensor) -> None:
        w = env.get_root_state_writer("cube")
        q = torch.zeros(env.num_envs, 4, device=env.device)
        q[:, 0] = 1.0
        zero = torch.zeros(env.num_envs, 3, device=env.device)
        w.set_root_pose(pos, q, env_ids=all_ids)
        w.set_root_velocity(zero, zero, env_ids=all_ids)
        w.eval_fk(env_ids=all_ids)
        env._invalidate_cache()

    if have_both:
        # E1 — cube resting on the table. The contact force is predictable:
        #      a body at rest is held up by exactly its own weight, so this is
        #      the check that the reported magnitude is a real force and not an
        #      arbitrary number that merely happens to be non-zero.
        table_pos = env.get_entity_data("table").root_link_pos_w
        rest_pos = table_pos.clone()
        rest_pos[:, 2] = table_pos[:, 2] + TABLE_HALF_Z + CUBE_HALF
        _place_cube(rest_pos)
        reset_settling = _step_n(env, zeros, settle_steps * 2)
        on_table = env.contact_manager.is_contact("cube_table_contact")
        f_table = env.contact_manager.contact_force("cube_table_contact")
        f_mag = f_table.reshape(env.num_envs, -1, 3).norm(dim=-1).max(dim=-1).values
        weight = CUBE_MASS * 9.81
        print(f"[contact] cube on table -> is_contact = {[bool(v) for v in on_table.reshape(env.num_envs, -1)[:, 0]]}")
        print(f"[contact] |force| per env = {[round(float(v), 3) for v in f_mag]} N   weight = {weight:.3f} N")
        print(f"[contact] (reset during settling: {reset_settling})")
        results["contact_object_object_detected"] = bool(on_table.all())
        results["contact_force_matches_weight"] = bool(((f_mag - weight).abs() < 0.5 * weight).all())
        measured["contact_force_on_table"] = [round(float(v), 3) for v in f_mag]
        measured["contact_expected_weight"] = round(weight, 3)

        # E2 — and it must report NO contact once the object is elsewhere. A
        #      sensor stuck at True is as useless as one stuck at False.
        away = table_pos.clone()
        away[:, 0] -= 3.0
        away[:, 2] = 1.5
        _place_cube(away)
        _step_n(env, zeros, 2)
        off_table = env.contact_manager.is_contact("cube_table_contact")
        f_off = env.contact_manager.contact_force("cube_table_contact")
        f_off_mag = float(f_off.reshape(env.num_envs, -1, 3).norm(dim=-1).max())
        print(f"[contact] cube in mid-air -> any is_contact = {bool(off_table.any())}   max|force| = {f_off_mag:.4f} N")
        results["contact_clears_when_apart"] = (not bool(off_table.any())) and f_off_mag < 1e-2

        # E3 — object<->robot. The cube is written into the trunk so the
        #      contact exists on the very next step regardless of how the
        #      robot happens to be standing; the penetration force is not
        #      predictable, so only the detection is asserted.
        trunk = env.get_entity_data("robot").body_pos_w(["trunk"])[:, 0, :]
        _place_cube(trunk.clone())
        _step_n(env, zeros, 1)
        on_robot = env.contact_manager.is_contact("cube_robot_contact")
        f_robot = env.contact_manager.contact_force("cube_robot_contact")
        f_robot_mag = f_robot.reshape(env.num_envs, -1, 3).norm(dim=-1).max(dim=-1).values
        print(
            f"[contact] cube inside trunk -> is_contact = {[bool(v) for v in on_robot.reshape(env.num_envs, -1)[:, 0]]}"
        )
        print(
            f"[contact] |force| per env = {[round(float(v), 2) for v in f_robot_mag]} N (penetration, magnitude not asserted)"
        )
        results["contact_object_robot_detected"] = bool(on_robot.all())
        measured["contact_force_on_robot"] = [round(float(v), 2) for v in f_robot_mag]

        # E4 — the accumulators every gait/grasp reward reads. Contact time
        #      must have advanced while the cube sat on the table, and air time
        #      while it hung in the air.
        air = env.contact_manager.current_air_time("cube_table_contact")
        print(
            f"[contact] cube_table current_air_time = {[round(float(v), 3) for v in air.reshape(env.num_envs, -1)[:, 0]]} s"
        )
        results["contact_air_time_accumulates"] = bool((air > 0).all())
        measured["contact_air_time"] = [round(float(v), 3) for v in air.reshape(env.num_envs, -1)[:, 0]]

    # ══════════════════════════════════════════════════════════════════
    # F. CONTACT MATCHING RULES (the tool-and-workpiece shape)
    # ══════════════════════════════════════════════════════════════════
    # One primary, three spellings of the counterpart, against a fixture whose
    # jaws are separate bodies. This is where the backends used to disagree
    # silently: Genesis ignored a named secondary and watched the whole entity,
    # mjlab kept only the first match of a multi-body one.
    print("-" * 78)
    print("CONTACT MATCHING RULES (multi-body tool)")
    print("-" * 78)

    grip_groups = ("cube_gripper_entity", "cube_gripper_left", "cube_gripper_right", "jaws_vs_cube")
    have_grip = all(env.contact_manager.has_group(g) for g in grip_groups)
    print(f"[grip] groups present: {have_grip}")
    if have_grip:
        jaw_names = env.contact_manager.tracked_names("jaws_vs_cube")
        print(f"[grip] jaws_vs_cube primaries = {jaw_names} (expect both jaws -> 2 columns)")
        # A multi-element PRIMARY must expand to one column per element on every
        # backend; mjlab does it by emitting one MuJoCo sensor per primary.
        results["grip_primary_expands"] = sorted(jaw_names) == ["jaw_left", "jaw_right"]
    else:
        results["grip_primary_expands"] = False

    def _grip_state(label: str) -> dict[str, bool]:
        """Read the four groups and print them as one row."""
        ent = bool(env.contact_manager.is_contact("cube_gripper_entity").all())
        mirror = bool(env.contact_manager.is_contact("mirror_cube_vs_jaw_left").all())
        base = bool(env.contact_manager.is_contact("cube_gripper_base").all())
        left = bool(env.contact_manager.is_contact("cube_gripper_left").all())
        right = bool(env.contact_manager.is_contact("cube_gripper_right").all())
        jaws = env.contact_manager.is_contact("jaws_vs_cube")
        grasped = bool(jaws.all(dim=-1).all())
        cols = [bool(v) for v in jaws.reshape(env.num_envs, -1)[0]]
        print(
            f"[grip] {label:<22} entity={ent!s:<5} left={left!s:<5} right={right!s:<5} "
            f"jaws={cols} grasped={grasped}   mirror={mirror} base={base}"
        )
        return {"entity": ent, "left": left, "right": right, "grasped": grasped, "mirror": mirror, "base": base}

    if have_grip:
        grip_pos = env.get_entity_data("gripper").root_link_pos_w
        cube_writer = env.get_root_state_writer("cube")
        cube_quat = torch.zeros(env.num_envs, 4, device=env.device)
        cube_quat[:, 0] = 1.0
        z3 = torch.zeros(env.num_envs, 3, device=env.device)

        def _put_cube(offset: tuple[float, float, float]) -> None:
            pos = grip_pos + torch.tensor(offset, device=env.device).unsqueeze(0)
            cube_writer.set_root_pose(pos, cube_quat, env_ids=all_ids)
            cube_writer.set_root_velocity(z3, z3, env_ids=all_ids)
            cube_writer.eval_fk(env_ids=all_ids)
            env._invalidate_cache()

        # F1 — cube resting on the LEFT jaw only. Gravity holds it there, so the
        #      reading is a settled contact rather than a teleport artefact.
        _put_cube((0.0, GRIPPER_JAW_Y, GRIPPER_JAW_TOP + CUBE_HALF))
        reset_a = _step_n(env, zeros, settle_steps)
        one_jaw = _grip_state("on left jaw:")
        print(f"[grip] (reset during settle: {reset_a})")
        if sim == "newton":
            _newton_raw_contacts(env, "gripper")
        results["grip_entity_sees_any_body"] = one_jaw["entity"]
        results["grip_named_body_is_exact"] = one_jaw["left"] and not one_jaw["right"]
        # Contact is symmetric: naming the jaw as primary or as secondary must
        # give the same answer.
        results["grip_primary_secondary_symmetric"] = one_jaw["left"] == one_jaw["mirror"]
        # The cube sits on a jaw, nowhere near the base plate. A backend that
        # merged the fixed-jointed bodies reports the touch on the base.
        results["grip_contact_not_attributed_to_parent"] = not one_jaw["base"]
        results["grip_not_grasped_on_one_jaw"] = not one_jaw["grasped"]
        measured["grip_one_jaw_state"] = one_jaw

        # F2 — cube in the gap, interfering with BOTH jaws. This is the grasp.
        _put_cube((0.0, 0.0, GRIPPER_JAW_TOP - GRIPPER_JAW_HALF_Z))
        _step_n(env, zeros, 1)
        both = _grip_state("wedged in jaws:")
        results["grip_both_jaws_detected"] = both["left"] and both["right"]
        results["grip_grasp_detected"] = both["grasped"]
        measured["grip_both_jaw_state"] = both

        # F3 — and nothing at all once the workpiece is elsewhere.
        _put_cube((0.0, 0.0, 3.0))
        _step_n(env, zeros, 2)
        away = _grip_state("far away:")
        results["grip_clears_when_apart"] = not (away["entity"] or away["left"] or away["right"])

    # Backend-specific raw evidence for the places where the backends diverge
    # internally. Printed unconditionally so a passing run still records the
    # indices and shapes a future regression would change.
    if sim == "mujoco":
        print("-" * 78)
        print("MJLAB BODY-INDEX EVIDENCE")
        print("-" * 78)
        _mjlab_body_evidence(env, "cube")
        _mjlab_body_evidence(env, "table")
    if sim == "newton":
        print("-" * 78)
        print("NEWTON VIEW-LAYOUT EVIDENCE")
        print("-" * 78)
        _newton_view_evidence(env, "cube")
        _newton_view_evidence(env, "table")
        _newton_view_evidence(env, "gripper")

    # ── verdict ──────────────────────────────────────────────────────────
    print("=" * 78)
    print("VERDICT")
    for k, v in results.items():
        print(f"  {k:32s}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"  {'OVERALL':32s}: {'PASS' if ok else 'FAIL'}")
    print("\nREPORTED (not pass/fail — design evidence)")
    for k, v in measured.items():
        print(f"  {k:32s}: {v}")
    print("=" * 78)

    return {"sim": sim, "ok": ok, "results": results, "measured": measured}


def run_multimatch_probe(sim: str, num_envs: int) -> dict:
    """Build a scene whose secondary names SEVERAL bodies, and report what happens.

    MuJoCo's contact sensor carries one reference object. "Both jaws but not the
    handle" therefore has no MuJoCo representation — it is neither the whole
    entity (a subtree) nor a single body. Genesis and Newton filter contacts
    against a set of bodies and can express it.

    So the correct behaviour is backend-specific, and both halves matter: the
    two backends that can do it must build, and the one that cannot must say so
    at build time instead of quietly watching one jaw. Runs as its own process
    because a correct mjlab result is a failed build.
    """
    print("=" * 78)
    print(f"MULTI-MATCH SECONDARY PROBE  [sim={sim}]")
    print("=" * 78)
    outcome: str
    try:
        env = _build_env(
            sim,
            num_envs,
            extra_sensors=(
                ContactSensorCfg(
                    name="cube_vs_both_jaws",
                    primary=ContactMatch(mode="body", pattern="cube", entity="cube"),
                    secondary=ContactMatch(mode="body", pattern="jaw_.*", entity="gripper"),
                ),
            ),
        )
        names = env.contact_manager.tracked_names("cube_vs_both_jaws")
        outcome = f"built (tracked={names})"
    except Exception as e:  # noqa: BLE001
        outcome = f"{type(e).__name__}: {e}"

    expected_to_build = sim != "mujoco"
    built = outcome.startswith("built")
    print(f"[probe] outcome  : {outcome}")
    print(
        f"[probe] expected : {'builds (backend supports a body set)' if expected_to_build else 'raises at build (MuJoCo takes one reference)'}"
    )
    ok = built == expected_to_build
    # When it must fail, the message has to name the cause; a bare crash would
    # leave a preset author guessing.
    if not expected_to_build and not built:
        informative = "single" in outcome.lower() or "multiple" in outcome.lower() or "matched" in outcome.lower()
        print(f"[probe] message names the cause: {informative}")
        ok = ok and informative
    print(f"[probe] verdict  : {'PASS' if ok else 'FAIL'}")
    return {
        "sim": sim,
        "ok": ok,
        "results": {"secondary_multimatch_policy": ok},
        "measured": {"multimatch_outcome": outcome},
    }


# ══════════════════════════════════════════════════════════════════════════
# Parent: run every backend, then compare
# ══════════════════════════════════════════════════════════════════════════


def run_parent(args) -> int:
    payloads: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="rigid_object_smoke_") as tmp:
        for sim in _SIMS:
            out = Path(tmp) / f"{sim}.json"
            cmd = [
                sys.executable,
                "-m",
                _MODULE,
                "--sim",
                sim,
                "--result-json",
                str(out),
                "--num-envs",
                str(args.num_envs),
                "--settle-steps",
                str(args.settle_steps),
            ]
            print("\n" + "#" * 78)
            print(f"# $ {' '.join(cmd)}")
            print("#" * 78, flush=True)
            proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            if out.exists():
                payloads[sim] = json.loads(out.read_text())
            else:
                payloads[sim] = {
                    "sim": sim,
                    "ok": False,
                    "results": {},
                    "measured": {},
                    "crash": proc.stderr or f"rc={proc.returncode}",
                }

            # Second child: the multi-body-secondary probe. Separate process
            # because on mjlab the correct outcome is a failed build, which
            # would otherwise take the whole run down with it.
            probe_out = Path(tmp) / f"{sim}_probe.json"
            probe_cmd = [
                sys.executable,
                "-m",
                _MODULE,
                "--sim",
                sim,
                "--probe-multimatch",
                "--result-json",
                str(probe_out),
                "--num-envs",
                "1",
            ]
            print("\n" + "#" * 78)
            print(f"# $ {' '.join(probe_cmd)}")
            print("#" * 78, flush=True)
            probe_proc = subprocess.run(probe_cmd, stderr=subprocess.PIPE, text=True)
            if probe_out.exists():
                probe = json.loads(probe_out.read_text())
                payloads[sim]["results"].update(probe["results"])
                payloads[sim]["measured"].update(probe["measured"])
                payloads[sim]["ok"] = payloads[sim]["ok"] and probe["ok"]
            else:
                tail = [ln for ln in (probe_proc.stderr or "").splitlines() if ln.strip()][-5:]
                print("\n".join(tail), file=sys.stderr)
                payloads[sim]["results"]["secondary_multimatch_policy"] = False
                payloads[sim]["measured"]["multimatch_outcome"] = f"probe crashed (rc={probe_proc.returncode})"
                payloads[sim]["ok"] = False

    print("\n" + "=" * 78)
    print("CROSS-SIM SUMMARY")
    print("=" * 78)

    check_names: list[str] = []
    for p in payloads.values():
        for k in p.get("results", {}):
            if k not in check_names:
                check_names.append(k)

    print(f"{'check':<34}" + "".join(f"{s:>10}" for s in _SIMS))
    print("-" * 78)
    all_ok = True
    for name in check_names:
        row = ""
        for s in _SIMS:
            v = payloads[s].get("results", {}).get(name)
            # A missing key means the backend never ran that check (crashed, or
            # the measurement was voided). Neither is a failure of the check
            # itself; crashes fail the run separately below.
            row += f"{('PASS' if v else 'FAIL') if v is not None else '-':>10}"
            if v is not None:
                all_ok &= bool(v)
        print(f"{name:<34}{row}")

    print("\nREPORTED values (backends must be compared by hand — these are not asserted)")
    keys: list[str] = []
    for p in payloads.values():
        for k in p.get("measured", {}):
            if k not in keys:
                keys.append(k)
    for k in keys:
        vals = [payloads[s].get("measured", {}).get(k) for s in _SIMS]
        agree = len({json.dumps(v, sort_keys=True) for v in vals}) == 1
        print(f"  {k:<30}" + "".join(f"{str(v):>26}" for v in vals) + ("   AGREE" if agree else "   <-- DIFFER"))

    crashed = [s for s in _SIMS if "crash" in payloads[s]]
    if crashed:
        print("\n" + "=" * 78)
        print("CRASHED BACKENDS — last 30 stderr lines each")
        for s in crashed:
            print(f"\n--- {s} " + "-" * (72 - len(s)))
            tail = [ln for ln in str(payloads[s]["crash"]).splitlines() if ln.strip()][-30:]
            print("\n".join(tail) if tail else "<empty>")
        all_ok = False

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sim",
        choices=list(_SIMS),
        default=None,
        help="Run a single backend. Omit to run all three and cross-compare.",
    )
    ap.add_argument("--result-json", default=None, help="Child mode: where to write this backend's result.")
    ap.add_argument(
        "--probe-multimatch",
        action="store_true",
        help="Child mode: build a scene with a multi-body secondary and report whether it builds.",
    )
    # Four, not one: the per-environment placement checks are vacuous at
    # num_envs=1, and a backend that collapses per-env state into one shared
    # value passes every single-env check.
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--settle-steps", type=int, default=20)
    args = ap.parse_args()

    if args.sim is None:
        return run_parent(args)

    if args.probe_multimatch:
        payload = run_multimatch_probe(args.sim, args.num_envs)
    else:
        payload = run_one_sim(args.sim, args.num_envs, args.settle_steps)
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(payload, default=str))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
