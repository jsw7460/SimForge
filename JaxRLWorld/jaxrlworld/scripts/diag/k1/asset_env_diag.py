"""Asset + MDP fingerprint for the Booster K1 velocity port, one backend at a time.

Every check compares against a value read from somewhere else -- the MJCF on
disk, or the source recipe's own numbers written out below -- rather than
against something this script derives from the environment it is testing.

Sections:

  A. Asset. Joint set and order, soft position limits against the MJCF ranges,
     effort limits, PD gains, armature, action scale and offset, and the
     collision geometry with its contact parameters.
  B. MDP. Observation dimensions, then the sign of every reward term over a
     rollout that is first random and then saturated against both ends of the
     joint ranges, so that no term goes unexercised and passes vacuously. A
     penalty that comes back positive is the one error in this port that would
     train silently.
  C. Contacts. All three groups register and each one behaves: the feet reach
     the ground, and the non-foot group stays quiet while the robot settles.
  D. Terminations. The tilt measure reads correctly at known attitudes, and the
     stochastic fall term fires at its configured rate rather than on the first
     tilted step.
  E. Curriculum. Both schedules apply the stage the step counter has reached.
  F. Push. The interval and the six velocity ranges equal the ones written in
     the source's config file (read by regex, not restated), and a kick of the
     source's maximum magnitude lands on the root velocity of this backend.

Run one backend per process (the single-sim invariant)::

    jaxpy -m jaxrlworld.scripts.diag.k1.asset_env_diag --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.k1.asset_env_diag --sim newton
    jaxpy -m jaxrlworld.scripts.diag.k1.asset_env_diag --sim genesis
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch

from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.events import EventTermConfig
from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig
from jaxrlworld.rl.configs.robots.k1 import K1Config
from jaxrlworld.rl.envs.mdp.events import common as ev
from jaxrlworld.rl.runners.base_runner import BaseRunner

# ── expected values, stated independently of the environment ─────────

# The source recipe's task config, from which the push event is read verbatim.
SOURCE_VELOCITY_CFG = Path(
    "third_party/rl_frameworks/booster_mjlab/src/booster_mjlab/tasks/velocity/velocity_env_cfg.py"
)

ACTOR_DIM = 75
CRITIC_DIM = 90

# Sign each reward term must have for EVERY environment on EVERY step. The
# preset's weights are all positive, so a weighted term's sign is its raw
# term's sign.
# Action magnitude for the saturating phases of the sign-check rollout. Large
# enough that every one of the 22 joints is commanded past a soft position
# limit at the home pose: the binding joint is the hip pitch, whose wide range
# and small action scale need |a| >= 11.1. Verified against the MJCF ranges,
# not guessed -- a value that reached only some joints would leave the
# joint-limit penalty firing by luck.
SATURATING_ACTION = 12.0

REWARD_SIGNS = {
    "track_linear_velocity": +1,
    "track_angular_velocity": +1,
    "upright": +1,
    "air_time": +1,
    "upper_body_posture": -1,
    "standing_pose_l1": -1,
    "foot_clearance": -1,
    "foot_swing_height": -1,
    "foot_slip": -1,
    "soft_landing": -1,
    "body_ang_vel": -1,
    "angular_momentum": -1,
    "dof_pos_limits": -1,
    "action_rate": -1,
    "self_collisions": -1,
}

# The source recipe's motor table, keyed by the joint-name suffix the left and
# right copies share.
REF_KP = {
    "Hip_Pitch": 80.0,
    "Hip_Roll": 80.0,
    "Hip_Yaw": 80.0,
    "Knee_Pitch": 80.0,
    "Ankle_Pitch": 50.0,
    "Ankle_Roll": 50.0,
    "Shoulder_Pitch": 10.0,
    "Shoulder_Roll": 10.0,
    "Elbow_Pitch": 10.0,
    "Elbow_Yaw": 10.0,
    "Head_Yaw": 4.0,
    "Head_Pitch": 4.0,
}
REF_KD = {
    "Hip_Pitch": 4.0,
    "Hip_Roll": 4.0,
    "Hip_Yaw": 4.0,
    "Knee_Pitch": 4.0,
    "Ankle_Pitch": 2.0,
    "Ankle_Roll": 2.0,
    "Shoulder_Pitch": 1.0,
    "Shoulder_Roll": 1.0,
    "Elbow_Pitch": 1.0,
    "Elbow_Yaw": 1.0,
    "Head_Yaw": 0.25,
    "Head_Pitch": 0.25,
}

# The one gain this port does not reproduce, and why. The source integrates its
# PD damping inside the solver, where any value is stable; this port applies it
# as a force, which needs dt < 2*J/kd. The elbow pitch carries 0.0024 kg m^2, so
# the source's 1.0 sits just past that limit at a 5 ms step and the damping term
# amplifies instead of removing velocity. Listed rather than folded into the
# table above so the departure has to be acknowledged, not discovered.
KD_DEPARTURES = {
    "Shoulder_Pitch": 0.45,
    "Shoulder_Roll": 0.45,
    "Elbow_Pitch": 0.45,
    "Elbow_Yaw": 0.45,
}
# Explicit-damping stability margin (dt * kd / J must stay under 2) that every
# joint has to clear at the home pose.
MIN_DAMPING_MARGIN = 2.0
REF_EFFORT = {
    "Hip_Pitch": 68.0,
    "Hip_Roll": 76.0,
    "Hip_Yaw": 38.3,
    "Knee_Pitch": 112.0,
    "Ankle_Pitch": 38.3,
    "Ankle_Roll": 38.3,
    "Shoulder_Pitch": 14.0,
    "Shoulder_Roll": 14.0,
    "Elbow_Pitch": 14.0,
    "Elbow_Yaw": 14.0,
    "Head_Yaw": 6.0,
    "Head_Pitch": 6.0,
}
REF_ARMATURE = {
    "Hip_Pitch": 0.0478125,
    "Hip_Roll": 0.0339552,
    "Hip_Yaw": 0.0282528,
    "Knee_Pitch": 0.095625,
    "Ankle_Pitch": 0.0565056,
    "Ankle_Roll": 0.0565056,
    "Shoulder_Pitch": 0.001,
    "Shoulder_Roll": 0.001,
    "Elbow_Pitch": 0.001,
    "Elbow_Yaw": 0.001,
    "Head_Yaw": 0.001,
    "Head_Pitch": 0.001,
}
REF_HOME = {
    "Left_Shoulder_Roll": -1.4,
    "Right_Shoulder_Roll": 1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Elbow_Yaw": 0.4,
    "Left_Hip_Pitch": -0.4,
    "Right_Hip_Pitch": -0.4,
    "Left_Knee_Pitch": 0.8,
    "Right_Knee_Pitch": 0.8,
    "Left_Ankle_Pitch": -0.4,
    "Right_Ankle_Pitch": -0.4,
}

# Contact parameters the asset carries, from the source's collision config.
REF_FOOT_FRICTION = 1.0
REF_BODY_FRICTION = 0.45
REF_SOLREF_TIMECONST = 0.01
REF_CONDIM = 3
REF_NUM_COLLISION_GEOMS = 22

STEPS_PER_ROLLOUT = 24

# Joints where the asset file's armature is a DIFFERENT VALUE from the motor
# table the actuator config applies, not merely a rounded one. The file gives
# the head 0.002 where the motor is rated 0.001; the config wins, and the
# source recipe carries the same split.
KNOWN_ARMATURE_OVERRIDES = ["Head_Pitch", "Head_Yaw"]

# Below this relative difference the asset is quoting the same number to fewer
# digits (the ankle pair is written 0.0565 for 0.0565056), which says nothing
# about the plant because the config's value is the one loaded.
ARMATURE_ROUNDING_REL_TOL = 1e-3

# Robot bodies, from the asset. The non-foot ground group must carry every one
# of them except the two feet.
NUM_ROBOT_BODIES = 25

# A drop onto the ground bounces, so a non-foot body may graze it for a step
# or two before the robot settles. What would be wrong is the group firing
# constantly, which is what a mis-specified exclusion looks like.
MAX_NON_FOOT_HIT_RATE = 0.02


def leaf(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def suffix(joint: str) -> str:
    return leaf(joint).replace("Left_", "").replace("Right_", "")


class Checker:
    def __init__(self) -> None:
        self.fails: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.fails.append(name)
        return ok


def ref(values) -> torch.Tensor:
    """A float64 CPU tensor of expected values.

    The device is stated explicitly because one of the backends sets torch's
    global default device to the GPU on import, which would otherwise put
    these reference tensors somewhere the values read back from the
    environment are not.
    """
    return torch.tensor(values, dtype=torch.float64, device="cpu")


def mjcf_joint_limits(path: str) -> dict[str, tuple[float, float]]:
    """Hinge joint ranges read straight off the asset file."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    out: dict[str, tuple[float, float]] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        out[name] = (float(model.jnt_range[j][0]), float(model.jnt_range[j][1]))
    return out


def mjcf_contact_params(path: str) -> dict[str, dict]:
    """Contact parameters of every collision geom, read off the asset file."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    out: dict[str, dict] = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if not name or not name.endswith("_collision"):
            continue
        out[name] = {
            "friction": float(model.geom_friction[g][0]),
            "condim": int(model.geom_condim[g]),
            "priority": int(model.geom_priority[g]),
            "solref_timeconst": float(model.geom_solref[g][0]),
            "contype": int(model.geom_contype[g]),
            "conaffinity": int(model.geom_conaffinity[g]),
        }
    return out


def mjcf_joint_armature(path: str) -> dict[str, float]:
    """Armature the asset file declares for each hinge joint."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(path)
    out: dict[str, float] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        out[name] = float(model.dof_armature[model.jnt_dofadr[j]])
    return out


def explicit_damping_margins(
    path: str, home: dict[str, float], kd: dict[str, float], dt: float
) -> dict[str, tuple[float, float]]:
    """Per joint: its effective inertia at the home pose, and how much room the
    explicit PD damping has before it amplifies instead of removing velocity.

    The PD torque here is computed outside the solver and applied as a force,
    which is stable only while ``dt < 2 * J / kd``. Past that limit the damping
    term overshoots the velocity it is meant to cancel and hands it back
    reversed and larger, once per step. The joint shakes, and it looks exactly
    like a policy that never learned to hold still.

    Nothing else in this diagnostic catches that: every gain matches its table,
    every reward is right, the observation is right, and the robot trembles
    anyway. Which is why this check exists.

    Returns ``{joint: (effective inertia, margin)}``; a margin of 1.0 IS the
    stability limit.
    """
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    for name, angle in home.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = angle
    mujoco.mj_forward(model, data)
    mass_matrix = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, mass_matrix)

    out: dict[str, tuple[float, float]] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        inertia = float(mass_matrix[model.jnt_dofadr[j], model.jnt_dofadr[j]])
        out[name] = (inertia, (2.0 * inertia / kd[suffix(name)]) / dt)
    return out


def per_joint_from_actuators(act, attr: str, joints: list[str]) -> torch.Tensor | None:
    """Assemble a full-width per-joint tensor from the explicit actuators.

    ``act_manager._actuators`` holds ``(actuator, joint_indices)`` pairs, each
    actuator carrying only the joints it drives, so a preset that splits its
    joints across several actuator groups still produces one vector here.
    Joints no actuator claims stay NaN, which the caller reports rather than
    quietly averaging away.
    """
    pairs = []
    for actuator, joint_indices in getattr(act, "_actuators", []):
        value = getattr(actuator, attr, None)
        if isinstance(value, torch.Tensor):
            pairs.append((value.detach().double().cpu(), torch.as_tensor(joint_indices).cpu()))
    if not pairs:
        return None
    rows = max(v.shape[0] if v.dim() > 1 else 1 for v, _ in pairs)
    out = torch.full((rows, len(joints)), float("nan"), dtype=torch.float64, device="cpu")
    for value, indices in pairs:
        if value.dim() == 1:
            value = value.unsqueeze(0).expand(rows, -1)
        out[:, indices] = value
    return out


def _as_row(value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().double().cpu()
    return value[0] if value.dim() > 1 else value


def section_a_asset(env, cfg, chk: Checker) -> None:
    print(f"\n=== A. asset ({cfg.sim_type}) ===")
    robot: K1Config = cfg.robot
    mjcf = str(Path(robot.mjcf_path).resolve())

    joints = [leaf(n) for n in env.act_manager.actuated_joint_names]
    print(f"  joint order ({len(joints)}): {joints}")
    ref_limits = mjcf_joint_limits(mjcf)

    chk("22 actuated joints", len(joints) == 22, str(len(joints)))
    chk(
        "joint set == MJCF hinge joints",
        set(joints) == set(ref_limits),
        f"only-env {sorted(set(joints) - set(ref_limits))} " f"only-mjcf {sorted(set(ref_limits) - set(joints))}",
    )

    rd = env.get_entity_data("robot")
    soft_lo, soft_hi = rd.soft_joint_pos_limits
    factor = robot.soft_joint_pos_limit_factor
    exp_lo, exp_hi = [], []
    for j in joints:
        lo, hi = ref_limits[j]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
        exp_lo.append(mid - half)
        exp_hi.append(mid + half)
    want_lo = ref(exp_lo)
    want_hi = ref(exp_hi)
    got_lo, got_hi = _as_row(soft_lo), _as_row(soft_hi)
    chk(
        "soft limits == MJCF range narrowed by the configured factor",
        torch.allclose(got_lo, want_lo, atol=1e-5) and torch.allclose(got_hi, want_hi, atol=1e-5),
        f"max|dlo|={float((got_lo - want_lo).abs().max()):.2e} "
        f"max|dhi|={float((got_hi - want_hi).abs().max()):.2e}",
    )

    act = env.act_manager
    for label, table, attr in [
        ("kp", REF_KP, "stiffness"),
        ("kd", REF_KD, "damping"),
        ("effort", REF_EFFORT, "effort_limit"),
    ]:
        tensor = per_joint_from_actuators(act, attr, joints)
        if tensor is None:
            chk(f"{label} readable off the actuators", False, f"no '{attr}' tensor found")
            continue
        if bool(torch.isnan(tensor).any()):
            unclaimed = [j for j, bad in zip(joints, torch.isnan(tensor[0]).tolist()) if bad]
            chk(f"{label}: every joint claimed by an actuator", False, f"unclaimed {unclaimed}")
            continue
        expected = dict(table)
        if label == "kd":
            expected.update(KD_DEPARTURES)
        want = ref([expected[suffix(j)] for j in joints])
        if label in ("kp", "kd") and tensor.shape[0] > 1:
            # Randomized per env; compare the across-env mean to the nominal.
            got = tensor.mean(dim=0)
            ok = torch.allclose(got, want, rtol=0.25)
            detail = f"across-env mean vs nominal, max rel {float(((got - want).abs() / want).max()):.3f}"
        else:
            got = tensor[0]
            ok = torch.allclose(got, want, atol=1e-6)
            detail = f"max|d|={float((got - want).abs().max()):.3e}"
        source_note = ""
        if label == "kd":
            source_note = (
                f" (arm departs from the source's " f"{REF_KD['Elbow_Pitch']} -> {KD_DEPARTURES['Elbow_Pitch']})"
            )
        chk(f"{label} == source motor table{source_note}", ok, detail)

    # Armature is written into the simulator's model, not held on the actuator.
    # The asset file and the source recipe's motor table do not agree on it:
    # the file gives the head joints twice the motor table's value, and the
    # actuator config is what wins at runtime. So the motor table is asserted,
    # and the set of joints where the file disagrees is pinned -- a new
    # disagreement means the asset changed under the config.
    mjcf_armature = mjcf_joint_armature(mjcf)
    want = ref([REF_ARMATURE[suffix(j)] for j in joints])

    from jaxrlworld.rl.utils import string as string_utils

    _, names, values = string_utils.resolve_matching_names_values(robot.armature, joints, preserve_order=True)
    chk("robot config armature covers every joint", len(names) == len(joints), f"{len(names)} of {len(joints)}")
    if len(names) == len(joints):
        configured = dict(zip(names, values))
        got = ref([configured[j] for j in joints])
        chk(
            "robot config armature == source motor table (this is what runs)",
            torch.allclose(got, want, atol=1e-9),
            f"max|d|={float((got - want).abs().max()):.3e}",
        )
        rounded, overridden = [], []
        for j in joints:
            asset_value, config_value = mjcf_armature[j], configured[j]
            if abs(asset_value - config_value) <= 1e-12:
                continue
            relative = abs(asset_value - config_value) / abs(config_value)
            (rounded if relative < ARMATURE_ROUNDING_REL_TOL else overridden).append(j)
        if rounded:
            print(
                f"        asset rounds armature on {len(rounded)} joints, "
                f"e.g. {rounded[0]} {mjcf_armature[rounded[0]]} for {configured[rounded[0]]}"
            )
        chk(
            "asset states a different armature only on the head joints",
            sorted(overridden) == KNOWN_ARMATURE_OVERRIDES,
            f"{[(j, mjcf_armature[j], configured[j]) for j in sorted(overridden)]}",
        )

    got = _as_row(act._scale)
    want = ref([0.25 * REF_EFFORT[suffix(j)] / REF_KP[suffix(j)] for j in joints])
    chk(
        "action scale == 0.25 * effort / kp",
        torch.allclose(got, want, atol=1e-9),
        f"max|d|={float((got - want).abs().max()):.3e}",
    )

    want_home = ref([REF_HOME.get(j, 0.0) for j in joints])
    got_offset = _as_row(act.offset)
    chk(
        "action offset == source home keyframe",
        torch.allclose(got_offset, want_home, atol=1e-9),
        f"max|d|={float((got_offset - want_home).abs().max()):.3e}",
    )
    got_default = _as_row(rd.default_joint_pos)
    chk(
        "default joint pos == source home keyframe",
        torch.allclose(got_default, want_home, atol=1e-6),
        f"max|d|={float((got_default - want_home).abs().max()):.3e}",
    )

    print("\n  -- explicit PD damping: dt < 2*J/kd, at the home pose --")
    from jaxrlworld.rl.configs.presets.k1_velocity.base import _SIM_TIMINGS

    dt = _SIM_TIMINGS[cfg.sim_type]["dt"]
    kd_table = dict(REF_KD)
    kd_table.update(KD_DEPARTURES)
    home_angles = {j: REF_HOME.get(j, 0.0) for j in joints}
    margins = explicit_damping_margins(mjcf, home_angles, kd_table, dt)
    tightest = sorted(margins.items(), key=lambda kv: kv[1][1])[:3]
    for name, (inertia, margin) in tightest:
        print(f"        {name:22s} J={inertia:8.5f}  kd={kd_table[suffix(name)]:.2f}  " f"margin={margin:5.2f}x")
    worst_name, (worst_inertia, worst_margin) = tightest[0]
    chk(
        f"every joint clears the explicit-damping limit by {MIN_DAMPING_MARGIN:g}x",
        worst_margin >= MIN_DAMPING_MARGIN,
        f"tightest is {worst_name} at {worst_margin:.2f}x "
        f"(J={worst_inertia:.5f}, kd={kd_table[suffix(worst_name)]:.2f}, dt={dt})",
    )

    print("\n  -- collision geometry, read off the file all three backends parse --")
    params = mjcf_contact_params(mjcf)
    chk(
        f"{REF_NUM_COLLISION_GEOMS} collision geoms",
        len(params) == REF_NUM_COLLISION_GEOMS,
        str(len(params)),
    )
    foot_geoms = set(robot.foot_geom_names)
    chk(
        "both feet present in the collision set",
        foot_geoms <= set(params),
        f"missing {sorted(foot_geoms - set(params))}",
    )
    bad = []
    for name, p in sorted(params.items()):
        want_friction = REF_FOOT_FRICTION if name in foot_geoms else REF_BODY_FRICTION
        if (
            abs(p["friction"] - want_friction) > 1e-9
            or p["condim"] != REF_CONDIM
            or p["contype"] != 1
            or p["conaffinity"] != 1
            or abs(p["solref_timeconst"] - REF_SOLREF_TIMECONST) > 1e-9
        ):
            bad.append((name, p))
    chk("every collision geom carries the source's contact parameters", not bad, str(bad[:2]))


def section_b_mdp(env, cfg, chk: Checker) -> None:
    print("\n=== B. MDP ===")
    dims = env.calculate_obs_dim()
    chk(
        f"actor {ACTOR_DIM} / critic {CRITIC_DIM}",
        dims["actor"] == ACTOR_DIM and dims["critic"] == CRITIC_DIM,
        str(dict(dims)),
    )

    print("\n  -- reward term signs over a random then saturating rollout --")
    print("     a penalty that comes back positive is the one error here that trains silently")

    # Prove the saturating phases below are sized to do their job, using the
    # runtime offset and scale rather than the config, so that a change to
    # either cannot quietly leave a joint unreached.
    rd = env.get_entity_data("robot")
    act = env.act_manager
    joints = [leaf(n) for n in act.actuated_joint_names]
    soft_lo, soft_hi = (_as_row(t) for t in rd.soft_joint_pos_limits)
    offset, scale = _as_row(act.offset), _as_row(act._scale)
    target_hi = offset + scale * SATURATING_ACTION
    target_lo = offset - scale * SATURATING_ACTION
    commanded_out = (target_hi > soft_hi) | (target_lo < soft_lo)
    unreached = [j for j, ok in zip(joints, commanded_out.tolist()) if not ok]
    # The tightest joint is the one whose better of the two overshoots is
    # smallest -- that is the margin the constant has to keep positive.
    worst = int(torch.maximum(target_hi - soft_hi, soft_lo - target_lo).argmin())
    chk(
        f"|action| = {SATURATING_ACTION:g} commands every joint past a soft position limit",
        not unreached,
        f"unreached {unreached}",
    )
    print(
        f"        tightest margin at {joints[worst]}: "
        f"target {float(target_hi[worst]):+.3f} / {float(target_lo[worst]):+.3f} rad "
        f"vs soft limits {float(soft_lo[worst]):+.3f}..{float(soft_hi[worst]):+.3f}"
    )
    # A term that never fires is a term whose sign was never checked, so the
    # rollout has to reach every one of them on purpose. Random actions cover
    # the ordinary terms but only wander near the joint limits; the two
    # saturating phases drive every joint hard against one soft limit and then
    # the other. The joint-limit penalty stopped firing on its own the moment
    # the arm damping changed -- the plant moved and a term went quietly
    # untested, which is exactly what this phase list exists to prevent.
    phases = (
        ("random", 60, None),
        ("saturated high", 25, +SATURATING_ACTION),
        ("saturated low", 25, -SATURATING_ACTION),
    )
    seen_nonzero: set[str] = set()
    first_seen: dict[str, str] = {}
    wrong_sign: dict[str, float] = {}
    extrema: dict[str, tuple[float, float]] = {}

    def accumulate(phase_name: str) -> None:
        for name, value in env.rew_buf_per_type.items():
            v = value.detach()
            lo, hi = float(v.min()), float(v.max())
            prev = extrema.get(name, (lo, hi))
            extrema[name] = (min(prev[0], lo), max(prev[1], hi))
            if float(v.abs().max()) > 1e-12:
                seen_nonzero.add(name)
                first_seen.setdefault(name, phase_name)
            want = REWARD_SIGNS.get(name)
            if want is None:
                continue
            if (want > 0 and lo < -1e-9) or (want < 0 and hi > 1e-9):
                offending = hi if want < 0 else lo
                wrong_sign[name] = max(abs(offending), abs(wrong_sign.get(name, 0.0)))

    torch.manual_seed(0)
    for phase_name, steps, drive in phases:
        for _ in range(steps):
            if drive is None:
                action = (torch.rand(env.num_envs, env.num_actions, device=env.device) - 0.5) * 2.0
            else:
                action = torch.full((env.num_envs, env.num_actions), drive, device=env.device)
            env.step(action)
            accumulate(phase_name)

    chk(
        "every configured reward term reports a value",
        set(REWARD_SIGNS) <= set(extrema),
        f"missing {sorted(set(REWARD_SIGNS) - set(extrema))}",
    )
    print(f"     {'term':24s} {'min':>13s} {'max':>13s}  {'kind':8s} first nonzero")
    for name in sorted(extrema):
        lo, hi = extrema[name]
        want = REWARD_SIGNS.get(name)
        kind = "" if want is None else ("reward" if want > 0 else "penalty")
        print(f"     {name:24s} {lo:13.6f} {hi:13.6f}  {kind:8s} " f"{first_seen.get(name, 'NEVER')}")
    chk("no term violates its sign", not wrong_sign, str(wrong_sign))
    dead = sorted(set(REWARD_SIGNS) - seen_nonzero)
    chk(
        "no term is identically zero, which would make its sign check vacuous",
        not dead,
        str(dead),
    )


def section_c_contacts(env, cfg, chk: Checker) -> None:
    print("\n=== C. contact groups ===")
    manager = env.contact_manager
    for group in ("feet_ground_contact", "non_foot_ground_contact", "self_collision"):
        try:
            force = manager.contact_force(group)
            chk(f"{group} registered", force is not None, f"shape {tuple(force.shape)}")
        except Exception as exc:  # noqa: BLE001 - the group name is what is under test
            chk(f"{group} registered", False, f"{type(exc).__name__}: {exc}")

    # The exclusion is structural, so check it structurally: the group must
    # carry every robot body but the two feet.
    non_foot_columns = manager.contact_force("non_foot_ground_contact").shape[1]
    chk(
        "non-foot group covers every body except the two feet",
        non_foot_columns == NUM_ROBOT_BODIES - 2,
        f"{non_foot_columns} columns, expected {NUM_ROBOT_BODIES - 2}",
    )
    feet_columns = manager.contact_force("feet_ground_contact").shape[1]
    chk("feet group covers exactly the two feet", feet_columns == 2, f"{feet_columns} columns")

    env.reset()
    steps = 80
    touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    non_foot_hits = 0
    foot_hits = 0
    for _ in range(steps):
        env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        feet = manager.contact_force("feet_ground_contact")
        hit = feet.flatten(1).abs().amax(dim=1) > 1.0
        touched |= hit
        foot_hits += int(hit.sum())
        non_foot = manager.contact_force("non_foot_ground_contact")
        non_foot_hits += int((non_foot.flatten(1).abs().amax(dim=1) > 1.0).sum())
    env_steps = steps * env.num_envs
    rate = non_foot_hits / env_steps
    chk(
        "feet reach the ground when released from the home pose",
        bool(touched.all()),
        f"{int(touched.sum())}/{env.num_envs} envs",
    )
    chk(
        "non-foot ground contact is rare while settling upright",
        rate <= MAX_NON_FOOT_HIT_RATE,
        f"{non_foot_hits}/{env_steps} env-steps ({rate:.4%}), limit {MAX_NON_FOOT_HIT_RATE:.0%}",
    )
    chk(
        "the feet touch the ground far more than anything else does",
        foot_hits > 10 * max(non_foot_hits, 1),
        f"feet {foot_hits} vs non-foot {non_foot_hits} env-steps",
    )


def _termination_bool(result) -> torch.Tensor:
    """The per-env mask a termination term returns."""
    return result.reset.bool()


def section_d_terminations(env, cfg, chk: Checker) -> None:
    print("\n=== D. terminations ===")
    from jaxrlworld.rl.envs.mdp.terminations import k1_velocity as booster_tf

    limit = math.radians(cfg.fall_limit_angle_deg)
    probability = cfg.fall_probability

    env.reset()
    angle = booster_tf.tilt_angle(env)
    chk(
        "tilt angle reads near zero at the home pose",
        float(angle.abs().max()) < 0.35,
        f"max {math.degrees(float(angle.max())):.2f} deg",
    )

    # Lay every robot on its side and re-read, so the measure is checked at a
    # second known attitude rather than only at zero.
    writer = env.get_robot_state_writer("robot")
    env_ids = torch.arange(env.num_envs, device=env.device)
    pos = env.scene_manager.env_origins.clone()
    pos[:, 2] = 0.3
    half = math.pi / 4.0  # a 90 deg roll
    quat = torch.tensor([math.cos(half), math.sin(half), 0.0, 0.0], device=env.device)
    writer.set_root_pose(pos, quat.repeat(env.num_envs, 1), env_ids=env_ids)
    writer.eval_fk(env_ids=env_ids)
    env._post_reset_forward()
    env._invalidate_cache()

    angle = booster_tf.tilt_angle(env)
    chk(
        "tilt angle reads ~90 deg when laid on its side",
        abs(float(angle.mean()) - math.pi / 2) < 0.05,
        f"mean {math.degrees(float(angle.mean())):.2f} deg",
    )

    fired = _termination_bool(booster_tf.stochastic_bad_orientation(env, limit_angle=limit, probability=1.0))
    chk(
        "deterministic form fires for every tilted env",
        bool(fired.all()),
        f"{int(fired.sum())}/{env.num_envs}",
    )

    # The draw is Bernoulli, so the band is the binomial standard error of the
    # sample, not a round number. Several seeds keep one unlucky sequence from
    # deciding the result -- a single seed at the earlier sample size landed
    # nearly three standard errors low, which says nothing about the term.
    trials_per_seed = 500
    seeds = range(5)
    total = 0
    draws = 0
    for seed in seeds:
        torch.manual_seed(seed)
        for _ in range(trials_per_seed):
            total += int(
                _termination_bool(
                    booster_tf.stochastic_bad_orientation(env, limit_angle=limit, probability=probability)
                ).sum()
            )
            draws += env.num_envs
    rate = total / draws
    standard_error = math.sqrt(probability * (1.0 - probability) / draws)
    band = 4.0 * standard_error
    chk(
        "stochastic form fires at its configured rate",
        abs(rate - probability) <= band,
        f"measured {rate:.5f} ({total}/{draws}) vs configured {probability}, " f"4 standard errors = {band:.5f}",
    )


def section_e_curriculum(env, cfg, chk: Checker) -> None:
    print("\n=== E. curriculum ===")
    manager = env.curriculum_manager
    soft_cfg = env.reward_manager.get_term_cfg("soft_landing")
    command_term = env.command_manager.get_term("velocity")
    # (step, expected soft-landing weight, expected vx range)
    expectations = [
        (0, 0.0001, (-1.0, 1.2)),
        (1000 * STEPS_PER_ROLLOUT, 0.001, (-1.0, 1.2)),
        (5000 * STEPS_PER_ROLLOUT, 0.001, (-1.0, 1.5)),
        (7000 * STEPS_PER_ROLLOUT, 0.005, (-1.0, 1.5)),
        (10000 * STEPS_PER_ROLLOUT, 0.005, (-1.25, 1.5)),
        (15000 * STEPS_PER_ROLLOUT, 0.005, (-1.5, 1.75)),
    ]
    original = env.env_step_counter
    env_ids = torch.arange(env.num_envs, device=env.device)
    for step, want_weight, want_vx in expectations:
        env.env_step_counter = step
        manager.compute(env_ids)
        got_weight = float(soft_cfg.weight)
        got_vx = tuple(float(v) for v in command_term.cfg.lin_vel_x_range)
        chk(
            f"step {step}: soft-landing weight",
            abs(got_weight - want_weight) < 1e-12,
            f"{got_weight} vs {want_weight}",
        )
        chk(f"step {step}: command vx range", got_vx == want_vx, f"{got_vx} vs {want_vx}")
    env.env_step_counter = original


def source_push_event(path: Path) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    """``(interval_range_s, velocity_range)`` of the source's push, read off its config file."""
    text = path.read_text()
    match = re.search(r'"push_robot":\s*EventTermCfg\((.*?)\n\s{8}\),', text, re.S)
    if match is None:
        raise ValueError(f"no push_robot block in {path}")
    block = match.group(1)
    func = re.search(r"func=mdp\.(\w+)", block).group(1)
    if func != "push_by_setting_velocity":
        raise ValueError(f"source push is {func}, this diag expects push_by_setting_velocity")
    interval = tuple(float(v) for v in re.search(r"interval_range_s=\(([^)]*)\)", block).group(1).split(","))
    ranges = {
        key: (float(lo), float(hi))
        for key, lo, hi in re.findall(r'"(x|y|z|roll|pitch|yaw)":\s*\(([-\d.]+),\s*([-\d.]+)\)', block)
    }
    if set(ranges) != {"x", "y", "z", "roll", "pitch", "yaw"}:
        raise ValueError(f"source push block lists {sorted(ranges)}, expected all six axes")
    return interval, ranges


def section_f_push(env, built, chk: Checker) -> None:
    print("\n=== F. push event ===")
    terms = iter_terms(built.event, EventTermConfig)
    chk("push term configured", "push" in terms, f"terms {sorted(terms)}")
    push = terms["push"]
    chk(
        "push adds a sampled root velocity (push_by_setting_velocity)",
        push.resolved_func is ev.push_by_setting_velocity,
        push.resolved_func.__name__,
    )
    chk("push mode is interval", push.mode == "interval", push.mode)
    got_ranges = {k: tuple(v) for k, v in push.params["velocity_range"].items()}
    chk("push kicks all six axes", set(got_ranges) == {"x", "y", "z", "roll", "pitch", "yaw"}, str(sorted(got_ranges)))
    print(f"  configured: interval {tuple(push.interval_range_s)}, ranges {got_ranges}")

    # Against the source's own config file. The file is not part of what the
    # training machine needs, so its absence is reported as a failed check
    # with the path to sync, not as a crash and not as a pass.
    if SOURCE_VELOCITY_CFG.is_file():
        src_interval, src_ranges = source_push_event(SOURCE_VELOCITY_CFG)
        chk(
            "push interval == source",
            tuple(push.interval_range_s) == src_interval,
            f"{tuple(push.interval_range_s)} vs source {src_interval}",
        )
        chk(
            "push velocity ranges == source, all six axes",
            got_ranges == src_ranges,
            f"{got_ranges} vs source {src_ranges}",
        )
    else:
        chk(
            "push interval and ranges == source config file",
            False,
            f"{SOURCE_VELOCITY_CFG} is not on this machine; sync it to run the comparison",
        )

    # Land a deterministic kick of the configured maxima on every env and read
    # it back through the same data the observations use. A backend whose
    # writer drops the angular part, or applies the kick in the wrong frame,
    # would pass every config check above and still train against a
    # different disturbance.
    env.reset()
    rd = env.get_entity_data("robot")
    kick = {k: (hi, hi) for k, (_, hi) in got_ranges.items()}
    lin_before = rd.root_link_lin_vel_w.clone()
    ang_before = rd.root_link_ang_vel_w.clone()
    env_ids = torch.arange(env.num_envs, device=env.device)
    ev.push_by_setting_velocity(env, env_ids, velocity_range=kick)
    # What World.step does right after applying interval events: bump the
    # read-cache generation. Newton memoizes derived root reads per
    # generation, and root_link_lin_vel_w is a fresh tensor (CoM -> origin
    # transfer) while root_link_ang_vel_w is a live view, so without the bump
    # the linear read here reports the pre-kick value and the angular read
    # the post-kick one -- the training path never sees that, this diag did.
    env._invalidate_cache()
    env._post_reset_forward()
    d_lin = rd.root_link_lin_vel_w - lin_before
    d_ang = rd.root_link_ang_vel_w - ang_before
    want_lin = torch.tensor([kick["x"][1], kick["y"][1], kick["z"][1]], device=d_lin.device)
    want_ang = torch.tensor([kick["roll"][1], kick["pitch"][1], kick["yaw"][1]], device=d_ang.device)
    err_lin = float((d_lin - want_lin).abs().max())
    err_ang = float((d_ang - want_ang).abs().max())
    chk(
        "linear kick lands on the root velocity in the world frame",
        err_lin < 1e-4,
        f"max|d| {err_lin:.2e}, got {d_lin[0].tolist()} want {want_lin.tolist()}",
    )
    chk(
        "angular kick lands on the root angular velocity",
        err_ang < 1e-4,
        f"max|d| {err_ang:.2e}, got {d_ang[0].tolist()} want {want_ang.tolist()}",
    )
    env.reset()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=["mujoco", "newton", "genesis"], required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    args = parser.parse_args()

    cfg = K1VelocityConfig(sim_type=args.sim, num_envs=args.num_envs)
    built = cfg.build()
    env = BaseRunner._create_env_from_config(built)
    env.reset()

    chk = Checker()
    section_a_asset(env, cfg, chk)
    section_b_mdp(env, cfg, chk)
    section_c_contacts(env, cfg, chk)
    section_d_terminations(env, cfg, chk)
    section_e_curriculum(env, cfg, chk)
    section_f_push(env, built, chk)

    print("\n=== RESULT:", "ALL OK" if not chk.fails else f"{len(chk.fails)} FAIL: {chk.fails}", "===")
    return 1 if chk.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
