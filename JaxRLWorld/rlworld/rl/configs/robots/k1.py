"""Booster K1 humanoid robot configuration (22 actuated DOF).

The kinematics/home keyframe are read verbatim from the MJCF at
``assets/K1/k1_mjx_feetonly.xml``. The actuator bundle (PD gains, effort,
armature, action-scale, torque-speed limits) is picked by the
``K1_ACTUATOR_MODE`` env var (see below):

- ``legacy`` (default): the hand-tuned setup. PD gains from ``PD_PROFILE``
  (only ``"stiff_legs"`` = arms 15, hip/knee 50, ankle 15, ``kd`` arm 2,
  leg 5; ``K1_PD_PROFILE`` env var), derated MJCF effort limits
  (``actuatorfrcrange``: hip 30, knee 40, ...), the recipe's own action
  scale, and plain PD (no torque-speed curve). Reproduces earlier runs.
- ``physical``: the Booster motor spec the real K1 runs (booster_train
  ``actuator.py``/``booster.py``). Per-joint kp = J·ω_n², kd = 2ζ·J·ω_n,
  the real motor effort rating (2-3× the MJCF: hip 68, knee 112, ...),
  action_scale = 0.25·effort/kp, plus the piecewise-linear torque-speed
  curve (``velocity_limit``/``knee_point_velocity``). Legs run at
  ω_n = 4 Hz (knee ζ=1, other legs ζ=1.5), arms/head at 10 Hz / ζ=2.

Shared by both modes:

- frictionloss 0.1 uniformly on every joint.
- armature per joint from the Booster reference model
  ``assets/K1/K1_22dof.xml`` (head 0.002, arm 0.001, legs
  0.028-0.096; legacy and physical use the same values).  The feetonly
  MJCF flattens all of these to a uniform 0.005, which understates the
  leg reflected rotor inertia by roughly an order of magnitude; the
  actuator configs override the XML value on every backend, so the
  reference values win at runtime.

Joint-name convention is the Booster family one (same as T1):
``AAHead_yaw``/``Head_pitch``, ``A{Left,Right}_Shoulder_Pitch``,
``{Left,Right}_{Shoulder_Roll,Elbow_Pitch,Elbow_Yaw}``, and
``{Left,Right}_{Hip_{Pitch,Roll,Yaw},Knee_Pitch,Ankle_{Pitch,Roll}}``.
K1 has no waist joint.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List

from rlworld.rl.configs.robots.base import RobotConfig

# ── PD gain profiles (kp = stiffness, kd = damping) ───────────────────
# Two coherent gain sets selected by ``PD_PROFILE``. Both keep the softer
# arms (kp 15, kd 2) and differ only in the legs; kd is scaled with kp so the
# legs stay well-damped (the Booster SDK's kp-80/kd-2 legs shook badly). The
# resolved kp/kd are baked into config.yaml per checkpoint.
_PD_PROFILES: Dict[str, Dict[str, float]] = {
    # Default: stiffer, well-damped legs that hold the body under load (kp 25
    # sagged / looked underpowered on the real robot).
    "stiff_legs": {
        "stiffness_head": 15.0,
        "stiffness_arm": 15.0,
        "stiffness_hip": 50.0,
        "stiffness_knee": 50.0,
        "stiffness_ankle": 15.0,
        "damping_head": 2.0,
        "damping_arm": 2.0,
        "damping_leg": 5.0,
    },
}

# Active legacy profile (only ``stiff_legs`` remains). Selected by the
# ``K1_PD_PROFILE`` env var; used only when ``K1_ACTUATOR_MODE=legacy`` (the
# default). The resolved kp/kd are baked into the saved config.yaml, so each
# checkpoint records the gains it trained with.
PD_PROFILE = os.environ.get("K1_PD_PROFILE", "stiff_legs")
if PD_PROFILE not in _PD_PROFILES:
    raise ValueError(f"K1_PD_PROFILE={PD_PROFILE!r} unknown; choose from {sorted(_PD_PROFILES)}")
_pd = _PD_PROFILES[PD_PROFILE]

STIFFNESS_HEAD = _pd["stiffness_head"]
STIFFNESS_ARM = _pd["stiffness_arm"]
STIFFNESS_HIP = _pd["stiffness_hip"]
STIFFNESS_KNEE = _pd["stiffness_knee"]
STIFFNESS_ANKLE = _pd["stiffness_ankle"]
DAMPING_HEAD = _pd["damping_head"]
DAMPING_ARM = _pd["damping_arm"]
DAMPING_LEG = _pd["damping_leg"]

# Armature (reflected rotor inertia) per joint group, read from the
# Booster reference model ``assets/K1/K1_22dof.xml``.  These override
# the feetonly MJCF's flattened uniform 0.005 (see module docstring).
ARMATURE_HEAD = 0.002
ARMATURE_ARM = 0.001
ARMATURE_HIP_PITCH = 0.0478125
ARMATURE_HIP_ROLL = 0.0339552
ARMATURE_HIP_YAW = 0.0282528
ARMATURE_KNEE = 0.095625
ARMATURE_ANKLE = 0.0565

# ``actuatorfrcrange`` magnitudes per class.
EFFORT_HEAD = 6.0
EFFORT_SHOULDER = 14.0
EFFORT_ELBOW = 14.0
EFFORT_HIP_PITCH = 30.0
EFFORT_HIP_ROLL = 35.0
EFFORT_HIP_YAW = 20.0
EFFORT_KNEE = 40.0
EFFORT_ANKLE = 20.0


# ── Joint-group regexes ───────────────────────────────────────────────
# Patterns are fullmatch'd against fully-qualified joint labels on each
# backend; the leading ``.*`` absorbs Newton's hierarchical MJCF XPath
# prefixes and harmlessly matches the empty string elsewhere (same
# convention as the T1 config).
_HEAD_PATTERNS = (r".*AAHead_yaw", r".*Head_pitch")
_SHOULDER_PATTERNS = (r".*_Shoulder_Pitch", r".*_Shoulder_Roll")
_ELBOW_PATTERNS = (r".*_Elbow_Pitch", r".*_Elbow_Yaw")
_HIP_PITCH_PATTERNS = (r".*_Hip_Pitch",)
_HIP_ROLL_PATTERNS = (r".*_Hip_Roll",)
_HIP_YAW_PATTERNS = (r".*_Hip_Yaw",)
_KNEE_PATTERNS = (r".*_Knee_Pitch",)
_ANKLE_PATTERNS = (r".*_Ankle_Pitch", r".*_Ankle_Roll")

_GROUPS: Dict[str, tuple] = {
    "head": _HEAD_PATTERNS,
    "shoulder": _SHOULDER_PATTERNS,
    "elbow": _ELBOW_PATTERNS,
    "hip_pitch": _HIP_PITCH_PATTERNS,
    "hip_roll": _HIP_ROLL_PATTERNS,
    "hip_yaw": _HIP_YAW_PATTERNS,
    "knee": _KNEE_PATTERNS,
    "ankle": _ANKLE_PATTERNS,
}


def _pattern_dict(value_by_group: Dict[str, float]) -> Dict[str, float]:
    """Flatten a per-group scalar into a per-regex dict."""
    out: Dict[str, float] = {}
    for group, patterns in _GROUPS.items():
        for p in patterns:
            out[p] = value_by_group[group]
    return out


# ── Actuator mode ─────────────────────────────────────────────────────
# ``K1_ACTUATOR_MODE`` toggles the whole actuator bundle (kp/kd, effort,
# action_scale, armature, torque-speed limits):
#
#   "legacy"   – the hand-tuned setup: PD gains from ``_PD_PROFILES``
#                (``K1_PD_PROFILE``, default stiff_legs), the derated MJCF
#                effort limits, per-recipe action_scale, plain PD (no
#                torque-speed curve). Reproduces earlier runs exactly.
#   "physical" – the Booster motor spec the REAL K1 runs (booster_train
#                actuator.py / booster.py): per-joint kp = J·ω_n²,
#                kd = 2ζ·J·ω_n, effort = the real motor rating (2-3× the
#                MJCF), action_scale = 0.25·effort/kp, plus the piecewise-
#                linear torque-speed curve (velocity_limit / knee point).
#                Legs run at ω_n = 4 Hz (knee ζ=1, others ζ=1.5); arms/head
#                at the 10 Hz / ζ=2 class default. Overrides the recipe's
#                action_scale.
#
# Per-group Booster motor specs (armature[kg·m²], effort[N·m], velocity_limit
# [rad/s], knee_point_velocity[rad/s], natural_freq[Hz], damping_ratio):
_BOOSTER_MOTOR: Dict[str, tuple] = {
    "head": (0.001, 6.0, 7.85, 10.47, 10.0, 2.0),  # HT4438
    "shoulder": (0.001, 14.0, 33.51, 5.24, 10.0, 2.0),  # R14
    "elbow": (0.001, 14.0, 33.51, 5.24, 10.0, 2.0),  # R14
    "hip_pitch": (0.0478125, 68.0, 14.66, 1.88, 4.0, 1.5),  # E6408
    "hip_roll": (0.0339552, 76.0, 12.57, 2.62, 4.0, 1.5),  # E4315
    "hip_yaw": (0.0282528, 38.3, 17.59, 7.85, 4.0, 1.5),  # E4310
    "knee": (0.095625, 112.0, 12.57, 2.09, 4.0, 1.0),  # E6416
    "ankle": (0.0565056, 38.3, 17.59, 7.85, 4.0, 1.5),  # E4310 parallel (armature ×2)
}


def _physical_bundle():
    """Derive kp/kd/action_scale/effort/armature/velocity_limit/knee_point
    per group from the Booster motor specs (matches booster_train's
    ``BoosterJointCfg`` + ``0.25·effort/kp`` action scale)."""
    kp, kd, ascale, eff, arm, vel, knee = ({} for _ in range(7))
    for g, (a, e, vlim, kpt, f, zeta) in _BOOSTER_MOTOR.items():
        wn = 2.0 * math.pi * f
        k = a * wn * wn
        kp[g] = k
        kd[g] = 2.0 * zeta * a * wn
        ascale[g] = 0.25 * e / k
        eff[g], arm[g], vel[g], knee[g] = e, a, vlim, kpt
    return kp, kd, ascale, eff, arm, vel, knee


K1_ACTUATOR_MODE = os.environ.get("K1_ACTUATOR_MODE", "legacy")
if K1_ACTUATOR_MODE not in ("legacy", "physical"):
    raise ValueError(f"K1_ACTUATOR_MODE={K1_ACTUATOR_MODE!r} unknown; choose 'legacy' or 'physical'")

if K1_ACTUATOR_MODE == "physical":
    _kp, _kd, _ascale, _eff, _arm, _vel, _knee = _physical_bundle()
    _P_GAINS = _pattern_dict(_kp)
    _D_GAINS = _pattern_dict(_kd)
    _ARMATURE = _pattern_dict(_arm)
    _EFFORT = _pattern_dict(_eff)
    _ACTION_SCALE = _pattern_dict(_ascale)
    _VELOCITY_LIMIT = _pattern_dict(_vel)
    _KNEE_POINT = _pattern_dict(_knee)
else:  # legacy
    _P_GAINS = _pattern_dict(
        {
            "head": STIFFNESS_HEAD,
            "shoulder": STIFFNESS_ARM,
            "elbow": STIFFNESS_ARM,
            "hip_pitch": STIFFNESS_HIP,
            "hip_roll": STIFFNESS_HIP,
            "hip_yaw": STIFFNESS_HIP,
            "knee": STIFFNESS_KNEE,
            "ankle": STIFFNESS_ANKLE,
        }
    )
    _D_GAINS = _pattern_dict(
        {
            "head": DAMPING_HEAD,
            "shoulder": DAMPING_ARM,
            "elbow": DAMPING_ARM,
            "hip_pitch": DAMPING_LEG,
            "hip_roll": DAMPING_LEG,
            "hip_yaw": DAMPING_LEG,
            "knee": DAMPING_LEG,
            "ankle": DAMPING_LEG,
        }
    )
    _ARMATURE = _pattern_dict(
        {
            "head": ARMATURE_HEAD,
            "shoulder": ARMATURE_ARM,
            "elbow": ARMATURE_ARM,
            "hip_pitch": ARMATURE_HIP_PITCH,
            "hip_roll": ARMATURE_HIP_ROLL,
            "hip_yaw": ARMATURE_HIP_YAW,
            "knee": ARMATURE_KNEE,
            "ankle": ARMATURE_ANKLE,
        }
    )
    _EFFORT = _pattern_dict(
        {
            "head": EFFORT_HEAD,
            "shoulder": EFFORT_SHOULDER,
            "elbow": EFFORT_ELBOW,
            "hip_pitch": EFFORT_HIP_PITCH,
            "hip_roll": EFFORT_HIP_ROLL,
            "hip_yaw": EFFORT_HIP_YAW,
            "knee": EFFORT_KNEE,
            "ankle": EFFORT_ANKLE,
        }
    )
    _ACTION_SCALE = None  # legacy: recipe decides the action scale
    _VELOCITY_LIMIT = None  # legacy: no torque-speed curve (plain PD)
    _KNEE_POINT = None


@dataclass
class K1Config(RobotConfig):
    """Configuration for Booster K1 humanoid robot (22 actuated DOF)."""

    name: str = "K1"
    urdf_path: str | None = None
    mjcf_path: str | None = "./JaxRLWorld/rlworld/assets/K1/k1_mjx_feetonly.xml"
    usd_path: str | None = None

    # Home keyframe: base at z=0.545, identity orientation.
    base_init_height: float = 0.545
    base_link_name: str = "Trunk"

    # Home keyframe joint targets (zeros omitted — dict is regex→angle
    # and unmatched joints default to 0).
    default_joint_angles: Dict[str, float] = field(
        default_factory=lambda: {
            r".*Left_Shoulder_Roll": -1.4,
            r".*Left_Elbow_Yaw": -0.4,
            r".*Right_Shoulder_Roll": 1.4,
            r".*Right_Elbow_Yaw": 0.4,
            r".*_Hip_Pitch": -0.2,
            r".*_Knee_Pitch": 0.4,
            r".*_Ankle_Pitch": -0.2,
        }
    )

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: [
            *_HEAD_PATTERNS,
            *_SHOULDER_PATTERNS,
            *_ELBOW_PATTERNS,
            *_HIP_PITCH_PATTERNS,
            *_HIP_ROLL_PATTERNS,
            *_HIP_YAW_PATTERNS,
            *_KNEE_PATTERNS,
            *_ANKLE_PATTERNS,
        ]
    )

    # PD gains, effort, armature, action-scale and torque-speed limits all come
    # from the resolved actuator bundle (``K1_ACTUATOR_MODE`` above): the
    # hand-tuned legacy set or the Booster physical motor spec.
    p_gains: Dict[str, float] = field(default_factory=lambda: dict(_P_GAINS))
    d_gains: Dict[str, float] = field(default_factory=lambda: dict(_D_GAINS))

    # frictionloss (0.1) needs no config field: it rides in via the MJCF
    # joint default class on every backend.
    armature: Dict[str, float] = field(default_factory=lambda: dict(_ARMATURE))

    # Per-joint motor saturation torques [N*m].
    effort_limits: Dict[str, float] = field(default_factory=lambda: dict(_EFFORT))

    # Torque-speed curve (physical mode only; ``None`` ⇒ plain PD hard clip).
    # When set, the actuator delivers full effort up to ``knee_point_velocity``,
    # then ramps linearly to zero at ``velocity_limit`` (booster_train T-N curve).
    velocity_limit: Dict[str, float] | None = field(
        default_factory=lambda: dict(_VELOCITY_LIMIT) if _VELOCITY_LIMIT is not None else None
    )
    knee_point_velocity: Dict[str, float] | None = field(
        default_factory=lambda: dict(_KNEE_POINT) if _KNEE_POINT is not None else None
    )

    # Physical action scale (0.25·effort/kp). Physical mode only; ``None`` ⇒ the
    # recipe's own action_scale is used. Builders prefer this when set.
    physical_action_scale: Dict[str, float] | None = field(
        default_factory=lambda: dict(_ACTION_SCALE) if _ACTION_SCALE is not None else None
    )

    # Actuator saturation scale kappa for the tanh torque model
    # ``tau_motor = kappa * tanh(tau_PD / kappa)``. DISABLED for K1 (``None`` ⇒
    # plain PD + hard effort clip). The deploy/firmware only ever does plain PD,
    # so the sim-only tanh widened the sim2real gap rather than closing it. The
    # model itself is kept (``actuator_pd`` tanh path, ``randomize_tau_scale``,
    # the saturation/DR diags); to re-enable, set this to a per-group dict like
    # ``effort_limits`` and add a ``dr_tau_scale`` event term back.
    tau_scale: Dict[str, float] | None = None

    # Foot bodies (ankle-roll links carrying the foot geoms/sites).
    foot_names: List[str] = field(default_factory=lambda: ["left_foot_link", "right_foot_link"])

    trunk_body_name: str = "Trunk"

    # Joint subsets referenced by the joint-deviation reward terms
    # (hip = roll/yaw only, matching the source task).
    hip_joint_patterns: tuple[str, ...] = (r".*_Hip_Roll", r".*_Hip_Yaw")
    knee_joint_patterns: tuple[str, ...] = (r".*_Knee_Pitch",)
    ankle_joint_patterns: tuple[str, ...] = _ANKLE_PATTERNS

    @property
    def foot_geom_names(self) -> tuple[str, ...]:
        """Ground-contact sphere geom names, left then right.

        The box geoms (``foot_box_geom_names``) only collide with each
        other and serve the foot-to-foot collision penalty; ground
        contact goes exclusively through these spheres.
        """
        return tuple(f"{side}_foot_{i}" for side in ("left", "right") for i in range(1, 5))

    @property
    def foot_box_geom_names(self) -> tuple[str, ...]:
        """Foot bounding-box geom names (foot-to-foot collision pair)."""
        return ("left_foot", "right_foot")
