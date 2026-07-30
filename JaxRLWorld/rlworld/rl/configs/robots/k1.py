"""Booster K1 humanoid robot configuration (22 actuated DOF).

All values except the PD gains and armature are read verbatim from the
MJCF at ``assets/K1/k1_mjx_feetonly.xml`` (default-class attributes and
the home keyframe of its companion scene file) so the model matches the
source:

- PD gains: chosen by ``PD_PROFILE`` (default ``"stiff_legs"`` = arms 15,
  hip/knee 50, ankle 15, ``kd`` arm 2, leg 5). Also available: the softer
  ``"mujoco_playground"`` gains (kp 15/25/10, kd 2/3) and the real-robot
  ``"booster"`` gains (kp 4/80/30, kd 1/2); switch with the ``K1_PD_PROFILE``
  env var.
- frictionloss 0.1 uniformly on every joint.
- armature per joint from the Booster reference model
  ``assets/K1/K1_22dof.xml`` (head 0.002, arm 0.001, legs
  0.028-0.096).  The feetonly MJCF flattens all of these to a uniform
  0.005, which understates the leg reflected rotor inertia by roughly
  an order of magnitude; the actuator configs override the XML value
  on every backend, so the reference values win at runtime.
- effort limits from ``actuatorfrcrange`` per class.
- action scale is 1.0 (raw action added to the default pose), so no
  per-group action-scale table exists for this robot.

Joint-name convention is the Booster family one (same as T1):
``AAHead_yaw``/``Head_pitch``, ``A{Left,Right}_Shoulder_Pitch``,
``{Left,Right}_{Shoulder_Roll,Elbow_Pitch,Elbow_Yaw}``, and
``{Left,Right}_{Hip_{Pitch,Roll,Yaw},Knee_Pitch,Ankle_{Pitch,Roll}}``.
K1 has no waist joint.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

from rlworld.rl.configs.robots.base import RobotConfig

# ── PD gain profiles (kp = stiffness, kd = damping) ───────────────────
# Two coherent gain sets, kept side by side and selected by ``PD_PROFILE``:
#   "booster"           – Booster SDK deploy gains (booster_deploy's K1_CFG),
#                         matched to the real K1 actuators. The default.
#   "mujoco_playground" – the k1_mjx_feetonly.xml sim gains this preset
#                         originally shipped with. NOT the real-robot gains;
#                         kept to reproduce older runs.
_PD_PROFILES: Dict[str, Dict[str, float]] = {
    "booster": {
        "stiffness_head": 4.0,
        "stiffness_arm": 4.0,
        "stiffness_hip": 80.0,
        "stiffness_knee": 80.0,
        "stiffness_ankle": 30.0,
        "damping_head": 1.0,
        "damping_arm": 1.0,
        "damping_leg": 2.0,
    },
    "mujoco_playground": {
        "stiffness_head": 15.0,
        "stiffness_arm": 15.0,
        "stiffness_hip": 25.0,
        "stiffness_knee": 25.0,
        "stiffness_ankle": 10.0,
        "damping_head": 2.0,
        "damping_arm": 2.0,
        "damping_leg": 3.0,
    },
    # Middle ground: mujoco_playground arms, stiffer legs so they hold the
    # body under load (kp 25 sagged / looked underpowered on the real robot).
    # kd scaled up with kp to keep it well-damped (avoids the kp-80 shaking).
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

# Active profile. Defaults to "stiff_legs" (stiffer, well-damped legs that hold
# the body under load better than the softer mujoco_playground gains). Override
# per-run with the ``K1_PD_PROFILE`` env var, e.g.
# ``K1_PD_PROFILE=mujoco_playground`` or ``K1_PD_PROFILE=booster``. The resolved
# kp/kd are baked into the saved config.yaml, so each checkpoint records the
# gains it trained with.
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

    p_gains: Dict[str, float] = field(
        default_factory=lambda: _pattern_dict(
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
    )

    d_gains: Dict[str, float] = field(
        default_factory=lambda: _pattern_dict(
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
    )

    # frictionloss (0.1) needs no config field: it rides in via the MJCF
    # joint default class on every backend.
    # Explicit per-group patterns, NOT a ".*" catch-all: Genesis resolves
    # this dict against ALL joint names including the free joint, so a
    # catch-all produces one value too many vs the actuated-DOF index set.
    armature: Dict[str, float] = field(
        default_factory=lambda: _pattern_dict(
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
    )

    # Per-joint motor saturation torques [N*m] from ``actuatorfrcrange``.
    effort_limits: Dict[str, float] = field(
        default_factory=lambda: _pattern_dict(
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
    )

    # Actuator saturation scale kappa for the tanh torque model
    # ``tau_motor = kappa * tanh(tau_PD / kappa)`` (smooth torque decay in the
    # high-torque regime instead of a hard clip). Initialized to the effort
    # limits, so v1 is a soft version of the existing clip; a SysID target —
    # lower kappa where the real motors show more decay. Same per-group format
    # as ``effort_limits``.
    tau_scale: Dict[str, float] = field(
        default_factory=lambda: _pattern_dict(
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
    )

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
