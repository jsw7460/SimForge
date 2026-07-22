"""Booster K1 humanoid robot configuration (22 actuated DOF).

All values are read verbatim from the MJCF at
``assets/K1/k1_mjx_feetonly.xml`` (default-class attributes and
the home keyframe of its companion scene file) so the joint-PD path
reproduces the source model exactly:

- PD gains: actuator ``kp`` per class (head/arm 15, hip/knee 25,
  ankle 10) with the joint ``damping`` as ``kd`` (head/arm 2, leg 3).
- armature 0.005 and frictionloss 0.1 uniformly on every joint.
- effort limits from ``actuatorfrcrange`` per class.
- action scale is 1.0 (raw action added to the default pose), so no
  per-group action-scale table exists for this robot.

Joint-name convention is the Booster family one (same as T1):
``AAHead_yaw``/``Head_pitch``, ``A{Left,Right}_Shoulder_Pitch``,
``{Left,Right}_{Shoulder_Roll,Elbow_Pitch,Elbow_Yaw}``, and
``{Left,Right}_{Hip_{Pitch,Roll,Yaw},Knee_Pitch,Ankle_{Pitch,Roll}}``.
K1 has no waist joint.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from rlworld.rl.configs.robots.base import RobotConfig

# ── Values from k1_mjx_feetonly.xml default classes ───────────────────
STIFFNESS_HEAD = 15.0
STIFFNESS_ARM = 15.0
STIFFNESS_HIP = 25.0
STIFFNESS_KNEE = 25.0
STIFFNESS_ANKLE = 10.0

# ``kd`` comes from the joint ``damping`` attribute in the MJCF.
DAMPING_HEAD = 2.0
DAMPING_ARM = 2.0
DAMPING_LEG = 3.0

ARMATURE_ALL = 0.005

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
    armature: Dict[str, float] = field(default_factory=lambda: _pattern_dict(dict.fromkeys(_GROUPS, ARMATURE_ALL)))

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
