"""Booster K1 (22 actuated DOF) with the full-body collision model.

The plant of the ``k1_velocity`` preset, taken from the booster_mjlab recipe:

- **Collision model.** The MJCF at ``assets/K1/k1.xml`` carries 22 collision
  geoms covering the whole body (trunk, waist, head, arms, hands, hips,
  shanks, feet), so self-collision and non-foot ground contact produce a real
  signal, which is what the posture/collision reward terms of the reference
  recipe are written against.
- **Actuator tuning.** PD gains are the reference recipe's flat per-group
  values (legs 80/4, ankle 50/2, arm 10/0.45, head 4/0.25). Effort ratings,
  armature and the piecewise-linear torque-speed curve are the Booster motor
  spec; the action scale follows from the gains (``0.25 * effort / kp``). The
  arm damping is the one gain that does NOT match the source's 1.0 -- see
  :data:`ARM_DAMPING_NOTE`.

Joint-name convention is the one the MJCF ships (``Head_Yaw``/``Head_Pitch``,
``{Left,Right}_{Shoulder,Elbow}_*``, ``{Left,Right}_{Hip_*,Knee_Pitch,
Ankle_*}``), which is NOT the Booster deploy SDK's convention (``AAHead_yaw``,
``ALeft_Shoulder_Pitch``); the deploy layer remaps joints by normalized name.

The MJCF joints carry no damping and no frictionloss, matching the source
asset. The PD law is therefore the only thing resisting joint motion, which is
what makes the arm damping value load-bearing here in a way it is not upstream;
the preset randomizes ``dof_damping`` on top to cover the real joint's own.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from jaxrlworld.rl.configs.robots.base import RobotConfig

_XML = "./JaxRLWorld/jaxrlworld/assets/K1/k1.xml"

# ── Joint-group regexes ───────────────────────────────────────────────
# Fullmatch'd against fully-qualified joint labels on each backend; the
# leading ``.*`` absorbs Newton's hierarchical MJCF XPath prefixes and
# harmlessly matches the empty string elsewhere.
_HEAD_PATTERNS = (r".*Head_Yaw", r".*Head_Pitch")
_ARM_PATTERNS = (
    r".*_Shoulder_Pitch",
    r".*_Shoulder_Roll",
    r".*_Elbow_Pitch",
    r".*_Elbow_Yaw",
)
_HIP_PITCH_PATTERNS = (r".*_Hip_Pitch",)
_HIP_ROLL_PATTERNS = (r".*_Hip_Roll",)
_HIP_YAW_PATTERNS = (r".*_Hip_Yaw",)
_KNEE_PATTERNS = (r".*_Knee_Pitch",)
_ANKLE_PATTERNS = (r".*_Ankle_Pitch", r".*_Ankle_Roll")

_GROUPS: Dict[str, tuple] = {
    "head": _HEAD_PATTERNS,
    "arm": _ARM_PATTERNS,
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
        for pattern in patterns:
            out[pattern] = value_by_group[group]
    return out


# ── Motor bundle ──────────────────────────────────────────────────────
# Per group: (armature [kg*m^2], effort [N*m], velocity_limit [rad/s],
# knee_point_velocity [rad/s], kp [N*m/rad], kd [N*m*s/rad]).
#
# Motor ratings and armature are the Booster spec (identical to the public
# K1 config). kp/kd are the reference recipe's flat per-group values.
#
# The ankle is a parallel linkage of two E4310s modelled as one serial joint,
# so its armature is the single-motor value doubled while its effort rating
# stays that of one motor.
#
# The head motor's quoted knee-point speed (10.47) exceeds its no-load speed,
# so the curve is clamped to the no-load speed and the group runs at constant
# torque up to saturation.
_MOTOR: Dict[str, tuple] = {
    "head": (0.001, 6.0, 7.85, 7.85, 4.0, 0.25),  # HT4438 (knee point clamped)
    # R14. The source rates this group at kd = 1.0; this port runs 0.45, which
    # is the one motor value it does not reproduce. See ARM_DAMPING_NOTE below.
    "arm": (0.001, 14.0, 33.51, 5.24, 10.0, 0.45),
    "hip_pitch": (0.0478125, 68.0, 14.66, 1.88, 80.0, 4.0),  # E6408
    "hip_roll": (0.0339552, 76.0, 12.57, 2.62, 80.0, 4.0),  # E4315
    "hip_yaw": (0.0282528, 38.3, 17.59, 7.85, 80.0, 4.0),  # E4310
    "knee": (0.095625, 112.0, 12.57, 2.09, 80.0, 4.0),  # E6416
    "ankle": (0.0565056, 38.3, 17.59, 7.85, 50.0, 2.0),  # E4310 pair
}

ARM_DAMPING_NOTE = """Why the arm PD damping departs from the source.

The source drives its joints with MuJoCo's built-in position actuators, where
the damping term is folded into the solve and is therefore stable at any step
size. This port computes the PD torque itself and applies it as a force, which
is only stable while ``dt < 2 * J / kd``.

At the home pose the elbow pitch carries an effective inertia of 0.0024 kg m^2.
With the source's kd of 1.0 and a 5 ms actuator step that ratio is 2.08 -- just
past the limit, where the damping term stops removing velocity and starts
reversing and amplifying it once per step. The arms visibly tremble at a
standstill; nothing else does, because every other joint has ten times the
margin or more.

0.45 puts the elbow at a 2.1x margin. The cost is that the shoulders, which
were never the problem, drop from a damping ratio of 0.80 to 0.36. The source
asset declares no passive joint damping to make that up, so the preset
randomizes ``dof_damping`` over [0, 1] instead -- which also covers the real
joint's own unknown damping.
"""

# Joint-position action scale: a quarter of the torque headroom the PD law
# has at full deflection, per group.
_ACTION_SCALE_FACTOR = 0.25

_ARMATURE = _pattern_dict({g: v[0] for g, v in _MOTOR.items()})
_EFFORT = _pattern_dict({g: v[1] for g, v in _MOTOR.items()})
_VELOCITY_LIMIT = _pattern_dict({g: v[2] for g, v in _MOTOR.items()})
_KNEE_POINT = _pattern_dict({g: v[3] for g, v in _MOTOR.items()})
_P_GAINS = _pattern_dict({g: v[4] for g, v in _MOTOR.items()})
_D_GAINS = _pattern_dict({g: v[5] for g, v in _MOTOR.items()})
_ACTION_SCALE = _pattern_dict({g: _ACTION_SCALE_FACTOR * v[1] / v[4] for g, v in _MOTOR.items()})


@dataclass
class K1Config(RobotConfig):
    """Booster K1, full-body collision model."""

    name: str = "K1"
    urdf_path: str | None = None
    mjcf_path: str | None = _XML
    usd_path: str | None = None

    # Home keyframe: a deeper crouch than the public K1's, from the source
    # recipe. Base height is the keyframe's, not a standing-tall figure.
    base_init_height: float = 0.5125
    base_link_name: str = "Trunk"

    # Regex -> angle; joints that match nothing default to 0.
    default_joint_angles: Dict[str, float] = field(
        default_factory=lambda: {
            r".*Left_Shoulder_Roll": -1.4,
            r".*Left_Elbow_Yaw": -0.4,
            r".*Right_Shoulder_Roll": 1.4,
            r".*Right_Elbow_Yaw": 0.4,
            r".*_Hip_Pitch": -0.4,
            r".*_Knee_Pitch": 0.8,
            r".*_Ankle_Pitch": -0.4,
        }
    )

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: [
            *_HEAD_PATTERNS,
            *_ARM_PATTERNS,
            *_HIP_PITCH_PATTERNS,
            *_HIP_ROLL_PATTERNS,
            *_HIP_YAW_PATTERNS,
            *_KNEE_PATTERNS,
            *_ANKLE_PATTERNS,
        ]
    )

    p_gains: Dict[str, float] = field(default_factory=lambda: dict(_P_GAINS))
    d_gains: Dict[str, float] = field(default_factory=lambda: dict(_D_GAINS))
    armature: Dict[str, float] = field(default_factory=lambda: dict(_ARMATURE))

    effort_limits: Dict[str, float] = field(default_factory=lambda: dict(_EFFORT))

    # Piecewise-linear torque-speed curve: full effort up to
    # ``knee_point_velocity``, then a linear ramp to zero at ``velocity_limit``.
    velocity_limit: Dict[str, float] = field(default_factory=lambda: dict(_VELOCITY_LIMIT))
    knee_point_velocity: Dict[str, float] = field(default_factory=lambda: dict(_KNEE_POINT))

    # Per-joint action scale; the sim builders use this over any scale the
    # recipe declares.
    physical_action_scale: Dict[str, float] = field(default_factory=lambda: dict(_ACTION_SCALE))

    # Plain PD plus a hard effort clip, as the firmware does. The tanh
    # saturation model stays disabled here for the same reason it is on the
    # public K1: it widened the sim2real gap rather than closing it.
    tau_scale: Dict[str, float] | None = None

    # Torque-delivery deficits, present but neutral (a zero time constant and
    # unit gain are exact passthrough) so the per-env buffers exist for
    # identification to write without changing behaviour.
    tau_lpf_time_constant: float = 0.0
    dyn_gain: float = 1.0
    dyn_gain_velocity: float = 0.5

    # The MJCF declares neither joint damping nor frictionloss; the PD law is
    # the only source of joint damping, as upstream.
    joint_frictionloss: float = 0.0

    soft_joint_pos_limit_factor: float = 0.9

    # Bodies and geoms the MDP terms select.
    foot_names: List[str] = field(default_factory=lambda: ["left_foot_link", "right_foot_link"])
    trunk_body_name: str = "Trunk"

    # Joint subsets the reward terms scope to.
    hip_joint_patterns: tuple[str, ...] = _HIP_ROLL_PATTERNS + _HIP_YAW_PATTERNS
    knee_joint_patterns: tuple[str, ...] = _KNEE_PATTERNS
    ankle_joint_patterns: tuple[str, ...] = _ANKLE_PATTERNS
    upper_body_joint_patterns: tuple[str, ...] = _HEAD_PATTERNS + _ARM_PATTERNS

    @property
    def foot_geom_names(self) -> tuple[str, ...]:
        """Ground-contact geoms, left then right.

        Each foot collides through one convex hull of its own mesh, not the
        sphere cluster the public K1 asset uses.
        """
        return ("left_foot_collision", "right_foot_collision")

    @property
    def foot_site_names(self) -> tuple[str, ...]:
        """Foot reference sites (foot-link origin), left then right."""
        return ("left_foot", "right_foot")

    @property
    def foot_sole_site_names(self) -> tuple[str, ...]:
        """Sole-plane reference sites, biased forward over the contact patch.

        These sit 3 cm below the foot-link origin, so a clearance target
        measured from them is ~3 cm smaller than the same target measured
        from :attr:`foot_site_names`.
        """
        return ("left_foot_sole", "right_foot_sole")

    @property
    def non_foot_body_pattern(self) -> str:
        """Every body except the two feet.

        Name resolution is a fullmatch, so the lookahead has to exclude the
        exact names rather than a substring.
        """
        return r"(?!left_foot_link$|right_foot_link$).*"
