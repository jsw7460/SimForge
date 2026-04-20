"""Booster T1 humanoid robot configuration.

Numbers mirror ``Mjlab/src/mjlab/asset_zoo/robots/booster_t1/t1_constants.py``
(same natural frequency / damping ratio / reflected-inertia formula) so that
Newton/Genesis/MuJoCo share identical PD gains, armature, effort, and action
scale. Reference for motor specs:
https://booster.feishu.cn/wiki/JGZAwk8CUi5m6nklgxMcp2KlnVe

Joint-name convention matches both the URDF at ``assets/T1/T1_locomotion.urdf``
and the MJCF at ``Mjlab/.../booster_t1/xmls/t1.xml`` (verified identical).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

from .base import RobotConfig
from rlworld.rl.configs.robots.utils import reflected_inertia_simple


# ── Motor specs (from Booster T1 datasheet) ──────────────────────────
# reflected_inertia_simple(rotor_inertia, gear_ratio) = rotor * gear^2
ARMATURE_NECK                = reflected_inertia_simple(18e-6,   10)
ARMATURE_ARM                 = reflected_inertia_simple(21.8e-6, 36)
ARMATURE_WAIST_HIP_ROLL_YAW  = reflected_inertia_simple(76.5e-6, 25)
ARMATURE_HIP_PITCH           = reflected_inertia_simple(161.7e-6, 18)
ARMATURE_KNEE                = reflected_inertia_simple(196.3e-6, 18)
ARMATURE_ANKLE               = reflected_inertia_simple(26.2e-6, 36)

EFFORT_NECK                = 7.0
EFFORT_ARM                 = 36.0
EFFORT_WAIST_HIP_ROLL_YAW  = 40.0
EFFORT_HIP_PITCH           = 55.0
EFFORT_KNEE                = 65.0
EFFORT_ANKLE               = 50.0

# PD gains: critically-damped second-order with natural frequency 5 Hz
# and damping ratio 2.0 (same as mjlab's t1_constants.py).
NATURAL_FREQ = 5.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0


def _kp(armature: float) -> float:
    return armature * NATURAL_FREQ ** 2


def _kv(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


STIFFNESS_NECK                = _kp(ARMATURE_NECK)
STIFFNESS_ARM                 = _kp(ARMATURE_ARM)
STIFFNESS_WAIST_HIP_ROLL_YAW  = _kp(ARMATURE_WAIST_HIP_ROLL_YAW)
STIFFNESS_HIP_PITCH           = _kp(ARMATURE_HIP_PITCH)
STIFFNESS_KNEE                = _kp(ARMATURE_KNEE)
STIFFNESS_ANKLE               = _kp(ARMATURE_ANKLE)

DAMPING_NECK                = _kv(ARMATURE_NECK)
DAMPING_ARM                 = _kv(ARMATURE_ARM)
DAMPING_WAIST_HIP_ROLL_YAW  = _kv(ARMATURE_WAIST_HIP_ROLL_YAW)
DAMPING_HIP_PITCH           = _kv(ARMATURE_HIP_PITCH)
DAMPING_KNEE                = _kv(ARMATURE_KNEE)
DAMPING_ANKLE               = _kv(ARMATURE_ANKLE)

# Action scale per joint group: 0.25 * effort / stiffness (mjlab convention).
ACTION_SCALE_NECK                = 0.25 * EFFORT_NECK                / STIFFNESS_NECK
ACTION_SCALE_ARM                 = 0.25 * EFFORT_ARM                 / STIFFNESS_ARM
ACTION_SCALE_WAIST_HIP_ROLL_YAW  = 0.25 * EFFORT_WAIST_HIP_ROLL_YAW  / STIFFNESS_WAIST_HIP_ROLL_YAW
ACTION_SCALE_HIP_PITCH           = 0.25 * EFFORT_HIP_PITCH           / STIFFNESS_HIP_PITCH
ACTION_SCALE_KNEE                = 0.25 * EFFORT_KNEE                / STIFFNESS_KNEE
ACTION_SCALE_ANKLE               = 0.25 * EFFORT_ANKLE               / STIFFNESS_ANKLE


# ── Joint-group regexes (match both URDF and MJCF joint names) ───────
# Neck: AAHead_yaw, Head_pitch
# Arm:  {Left,Right}_{Shoulder,Elbow}_{Pitch,Roll,Yaw}
# Waist + Hip Roll/Yaw: Waist, {Left,Right}_Hip_{Roll,Yaw}
# Hip Pitch:            {Left,Right}_Hip_Pitch
# Knee:                 {Left,Right}_Knee_Pitch
# Ankle:                {Left,Right}_Ankle_{Pitch,Roll}

#
# Patterns are fullmatch'd (via ``re.fullmatch``) against fully-
# qualified joint/body labels on each backend. Newton's URDF parser
# produces flat labels (``T1/Waist``); Newton's MJCF parser produces
# hierarchical XPath labels (``T1/worldbody/Trunk/Waist/Waist``).
# Every pattern therefore starts with ``.*`` so the ``.*`` can absorb
# the intermediate XPath segments on MJCF and harmlessly match the
# empty string on URDF.
_NECK_PATTERNS               = (r".*AAHead_yaw", r".*Head_pitch")
_ARM_PATTERNS                = (
    r".*_Shoulder_Pitch", r".*_Shoulder_Roll",
    r".*_Elbow_Pitch",    r".*_Elbow_Yaw",
)
_WAIST_HIP_ROLL_YAW_PATTERNS = (r".*Waist", r".*_Hip_Roll", r".*_Hip_Yaw")
_HIP_PITCH_PATTERNS          = (r".*_Hip_Pitch",)
_KNEE_PATTERNS               = (r".*_Knee_Pitch",)
_ANKLE_PATTERNS              = (r".*_Ankle_Pitch", r".*_Ankle_Roll")


def _pattern_dict(value_by_group: Dict[str, float]) -> Dict[str, float]:
    """Flatten a per-group scalar into a per-regex dict."""
    groups = {
        "neck":               _NECK_PATTERNS,
        "arm":                _ARM_PATTERNS,
        "waist_hip_roll_yaw": _WAIST_HIP_ROLL_YAW_PATTERNS,
        "hip_pitch":          _HIP_PITCH_PATTERNS,
        "knee":                _KNEE_PATTERNS,
        "ankle":               _ANKLE_PATTERNS,
    }
    out: Dict[str, float] = {}
    for group, patterns in groups.items():
        for p in patterns:
            out[p] = value_by_group[group]
    return out


T1_ACTION_SCALE: Dict[str, float] = _pattern_dict({
    "neck":               ACTION_SCALE_NECK,
    "arm":                ACTION_SCALE_ARM,
    "waist_hip_roll_yaw": ACTION_SCALE_WAIST_HIP_ROLL_YAW,
    "hip_pitch":          ACTION_SCALE_HIP_PITCH,
    "knee":                ACTION_SCALE_KNEE,
    "ankle":               ACTION_SCALE_ANKLE,
})


@dataclass
class T1Config(RobotConfig):
    """Configuration for Booster T1 humanoid robot (24 actuated DOF)."""

    name: str = "T1"
    # T1_serial.urdf has all 23 revolute joints (head + arms + waist + legs),
    # matching the mjlab T1 XML. T1_locomotion.urdf has the upper-body
    # joints fixed and only exposes the 12 leg DOFs — unusable for the
    # full-body getup task.
    urdf_path: str | None = "./JaxRLWorld/rlworld/assets/T1/T1_23dof.urdf"

    # Newton uses T1_serial.xml (deprecated) because Newton's MJCF
    # parser doesn't fully support menagerie-style features
    # (childclass, nested defaults, position actuator defaults).
    # Genesis uses menagerie t1.xml (no ground, clean class defaults).
    # MuJoCo uses mjlab asset_zoo (spec_fn).
    mjcf_path: str | None = (
        # "./JaxRLWorld/rlworld/assets/T1_deprecated/T1_serial.xml"
        "./JaxRLWorld/rlworld/assets/menagerie_T1/t1.xml"
    )

    # Menagerie T1 MJCF for Genesis (no inline ground plane, clean
    # class defaults). Newton can't use this due to parser limitations.
    mjcf_path_genesis: str | None = (
        "./JaxRLWorld/rlworld/assets/menagerie_T1/t1.xml"
    )

    # From HOME_KEYFRAME in mjlab t1_constants.py.
    base_init_height: float = 0.665
    base_link_name: str = "Trunk"

    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        r".*Left_Shoulder_Roll":  -1.4,
        r".*Left_Elbow_Yaw":      -0.4,
        r".*Right_Shoulder_Roll":  1.4,
        r".*Right_Elbow_Yaw":      0.4,
        r".*_Hip_Pitch":           -0.2,
        r".*_Knee_Pitch":           0.4,
        r".*_Ankle_Pitch":         -0.2,
    })

    actuated_dof_patterns: List[str] = field(default_factory=lambda: [
        *_NECK_PATTERNS,
        *_ARM_PATTERNS,
        *_WAIST_HIP_ROLL_YAW_PATTERNS,
        *_HIP_PITCH_PATTERNS,
        *_KNEE_PATTERNS,
        *_ANKLE_PATTERNS,
    ])

    p_gains: Dict[str, float] = field(default_factory=lambda: _pattern_dict({
        "neck":               STIFFNESS_NECK,
        "arm":                STIFFNESS_ARM,
        "waist_hip_roll_yaw": STIFFNESS_WAIST_HIP_ROLL_YAW,
        "hip_pitch":          STIFFNESS_HIP_PITCH,
        "knee":                STIFFNESS_KNEE,
        "ankle":               STIFFNESS_ANKLE,
    }))

    d_gains: Dict[str, float] = field(default_factory=lambda: _pattern_dict({
        "neck":               DAMPING_NECK,
        "arm":                DAMPING_ARM,
        "waist_hip_roll_yaw": DAMPING_WAIST_HIP_ROLL_YAW,
        "hip_pitch":          DAMPING_HIP_PITCH,
        "knee":                DAMPING_KNEE,
        "ankle":               DAMPING_ANKLE,
    }))

    armature: Dict[str, float] = field(default_factory=lambda: _pattern_dict({
        "neck":               ARMATURE_NECK,
        "arm":                ARMATURE_ARM,
        "waist_hip_roll_yaw": ARMATURE_WAIST_HIP_ROLL_YAW,
        "hip_pitch":          ARMATURE_HIP_PITCH,
        "knee":                ARMATURE_KNEE,
        "ankle":               ARMATURE_ANKLE,
    }))

    # Per-joint motor saturation torques [N*m], matching mjlab's
    # booster_t1 ElectricActuator spec. Newton uses this to override the
    # tighter ``actuatorfrcrange`` values in the menagerie MJCF (e.g.
    # ankle 15/20 Nm → 50 Nm) that otherwise prevent getup from training.
    effort_limits: Dict[str, float] = field(default_factory=lambda: _pattern_dict({
        "neck":               EFFORT_NECK,
        "arm":                EFFORT_ARM,
        "waist_hip_roll_yaw": EFFORT_WAIST_HIP_ROLL_YAW,
        "hip_pitch":          EFFORT_HIP_PITCH,
        "knee":                EFFORT_KNEE,
        "ankle":               EFFORT_ANKLE,
    }))

    foot_names: List[str] = field(
        default_factory=lambda: ["left_foot_link", "right_foot_link"]
    )

    # Body names used by getup rewards and self-collision subtree.
    trunk_body_name: str = "Trunk"
    waist_body_name: str = "Waist"

    # Regex patterns used by the 3-axis geom-friction DR.
    #
    # MuJoCo path (mjlab asset_zoo ``booster_t1`` XML) keeps explicit
    # geom names like ``left_foot1_collision``, so mjlab's
    # ``dr.geom_friction`` filters by geom name directly — see
    # ``foot_geom_names_mjlab`` below for the exact names the builder
    # passes to mjlab's asset_cfg.
    #
    # Newton path loads the same robot from URDF, and Newton's URDF
    # loader drops collision-geom names (every shape becomes
    # ``shape_N``). So the Newton builder filters by *body name*
    # instead, using ``model.body_shapes`` to resolve the attached
    # shape indices — this is the same path SysID's
    # ``apply_contact_friction`` uses.
    # Newton-side foot-body regex. The ``.*`` prefix lets a single
    # pattern fullmatch both URDF labels (``T1/left_foot_link``) and
    # MJCF XPath labels (``T1/worldbody/Trunk/Waist/.../left_foot_link``)
    # so the same preset works with either loader.
    foot_body_pattern_newton: str = r"T1/.*(left|right)_foot_link"

    @property
    def foot_geom_names_mjlab(self) -> tuple[str, ...]:
        """mjlab asset_zoo collision geom names for the feet.

        Matches the ``_foot_regex`` in ``Mjlab/.../booster_t1/
        t1_constants.py`` (``^(left|right)_foot\\d+_collision$``),
        expanded into the explicit name tuple that mjlab's
        ``SceneEntityCfg.geom_names`` filter expects.
        """
        return tuple(
            f"{side}_foot{i}_collision"
            for side in ("left", "right")
            for i in range(1, 5)
        )

    @property
    def prefixed_foot_names(self) -> tuple[str, ...]:
        return self.prefixed_list(self.foot_names)

    @property
    def prefixed_action_scale(self) -> Dict[str, float]:
        return {f"{self.name}/{k}": v for k, v in T1_ACTION_SCALE.items()}
