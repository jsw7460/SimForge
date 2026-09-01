from dataclasses import dataclass, field
from typing import Dict, List

from jaxrlworld.rl.configs.robots.utils import reflected_inertia_from_two_stage_planetary

from .base import RobotConfig

# Motor 5020 (elbows, shoulders, wrist_roll)
ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)  # (1.39e-5, 1.7e-6, 1.69e-5) kg·m^2 — per-stage rotor inertias
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))  # (1, 3.5556, 4.5) — two-stage planetary; total gear ratio ≈ 16
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_5020, GEARS_5020)  # ≈ 3.610e-3 kg·m^2

# Motor 7520_14 (hip_pitch, hip_yaw, waist_yaw)
ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)  # (4.89e-5, 9.8e-6, 5.33e-5) kg·m^2
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))  # (1, 4.5, 3.1818) — total gear ratio ≈ 14.32
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_7520_14, GEARS_7520_14
)  # ≈ 1.018e-2 kg·m^2

# Motor 7520_22 (hip_roll, knee)
ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)  # (4.89e-5, 1.09e-5, 7.38e-5) kg·m^2
GEARS_7520_22 = (1, 4.5, 5)  # total gear ratio = 22.5
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
    ROTOR_INERTIAS_7520_22, GEARS_7520_22
)  # ≈ 2.510e-2 kg·m^2

# Motor 4010 (wrist_pitch, wrist_yaw)
ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)  # (6.8e-6, 0, 0) kg·m^2 — second/third stages absent
GEARS_4010 = (1, 5, 5)  # total gear ratio = 25
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_4010, GEARS_4010)  # ≈ 4.250e-3 kg·m^2

# Parallel linkage actuators (2x 5020)
ARMATURE_WAIST = ARMATURE_5020 * 2  # ≈ 7.220e-3 kg·m^2
ARMATURE_ANKLE = ARMATURE_5020 * 2  # ≈ 7.220e-3 kg·m^2

# PD gains
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz → 62.832 rad/s
DAMPING_RATIO = 2.0  # overdamped (≥1 critically damped)

# Stiffness ke = armature * ω_n^2  (ω_n^2 ≈ 3947.84)  [units: N·m/rad]
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2  # ≈ 14.25
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2  # ≈ 40.18
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2  # ≈ 99.10
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2  # ≈ 16.78
STIFFNESS_WAIST = STIFFNESS_5020 * 2  # ≈ 28.50
STIFFNESS_ANKLE = STIFFNESS_5020 * 2  # ≈ 28.50

# Damping kd = 2·ζ·armature·ω_n  (= 251.33·armature)  [units: N·m·s/rad]
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ  # ≈ 0.907
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # ≈ 2.558
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ  # ≈ 6.309
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ  # ≈ 1.068
DAMPING_WAIST = DAMPING_5020 * 2  # ≈ 1.814
DAMPING_ANKLE = DAMPING_5020 * 2  # ≈ 1.814

# Motor effort limits  [N·m]
EFFORT_5020 = 25.0
EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_4010 = 5.0
EFFORT_WAIST = EFFORT_5020 * 2  # = 50.0
EFFORT_ANKLE = EFFORT_5020 * 2  # = 50.0

# Action scale: 0.25 * effort / stiffness  [rad — target deviation that saturates motor at quarter-effort]
ACTION_SCALE_5020 = 0.25 * EFFORT_5020 / STIFFNESS_5020  # ≈ 0.4386 rad (≈ 25.1°)
ACTION_SCALE_7520_14 = 0.25 * EFFORT_7520_14 / STIFFNESS_7520_14  # ≈ 0.5475 rad (≈ 31.4°)
ACTION_SCALE_7520_22 = 0.25 * EFFORT_7520_22 / STIFFNESS_7520_22  # ≈ 0.3507 rad (≈ 20.1°)
ACTION_SCALE_4010 = 0.25 * EFFORT_4010 / STIFFNESS_4010  # ≈ 0.0745 rad (≈  4.3°)
ACTION_SCALE_WAIST = 0.25 * EFFORT_WAIST / STIFFNESS_WAIST  # ≈ 0.4386 rad (= ACTION_SCALE_5020 — 2× / 2× cancel)
ACTION_SCALE_ANKLE = 0.25 * EFFORT_ANKLE / STIFFNESS_ANKLE  # ≈ 0.4386 rad (= ACTION_SCALE_5020)

G1_ACTION_SCALE: Dict[str, float] = {
    r".*_hip_pitch_joint": ACTION_SCALE_7520_14,
    r".*_hip_yaw_joint": ACTION_SCALE_7520_14,
    r"waist_yaw_joint": ACTION_SCALE_7520_14,
    r".*_hip_roll_joint": ACTION_SCALE_7520_22,
    r".*_knee_joint": ACTION_SCALE_7520_22,
    r"waist_pitch_joint": ACTION_SCALE_WAIST,
    r"waist_roll_joint": ACTION_SCALE_WAIST,
    r".*_ankle_pitch_joint": ACTION_SCALE_ANKLE,
    r".*_ankle_roll_joint": ACTION_SCALE_ANKLE,
    r".*_shoulder_pitch_joint": ACTION_SCALE_5020,
    r".*_shoulder_roll_joint": ACTION_SCALE_5020,
    r".*_shoulder_yaw_joint": ACTION_SCALE_5020,
    r".*_elbow_joint": ACTION_SCALE_5020,
    r".*_wrist_roll_joint": ACTION_SCALE_5020,
    r".*_wrist_pitch_joint": ACTION_SCALE_4010,
    r".*_wrist_yaw_joint": ACTION_SCALE_4010,
}


@dataclass
class G1MujocoConfig(RobotConfig):
    """G1 config with mjlab-derived actuator parameters."""

    name: str = "g1_29dof"
    urdf_path: str | None = "./JaxRLWorld/jaxrlworld/assets/g1_description/g1_29dof.urdf"
    mjcf_path: str | None = "./JaxRLWorld/jaxrlworld/assets/g1/g1.xml"

    base_init_height: float = 0.76
    base_link_name: str = "pelvis"

    default_joint_angles: Dict[str, float] = field(
        default_factory=lambda: {
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            ".*left_shoulder_roll_joint": 0.2,
            ".*left_shoulder_pitch_joint": 0.2,
            ".*right_shoulder_roll_joint": -0.2,
            ".*right_shoulder_pitch_joint": 0.2,
        }
    )

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: [
            # Exclude hand_palm_joint (legacy) and foot_frame_joint
            # (welded dummy bodies for the foot-pad kinematic frame in g1.xml).
            r"left_(?!hand_palm_joint|foot_frame_joint).*",
            r"right_(?!hand_palm_joint|foot_frame_joint).*",
            r"waist_(?!support_joint).*",
        ]
    )

    p_gains: Dict[str, float] = field(
        default_factory=lambda: {
            ".*_hip_pitch_joint": STIFFNESS_7520_14,
            ".*_hip_yaw_joint": STIFFNESS_7520_14,
            ".*waist_yaw_joint": STIFFNESS_7520_14,
            ".*_hip_roll_joint": STIFFNESS_7520_22,
            ".*_knee_joint": STIFFNESS_7520_22,
            ".*waist_pitch_joint": STIFFNESS_WAIST,
            ".*waist_roll_joint": STIFFNESS_WAIST,
            ".*_ankle_pitch_joint": STIFFNESS_ANKLE,
            ".*_ankle_roll_joint": STIFFNESS_ANKLE,
            ".*_shoulder_pitch_joint": STIFFNESS_5020,
            ".*_shoulder_roll_joint": STIFFNESS_5020,
            ".*_shoulder_yaw_joint": STIFFNESS_5020,
            ".*_elbow_joint": STIFFNESS_5020,
            ".*_wrist_roll_joint": STIFFNESS_5020,
            ".*_wrist_pitch_joint": STIFFNESS_4010,
            ".*_wrist_yaw_joint": STIFFNESS_4010,
        }
    )

    d_gains: Dict[str, float] = field(
        default_factory=lambda: {
            ".*_hip_pitch_joint": DAMPING_7520_14,
            ".*_hip_yaw_joint": DAMPING_7520_14,
            ".*waist_yaw_joint": DAMPING_7520_14,
            ".*_hip_roll_joint": DAMPING_7520_22,
            ".*_knee_joint": DAMPING_7520_22,
            ".*waist_pitch_joint": DAMPING_WAIST,
            ".*waist_roll_joint": DAMPING_WAIST,
            ".*_ankle_pitch_joint": DAMPING_ANKLE,
            ".*_ankle_roll_joint": DAMPING_ANKLE,
            ".*_shoulder_pitch_joint": DAMPING_5020,
            ".*_shoulder_roll_joint": DAMPING_5020,
            ".*_shoulder_yaw_joint": DAMPING_5020,
            ".*_elbow_joint": DAMPING_5020,
            ".*_wrist_roll_joint": DAMPING_5020,
            ".*_wrist_pitch_joint": DAMPING_4010,
            ".*_wrist_yaw_joint": DAMPING_4010,
        }
    )

    armature: Dict[str, float] = field(
        default_factory=lambda: {
            ".*_hip_pitch_joint": ARMATURE_7520_14,
            ".*_hip_yaw_joint": ARMATURE_7520_14,
            ".*waist_yaw_joint": ARMATURE_7520_14,
            ".*_hip_roll_joint": ARMATURE_7520_22,
            ".*_knee_joint": ARMATURE_7520_22,
            ".*waist_pitch_joint": ARMATURE_WAIST,
            ".*waist_roll_joint": ARMATURE_WAIST,
            ".*_ankle_pitch_joint": ARMATURE_ANKLE,
            ".*_ankle_roll_joint": ARMATURE_ANKLE,
            ".*_shoulder_pitch_joint": ARMATURE_5020,
            ".*_shoulder_roll_joint": ARMATURE_5020,
            ".*_shoulder_yaw_joint": ARMATURE_5020,
            ".*_elbow_joint": ARMATURE_5020,
            ".*_wrist_roll_joint": ARMATURE_5020,
            ".*_wrist_pitch_joint": ARMATURE_4010,
            ".*_wrist_yaw_joint": ARMATURE_4010,
        }
    )

    foot_names: List[str] = field(default_factory=lambda: ["left_ankle_roll_link", "right_ankle_roll_link"])

    # ── Per-joint PD overrides (Newton only, default None = legacy) ──
    # When set, ``_newton_builders`` feeds these per-joint dicts to the
    # actuator's ``stiffness`` / ``damping`` (which natively accept a
    # ``{joint_regex: value}`` map) in place of the nominal ``p_gains`` /
    # ``d_gains``, so callers can pin heterogeneous per-joint PD
    # without touching the actuator wiring. Keys are joint-name regexes —
    # for per-DOF identification use exact joint names. Default ``None``
    # means "no override" (every existing training run is bit-identical to
    # before these fields were added).
    kp_per_dof_override: dict[str, float] | None = None
    kd_per_dof_override: dict[str, float] | None = None

    @property
    def action_scale(self) -> Dict[str, float]:
        return dict(G1_ACTION_SCALE)
