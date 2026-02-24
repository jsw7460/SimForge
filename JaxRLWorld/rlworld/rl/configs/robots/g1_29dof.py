from dataclasses import dataclass, field
from typing import Dict, List

from .base import RobotConfig


@dataclass
class G1Config(RobotConfig):
    """Configuration for Unitree Go1 quadruped robot."""

    # Identity
    name: str = "g1_29dof"
    urdf_path: str | None = "./JaxRLWorld/rlworld/assets/g1_description/g1_29dof.urdf"

    # Physical properties
    base_init_height: float = 0.81
    base_link_name: str = "torso_link"

    # Joint configuration
    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        ".*_hip_pitch_joint": -0.20,
        ".*_knee_joint": 0.42,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_pitch_joint": 0.35,
        "left_wrist_roll_joint": 0.52,
        "right_shoulder_roll_joint": -0.16,
        "right_shoulder_pitch_joint": 0.35,
        "right_wrist_roll_joint": -0.52
    })

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: [
            r"left_(?!hand_palm_joint).*",
            r"right_(?!hand_palm_joint).*",
            r"waist_(?!support_joint).*"
        ]
    )

    # Control gains
    p_gains: Dict[str, float] = field(default_factory=lambda:{
        ".*_hip_yaw_joint": 150.0,
        ".*_hip_roll_joint": 150.0,
        ".*_hip_pitch_joint": 200.0,
        ".*_knee_joint": 200.0,
        ".*_ankle_pitch_joint": 20.0,
        ".*_ankle_roll_joint": 20.0,
        ".*_shoulder_pitch_joint": 40.0,
        ".*_shoulder_roll_joint": 40.0,
        ".*_shoulder_yaw_joint": 40.0,
        ".*_elbow_joint": 40.0,
    })

    d_gains: Dict[str, float] = field(default_factory=lambda: {
        ".*_hip_yaw_joint": 5.0,
        ".*_hip_roll_joint": 5.0,
        ".*_hip_pitch_joint": 5.0,
        ".*_knee_joint": 5.0,
        ".*_ankle_pitch_joint": 2.0,
        ".*_ankle_roll_joint": 2.0,
        ".*_shoulder_pitch_joint": 10.0,
        ".*_shoulder_roll_joint": 10.0,
        ".*_shoulder_yaw_joint": 10.0,
        ".*_elbow_joint": 10.0,
    })

    armature: Dict[str, float] = field(default_factory=lambda: {
        ".*hip.*": 0.01,
        ".*knee.*": 0.01,
        ".*_shoulder_.*": 0.01,
        ".*_elbow_.*": 0.01,
        ".*_wrist_.*": 0.001,
    })

    foot_names: str | list[str] = field(default_factory=lambda: ".*_ankle_roll_link")




# ============================================================
# mjlab G1 actuator constants (from Document 9)
# ============================================================

from rlworld.rl.configs.robots.utils import reflected_inertia_from_two_stage_planetary

# Motor 5020 (elbows, shoulders, wrist_roll)
ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_5020, GEARS_5020)

# Motor 7520_14 (hip_pitch, hip_yaw, waist_yaw)
ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_14, GEARS_7520_14)

# Motor 7520_22 (hip_roll, knee)
ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_22, GEARS_7520_22)

# Motor 4010 (wrist_pitch, wrist_yaw)
ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_4010, GEARS_4010)

# Parallel linkage actuators (2x 5020)
ARMATURE_WAIST = ARMATURE_5020 * 2  # waist_pitch, waist_roll
ARMATURE_ANKLE = ARMATURE_5020 * 2  # ankle_pitch, ankle_roll

# PD gains
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
STIFFNESS_WAIST = STIFFNESS_5020 * 2
STIFFNESS_ANKLE = STIFFNESS_5020 * 2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
DAMPING_WAIST = DAMPING_5020 * 2
DAMPING_ANKLE = DAMPING_5020 * 2

# Motor effort limits (from mjlab)
EFFORT_5020 = 25.0
EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_4010 = 5.0
EFFORT_WAIST = EFFORT_5020 * 2  # 50.0
EFFORT_ANKLE = EFFORT_5020 * 2  # 50.0

# Action scale: 0.25 * effort / stiffness
ACTION_SCALE_5020 = 0.25 * EFFORT_5020 / STIFFNESS_5020
ACTION_SCALE_7520_14 = 0.25 * EFFORT_7520_14 / STIFFNESS_7520_14
ACTION_SCALE_7520_22 = 0.25 * EFFORT_7520_22 / STIFFNESS_7520_22
ACTION_SCALE_4010 = 0.25 * EFFORT_4010 / STIFFNESS_4010
ACTION_SCALE_WAIST = 0.25 * EFFORT_WAIST / STIFFNESS_WAIST  # Same as ACTION_SCALE_5020
ACTION_SCALE_ANKLE = 0.25 * EFFORT_ANKLE / STIFFNESS_ANKLE  # Same as ACTION_SCALE_5020

G1_ACTION_SCALE: Dict[str, float] = {
    # 7520_14: hip_pitch, hip_yaw, waist_yaw
    r".*_hip_pitch_joint": ACTION_SCALE_7520_14,
    r".*_hip_yaw_joint": ACTION_SCALE_7520_14,
    r"waist_yaw_joint": ACTION_SCALE_7520_14,
    # 7520_22: hip_roll, knee
    r".*_hip_roll_joint": ACTION_SCALE_7520_22,
    r".*_knee_joint": ACTION_SCALE_7520_22,
    # 5020 x2: waist_pitch, waist_roll, ankles
    r"waist_pitch_joint": ACTION_SCALE_WAIST,
    r"waist_roll_joint": ACTION_SCALE_WAIST,
    r".*_ankle_pitch_joint": ACTION_SCALE_ANKLE,
    r".*_ankle_roll_joint": ACTION_SCALE_ANKLE,
    # 5020: shoulders, elbows, wrist_roll
    r".*_shoulder_pitch_joint": ACTION_SCALE_5020,
    r".*_shoulder_roll_joint": ACTION_SCALE_5020,
    r".*_shoulder_yaw_joint": ACTION_SCALE_5020,
    r".*_elbow_joint": ACTION_SCALE_5020,
    r".*_wrist_roll_joint": ACTION_SCALE_5020,
    # 4010: wrist_pitch, wrist_yaw
    r".*_wrist_pitch_joint": ACTION_SCALE_4010,
    r".*_wrist_yaw_joint": ACTION_SCALE_4010,
}


@dataclass
class G1MjlabConfig(RobotConfig):
    """G1 config with mjlab-derived actuator parameters."""

    name: str = "g1_29dof"
    urdf_path: str | None = "./JaxRLWorld/rlworld/assets/g1_description/g1_29dof.urdf"

    # From mjlab KNEES_BENT_KEYFRAME
    base_init_height: float = 0.76
    base_link_name: str = "torso_link"

    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    })

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: [
            r"left_(?!hand_palm_joint).*",
            r"right_(?!hand_palm_joint).*",
            r"waist_(?!support_joint).*"
        ]
    )

    p_gains: Dict[str, float] = field(default_factory=lambda: {
        # 7520_14: hip_pitch, hip_yaw, waist_yaw
        ".*_hip_pitch_joint": STIFFNESS_7520_14,
        ".*_hip_yaw_joint": STIFFNESS_7520_14,
        "waist_yaw_joint": STIFFNESS_7520_14,
        # 7520_22: hip_roll, knee
        ".*_hip_roll_joint": STIFFNESS_7520_22,
        ".*_knee_joint": STIFFNESS_7520_22,
        # 5020 x2: waist_pitch, waist_roll, ankles
        "waist_pitch_joint": STIFFNESS_WAIST,
        "waist_roll_joint": STIFFNESS_WAIST,
        ".*_ankle_pitch_joint": STIFFNESS_ANKLE,
        ".*_ankle_roll_joint": STIFFNESS_ANKLE,
        # 5020: shoulders, elbows, wrist_roll
        ".*_shoulder_pitch_joint": STIFFNESS_5020,
        ".*_shoulder_roll_joint": STIFFNESS_5020,
        ".*_shoulder_yaw_joint": STIFFNESS_5020,
        ".*_elbow_joint": STIFFNESS_5020,
        ".*_wrist_roll_joint": STIFFNESS_5020,
        # 4010: wrist_pitch, wrist_yaw
        ".*_wrist_pitch_joint": STIFFNESS_4010,
        ".*_wrist_yaw_joint": STIFFNESS_4010,
    })

    d_gains: Dict[str, float] = field(default_factory=lambda: {
        # 7520_14
        ".*_hip_pitch_joint": DAMPING_7520_14,
        ".*_hip_yaw_joint": DAMPING_7520_14,
        "waist_yaw_joint": DAMPING_7520_14,
        # 7520_22
        ".*_hip_roll_joint": DAMPING_7520_22,
        ".*_knee_joint": DAMPING_7520_22,
        # 5020 x2
        "waist_pitch_joint": DAMPING_WAIST,
        "waist_roll_joint": DAMPING_WAIST,
        ".*_ankle_pitch_joint": DAMPING_ANKLE,
        ".*_ankle_roll_joint": DAMPING_ANKLE,
        # 5020
        ".*_shoulder_pitch_joint": DAMPING_5020,
        ".*_shoulder_roll_joint": DAMPING_5020,
        ".*_shoulder_yaw_joint": DAMPING_5020,
        ".*_elbow_joint": DAMPING_5020,
        ".*_wrist_roll_joint": DAMPING_5020,
        # 4010
        ".*_wrist_pitch_joint": DAMPING_4010,
        ".*_wrist_yaw_joint": DAMPING_4010,
    })

    armature: Dict[str, float] = field(default_factory=lambda: {
        # 7520_14
        ".*_hip_pitch_joint": ARMATURE_7520_14,
        ".*_hip_yaw_joint": ARMATURE_7520_14,
        "waist_yaw_joint": ARMATURE_7520_14,
        # 7520_22
        ".*_hip_roll_joint": ARMATURE_7520_22,
        ".*_knee_joint": ARMATURE_7520_22,
        # 5020 x2
        "waist_pitch_joint": ARMATURE_WAIST,
        "waist_roll_joint": ARMATURE_WAIST,
        ".*_ankle_pitch_joint": ARMATURE_ANKLE,
        ".*_ankle_roll_joint": ARMATURE_ANKLE,
        # 5020
        ".*_shoulder_pitch_joint": ARMATURE_5020,
        ".*_shoulder_roll_joint": ARMATURE_5020,
        ".*_shoulder_yaw_joint": ARMATURE_5020,
        ".*_elbow_joint": ARMATURE_5020,
        ".*_wrist_roll_joint": ARMATURE_5020,
        # 4010
        ".*_wrist_pitch_joint": ARMATURE_4010,
        ".*_wrist_yaw_joint": ARMATURE_4010,
    })

    foot_names: str | list[str] = field(
        default_factory=lambda: ["left_ankle_roll_link", "right_ankle_roll_link"]
    )