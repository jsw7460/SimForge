from dataclasses import dataclass, field
from typing import Dict, List

from rlworld.rl.configs.robots.utils import reflected_inertia_from_two_stage_planetary

from .base import RobotConfig

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
ARMATURE_WAIST = ARMATURE_5020 * 2
ARMATURE_ANKLE = ARMATURE_5020 * 2

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

# Motor effort limits
EFFORT_5020 = 25.0
EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_4010 = 5.0
EFFORT_WAIST = EFFORT_5020 * 2
EFFORT_ANKLE = EFFORT_5020 * 2

# Action scale: 0.25 * effort / stiffness
ACTION_SCALE_5020 = 0.25 * EFFORT_5020 / STIFFNESS_5020
ACTION_SCALE_7520_14 = 0.25 * EFFORT_7520_14 / STIFFNESS_7520_14
ACTION_SCALE_7520_22 = 0.25 * EFFORT_7520_22 / STIFFNESS_7520_22
ACTION_SCALE_4010 = 0.25 * EFFORT_4010 / STIFFNESS_4010
ACTION_SCALE_WAIST = 0.25 * EFFORT_WAIST / STIFFNESS_WAIST
ACTION_SCALE_ANKLE = 0.25 * EFFORT_ANKLE / STIFFNESS_ANKLE

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
    urdf_path: str | None = "./JaxRLWorld/rlworld/assets/g1_description/g1_29dof.urdf"
    mjcf_path: str | None = "./JaxRLWorld/rlworld/assets/mujoco_menagerie/unitree_g1/g1.xml"
    # mjcf_path: str | None = "./Mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"

    base_init_height: float = 0.74
    base_link_name: str = "torso_link"

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
            r"left_(?!hand_palm_joint).*",
            r"right_(?!hand_palm_joint).*",
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

    @property
    def action_scale(self) -> Dict[str, float]:
        return dict(G1_ACTION_SCALE)
