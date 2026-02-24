from dataclasses import dataclass, field
from typing import Dict, List

from .base import RobotConfig


@dataclass
class G1Config(RobotConfig):
    """Configuration for Unitree Go1 quadruped robot."""

    # Identity
    name: str = "g1"
    urdf_path: str = "./JaxRLWorld/rlworld/assets/g1_description/g1_12dof.urdf"

    # Physical properties
    # base_init_height: float = 0.8
    base_init_height: float = 0.79
    base_link_name: str = "torso_link"

    # Joint configuration
    # default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
    #    'left_hip_yaw_joint': 0.0 ,
    #    'left_hip_roll_joint': 0.0,
    #    'left_hip_pitch_joint': -0.1,
    #    'left_knee_joint': 0.3,
    #    'left_ankle_pitch_joint': -0.2,
    #    'left_ankle_roll_joint': 0.0,
    #    'right_hip_yaw_joint': 0.0,
    #    'right_hip_roll_joint': 0.0,
    #    'right_hip_pitch_joint': -0.1,
    #    'right_knee_joint': 0.3,
    #    'right_ankle_pitch_joint': -0.2,
    #    'right_ankle_roll_joint': 0.0,
    # })

    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        'left_hip_pitch_joint': -0.312,
        'right_hip_pitch_joint': -0.312,
        'left_hip_roll_joint': 0.0,
        'right_hip_roll_joint': 0.0,
        'left_hip_yaw_joint': 0.0,
        'right_hip_yaw_joint': 0.0,
        'left_knee_joint': 0.669,
        'right_knee_joint': 0.669,
        'left_ankle_pitch_joint': -0.363,
        'right_ankle_pitch_joint': -0.363,
        'left_ankle_roll_joint': 0.0,
        'right_ankle_roll_joint': 0.0
    })

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: [
            'left_hip_yaw_joint',
            'left_hip_roll_joint',
            'left_hip_pitch_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_hip_yaw_joint',
            'right_hip_roll_joint',
            'right_hip_pitch_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint',
        ]
    )

    foot_names: str | list[str] = field(default_factory=lambda:".*_ankle_roll_link")

    # Control gains
    p_gains: Dict[str, float] = field(default_factory=lambda:{
        ".*hip.*": 150.0,
        ".*knee.*": 200.0,
        ".*ankle.*": 20.0,
    })

    d_gains: Dict[str, float] = field(default_factory=lambda: {
        ".*hip.*": 5.0,
        ".*knee.*": 5.0,
        ".*ankle.*": 2.0,
    })

    armature: Dict[str, float] = field(default_factory=lambda: {
        ".*hip.*": 0.01,
        ".*knee.*": 0.01,
    })