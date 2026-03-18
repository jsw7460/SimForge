from dataclasses import dataclass, field
from typing import Dict, List

from .base import RobotConfig
from rlworld.rl.configs.robots.utils import reflected_inertia_simple

# Go2 motor specs (from mjlab_sim2real)
# Ref: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf#L90
ROTOR_INERTIA = 0.000111842

# Gear ratios
# Ref: https://www.unitree.com/cn/go1/motor
HIP_GEAR_RATIO = 6.33
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.92

ARMATURE_HIP = reflected_inertia_simple(ROTOR_INERTIA, HIP_GEAR_RATIO)
ARMATURE_KNEE = reflected_inertia_simple(ROTOR_INERTIA, KNEE_GEAR_RATIO)

# PD gains derived from reflected inertia
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP = ARMATURE_HIP * NATURAL_FREQ**2
DAMPING_HIP = 2 * DAMPING_RATIO * ARMATURE_HIP * NATURAL_FREQ

STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ**2
DAMPING_KNEE = 2 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ

# Effort limits
EFFORT_HIP = 23.7
EFFORT_KNEE = 45.43

# Action scale: 0.25 * effort / stiffness
ACTION_SCALE_HIP = 0.25 * EFFORT_HIP / STIFFNESS_HIP
ACTION_SCALE_KNEE = 0.25 * EFFORT_KNEE / STIFFNESS_KNEE

GO2_ACTION_SCALE: Dict[str, float] = {
    r".*_hip_joint": ACTION_SCALE_HIP,
    r".*_thigh_joint": ACTION_SCALE_HIP,
    r".*_calf_joint": ACTION_SCALE_KNEE,
}


@dataclass
class Go2Config(RobotConfig):
    """Configuration for Unitree Go2 quadruped robot."""

    name: str = "go2_description"
    urdf_path: str = "Genesis/genesis/assets/urdf/go2/urdf/go2.urdf"

    base_init_height: float = 0.278
    base_link_name: str = "base"

    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
        ".*R_hip_joint": 0.1,
        ".*L_hip_joint": -0.1,
    })

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: [
            r"FL_(hip|thigh|calf)_joint",
            r"FR_(hip|thigh|calf)_joint",
            r"RL_(hip|thigh|calf)_joint",
            r"RR_(hip|thigh|calf)_joint",
        ]
    )

    p_gains: Dict[str, float] = field(default_factory=lambda: {
        ".*_hip_joint": STIFFNESS_HIP,
        ".*_thigh_joint": STIFFNESS_HIP,
        ".*_calf_joint": STIFFNESS_KNEE,
    })

    d_gains: Dict[str, float] = field(default_factory=lambda: {
        ".*_hip_joint": DAMPING_HIP,
        ".*_thigh_joint": DAMPING_HIP,
        ".*_calf_joint": DAMPING_KNEE,
    })

    armature: Dict[str, float] = field(default_factory=lambda: {
        ".*_hip_joint": ARMATURE_HIP,
        ".*_thigh_joint": ARMATURE_HIP,
        ".*_calf_joint": ARMATURE_KNEE,
    })

    foot_names: List[str] = field(default_factory=lambda: [
        "FR_foot", "FL_foot", "RL_foot", "RR_foot"
    ])

    @property
    def prefixed_foot_names(self) -> tuple[str, ...]:
        return self.prefixed_list(self.foot_names)