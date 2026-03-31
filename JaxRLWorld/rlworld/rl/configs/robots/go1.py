from dataclasses import dataclass, field
from typing import Dict, List

from .base import RobotConfig


@dataclass
class Go1Config(RobotConfig):
    """Configuration for Unitree Go1 quadruped robot."""

    # Identity
    name: str = "go1"
    urdf_path: str = "./JaxRLWorld/rlworld/assets/go1_model_clean/urdf/go1_simplified_stl.urdf"

    # Physical properties
    base_init_height: float = 0.41
    base_link_name: str = "base"

    # Joint configuration
    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        "FL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RL_hip_joint": 0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    })

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: ["FL.*", "RL.*", "FR.*", "RR.*"]
    )

    # Control gains
    p_gains: Dict[str, float] = field(default_factory=lambda: {
        "FL.*": 20.0,
        "FR.*": 20.0,
        "RL.*": 20.0,
        "RR.*": 20.0,
    })

    d_gains: Dict[str, float] = field(default_factory=lambda: {
        "FL.*": 0.5,
        "FR.*": 0.5,
        "RL.*": 0.5,
        "RR.*": 0.5,
    })

    armature: Dict[str, float] = field(default_factory=lambda: {
        "FL.*": 0.1,
        "FR.*": 0.1,
        "RL.*": 0.1,
        "RR.*": 0.1,
    })

    foot_names: List[str] = field(default_factory=lambda: ["FR_foot", "FL_foot", "RL_foot", "RR_foot"])



from dataclasses import dataclass, field
from typing import Dict, List
from rlworld.rl.configs.robots.base import RobotConfig

# mjlab constants
ROTOR_INERTIA = 0.000111842
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

HIP_REFLECTED_INERTIA = ROTOR_INERTIA * HIP_GEAR_RATIO**2    # ~0.00403
KNEE_REFLECTED_INERTIA = ROTOR_INERTIA * KNEE_GEAR_RATIO**2  # ~0.00906

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_REFLECTED_INERTIA * NATURAL_FREQ**2      # ~15.87
DAMPING_HIP = 2 * DAMPING_RATIO * HIP_REFLECTED_INERTIA * NATURAL_FREQ  # ~3.18

STIFFNESS_KNEE = KNEE_REFLECTED_INERTIA * NATURAL_FREQ**2    # ~35.71
DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_REFLECTED_INERTIA * NATURAL_FREQ  # ~4.77


@dataclass
class Go1MujocoConfig(RobotConfig):
    """Go1 config with mjlab-derived actuator parameters."""

    name: str = "go1"
    urdf_path: str = "./JaxRLWorld/rlworld/assets/go1_model_clean/urdf/go1_simplified_stl.urdf"

    base_init_height: float = 0.278  # from mjlab INIT_STATE
    base_link_name: str = "base"

    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        "FL_hip_joint": -0.1,
        "FR_hip_joint": 0.1,
        "RL_hip_joint": -0.1,
        "RR_hip_joint": 0.1,
        "FL_thigh_joint": 0.9,
        "FR_thigh_joint": 0.9,
        "RL_thigh_joint": 0.9,
        "RR_thigh_joint": 0.9,
        "FL_calf_joint": -1.8,
        "FR_calf_joint": -1.8,
        "RL_calf_joint": -1.8,
        "RR_calf_joint": -1.8,
    })

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: ["FL.*", "RL.*", "FR.*", "RR.*"]
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
        ".*_hip_joint": HIP_REFLECTED_INERTIA,
        ".*_thigh_joint": HIP_REFLECTED_INERTIA,
        ".*_calf_joint": KNEE_REFLECTED_INERTIA,
    })

    foot_names: List[str] = field(default_factory=lambda: [
        "FR_foot", "FL_foot", "RL_foot", "RR_foot"
    ])