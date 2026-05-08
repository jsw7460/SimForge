from dataclasses import dataclass, field
from typing import Dict, List

from rlworld.rl.configs.robots.utils import reflected_inertia_simple

from .base import RobotConfig

# Go2 motor specs (from mjlab_sim2real)
# Ref: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf#L90
ROTOR_INERTIA = 0.000111842  # kg·m^2

# Gear ratios
# Ref: https://www.unitree.com/cn/go1/motor
HIP_GEAR_RATIO = 6.33
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.92  # ≈ 12.1536

ARMATURE_HIP = reflected_inertia_simple(ROTOR_INERTIA, HIP_GEAR_RATIO)  # ≈ 0.004481  kg·m^2
ARMATURE_KNEE = reflected_inertia_simple(ROTOR_INERTIA, KNEE_GEAR_RATIO)  # ≈ 0.016520  kg·m^2

# PD gains derived from reflected inertia
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz → ≈ 62.8319 rad/s
DAMPING_RATIO = 2.0

STIFFNESS_HIP = ARMATURE_HIP * NATURAL_FREQ**2  # ≈ 17.6918  N·m/rad
DAMPING_HIP = 2 * DAMPING_RATIO * ARMATURE_HIP * NATURAL_FREQ  # ≈  1.1263  N·m·s/rad

STIFFNESS_KNEE = ARMATURE_KNEE * NATURAL_FREQ**2  # ≈ 65.2191  N·m/rad
DAMPING_KNEE = 2 * DAMPING_RATIO * ARMATURE_KNEE * NATURAL_FREQ  # ≈  4.1520  N·m·s/rad

# Effort limits
EFFORT_HIP = 23.7  # N·m
EFFORT_KNEE = 45.43  # N·m

# Action scale: 0.25 * effort / stiffness
ACTION_SCALE_HIP = 0.25 * EFFORT_HIP / STIFFNESS_HIP  # ≈ 0.3349  rad
ACTION_SCALE_KNEE = 0.25 * EFFORT_KNEE / STIFFNESS_KNEE  # ≈ 0.1741  rad

GO2_ACTION_SCALE: Dict[str, float] = {
    r".*_hip_joint": ACTION_SCALE_HIP,
    r".*_thigh_joint": ACTION_SCALE_HIP,
    r".*_calf_joint": ACTION_SCALE_KNEE,
}


@dataclass
class Go2Config(RobotConfig):
    """Configuration for Unitree Go2 quadruped robot."""

    name: str = "go2"
    urdf_path: str = "Genesis/genesis/assets/urdf/go2/urdf/go2.urdf"
    mjcf_path: str = "Mjlab/src/mjlab/asset_zoo/robots/unitree_go2/xmls/go2.xml"

    base_init_height: float = 0.278
    # base_link_name: str = "base"
    base_link_name: str = "trunk"

    default_joint_angles: Dict[str, float] = field(
        default_factory=lambda: {
            ".*thigh_joint": 0.9,
            ".*calf_joint": -1.8,
            ".*R_hip_joint": 0.1,
            ".*L_hip_joint": -0.1,
        }
    )

    actuated_dof_patterns: List[str] = field(
        default_factory=lambda: [
            r"FL_(hip|thigh|calf)_joint",
            r"FR_(hip|thigh|calf)_joint",
            r"RL_(hip|thigh|calf)_joint",
            r"RR_(hip|thigh|calf)_joint",
        ]
    )

    p_gains: Dict[str, float] = field(
        default_factory=lambda: {
            ".*_hip_joint": STIFFNESS_HIP,
            ".*_thigh_joint": STIFFNESS_HIP,
            ".*_calf_joint": STIFFNESS_KNEE,
        }
    )

    d_gains: Dict[str, float] = field(
        default_factory=lambda: {
            ".*_hip_joint": DAMPING_HIP,
            ".*_thigh_joint": DAMPING_HIP,
            ".*_calf_joint": DAMPING_KNEE,
        }
    )

    armature: Dict[str, float] = field(
        default_factory=lambda: {
            ".*_hip_joint": ARMATURE_HIP,
            ".*_thigh_joint": ARMATURE_HIP,
            ".*_calf_joint": ARMATURE_KNEE,
        }
    )

    foot_names: List[str] = field(default_factory=lambda: ["FR_foot", "FL_foot", "RR_foot", "RL_foot"])

    # ── SysID-result override fields (optional, default None = legacy) ──
    # When set on a preset (e.g. via the SysID-aligned training script),
    # downstream builders use these values instead of the module-level
    # ``STIFFNESS_*`` / ``DAMPING_*`` constants and install deterministic
    # friction event terms. Default ``None`` means "no override" — every
    # existing training run is bit-identical to before this field was
    # added. See ``rlworld/rl/configs/presets/go2_flat/_newton_builders.py``
    # for the read sites.
    kp_hip_override: float | None = None
    kd_hip_override: float | None = None
    kp_knee_override: float | None = None
    kd_knee_override: float | None = None
    joint_frictionloss_override: float | None = None
    foot_friction_override: float | None = 0.3
