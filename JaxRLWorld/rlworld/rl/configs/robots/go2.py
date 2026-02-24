from dataclasses import dataclass, field
from typing import Dict, List

from .base import RobotConfig


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

    foot_names: List[str] = field(default_factory=lambda: [
        "FR_foot", "FL_foot", "RL_foot", "RR_foot"
    ])

    @property
    def prefixed_foot_names(self) -> tuple[str, ...]:
        return self.prefixed_list(self.foot_names)