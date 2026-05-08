"""Base robot configuration class.

Names (joint / body / action / gain / armature dict keys) are always
bare leaf names. Entity-level scoping happens at the Newton
``ArticulationView`` layer (scene manager's ``_build_robot_view``
filters by entity prefix at view construction time), not by string
munging on the config side. This matches IsaacLab's Newton integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RobotConfig:
    """Base class for robot-specific configuration."""

    name: str = ""
    urdf_path: str | None = ""
    mjcf_path: str | None = ""
    usd_path: str | None = ""

    base_init_height: float = 0.0
    base_link_name: str = "base"

    default_joint_angles: Dict[str, float] = field(default_factory=dict)
    actuated_dof_patterns: List[str] = field(default_factory=list)

    p_gains: Dict[str, float] = field(default_factory=dict)
    d_gains: Dict[str, float] = field(default_factory=dict)
    armature: Dict[str, float] = field(default_factory=dict)

    def get_action_offset(self) -> Dict[str, float]:
        return self.default_joint_angles.copy()
