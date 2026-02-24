"""Base robot configuration class."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RobotConfig:
    """Base class for robot-specific configuration."""

    name: str = ""
    urdf_path: str | None = ""
    usd_path: str | None = ""

    base_init_height: float = 0.0
    base_link_name: str = "base"

    default_joint_angles: Dict[str, float] = field(default_factory=dict)
    actuated_dof_patterns: List[str] = field(default_factory=list)

    p_gains: Dict[str, float] = field(default_factory=dict)
    d_gains: Dict[str, float] = field(default_factory=dict)
    armature: Dict[str, float] = field(default_factory=dict)

    def prefixed(self, name: str) -> str:
        """Prepend robot name prefix if not already present."""
        if "/" in name:
            return name
        return f"{self.name}/{name}"

    def prefixed_list(self, names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        """Prepend robot name prefix to a list of names."""
        return tuple(self.prefixed(n) for n in names)

    @property
    def prefixed_actuated_dof_patterns(self) -> list[str]:
        return [f"{self.name}/{p}" for p in self.actuated_dof_patterns]

    @property
    def prefixed_p_gains(self) -> Dict[str, float]:
        return {f"{self.name}/{k}": v for k, v in self.p_gains.items()}

    @property
    def prefixed_d_gains(self) -> Dict[str, float]:
        return {f"{self.name}/{k}": v for k, v in self.d_gains.items()}

    @property
    def prefixed_armature(self) -> Dict[str, float]:
        return {f"{self.name}/{k}": v for k, v in self.armature.items()}

    def get_action_offset(self) -> Dict[str, float]:
        return self.default_joint_angles.copy()

    def get_prefixed_action_offset(self) -> Dict[str, float] | None:
        offset = self.get_action_offset()
        if not offset:
            return None
        return {f"{self.name}/{k}": v for k, v in offset.items()}