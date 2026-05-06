from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from genesis.utils.misc import tensor_to_array

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class HUDItemConfig:
    """Base configuration for HUD items."""

    enabled: bool = True


class HUDItem(ABC):
    """
    Abstract base class for HUD display items.

    Each item is responsible for fetching its own data from the environment.
    """

    def __init__(self, config: HUDItemConfig | None = None):
        self.config = config or HUDItemConfig()
        self._enabled = self.config.enabled

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this item."""
        pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @abstractmethod
    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        """
        Build display lines for this item.

        Args:
            env: GenesisEnv instance to fetch data from
            env_idx: Environment index to display

        Returns:
            List of strings or dicts (for bar graphs).
            Dict format: {"type": "bar", "label": str, "value": float, "max_value": float}
        """
        pass


# =============================================================================
# Concrete HUD Items
# =============================================================================


@dataclass
class BaseHeightItemConfig(HUDItemConfig):
    """Config for base height display."""

    label: str = "Height"
    precision: int = 3


class BaseHeightItem(HUDItem):
    """Displays robot base height."""

    def __init__(self, config: BaseHeightItemConfig | None = None):
        super().__init__(config or BaseHeightItemConfig())
        self.config: BaseHeightItemConfig = self.config

    @property
    def name(self) -> str:
        return "base_height"

    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        robot = env.scene_manager.entities.get("robot")
        if robot is None:
            return []

        base_pos = tensor_to_array(robot.get_pos())
        height = float(base_pos[env_idx, 2])
        precision = self.config.precision
        return [f"{self.config.label}: {height:.{precision}f}m"]


@dataclass
class CommandVelItemConfig(HUDItemConfig):
    """Config for command velocity display."""

    show_linear: bool = True
    show_angular: bool = True
    precision: int = 2


class CommandVelItem(HUDItem):
    """Displays command velocities."""

    def __init__(self, config: CommandVelItemConfig | None = None):
        super().__init__(config or CommandVelItemConfig())
        self.config: CommandVelItemConfig = self.config

    @property
    def name(self) -> str:
        return "command_vel"

    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        cmd_manager = getattr(env, "command_manager", None)
        if cmd_manager is None:
            return []

        lines = []
        precision = self.config.precision

        if self.config.show_linear:
            vx = vy = 0.0
            if hasattr(cmd_manager, "lin_vel_x"):
                vx = float(tensor_to_array(cmd_manager.lin_vel_x)[env_idx])
            if hasattr(cmd_manager, "lin_vel_y"):
                vy = float(tensor_to_array(cmd_manager.lin_vel_y)[env_idx])
            lines.append(f"Cmd vel: ({vx:.{precision}f}, {vy:.{precision}f})")

        if self.config.show_angular:
            if hasattr(cmd_manager, "ang_vel"):
                ang = float(tensor_to_array(cmd_manager.ang_vel)[env_idx])
                lines.append(f"Cmd ang: {ang:.{precision}f}")

        return lines


@dataclass
class FeetHeightItemConfig(HUDItemConfig):
    """Config for feet height display."""

    feet_names: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    feet_links: tuple[str, ...] | None = None  # Auto-generate if None
    bar_width: int = 60
    bar_height: int = 12
    max_height: float = 0.15
    show_bars: bool = True
    precision: int = 2


class FeetHeightItem(HUDItem):
    """Displays feet heights with optional bar graphs."""

    def __init__(self, config: FeetHeightItemConfig | None = None):
        super().__init__(config or FeetHeightItemConfig())
        self.config: FeetHeightItemConfig = self.config
        self._links_idx: list[int] | None = None

    @property
    def name(self) -> str:
        return "feet_height"

    def _get_feet_links(self) -> tuple[str, ...]:
        """Get feet link names."""
        if self.config.feet_links is not None:
            return self.config.feet_links
        return tuple(f"{name}_foot" if not name.endswith("_foot") else name for name in self.config.feet_names)

    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        robot = env.scene_manager.entities.get("robot")
        if robot is None:
            return []

        # Get feet link indices (cached)
        if self._links_idx is None:
            try:
                from rlworld.rl.utils import entity_utils as eu

                feet_links = list(self._get_feet_links())
                self._links_idx, _ = eu.find_links(robot, feet_links, global_ids=False, preserve_order=True)
            except Exception:
                return []

        # Get feet positions
        try:
            feet_pos = robot.get_links_pos(links_idx_local=self._links_idx)
            feet_h = tensor_to_array(feet_pos[env_idx, :, 2])
        except Exception:
            return []

        lines: list[str | dict] = []
        precision = self.config.precision

        for i, name in enumerate(self.config.feet_names):
            if i >= len(feet_h):
                break

            value = float(feet_h[i])

            if self.config.show_bars:
                lines.append(
                    {
                        "type": "bar",
                        "label": name,
                        "value": value,
                        "max_value": self.config.max_height,
                        "bar_width": self.config.bar_width,
                        "bar_height": self.config.bar_height,
                    }
                )
            else:
                lines.append(f"{name}: {value:.{precision}f}m")

        return lines


@dataclass
class EpisodeInfoItemConfig(HUDItemConfig):
    """Config for episode info display."""

    show_episode_count: bool = True
    show_step: bool = True


class EpisodeInfoItem(HUDItem):
    """Displays episode information."""

    def __init__(self, config: EpisodeInfoItemConfig | None = None):
        super().__init__(config or EpisodeInfoItemConfig())
        self.config: EpisodeInfoItemConfig = self.config
        self._episode_count: int = 0

    @property
    def name(self) -> str:
        return "episode_info"

    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        lines = []

        if self.config.show_episode_count:
            lines.append(f"Episode: {self._episode_count}")

        if self.config.show_step:
            term_manager = getattr(env, "termination_manager", None)
            if term_manager is not None:
                episode_buf = getattr(term_manager, "episode_length_buf", None)
                if episode_buf is not None:
                    step = int(tensor_to_array(episode_buf)[env_idx])
                    lines.append(f"Step: {step}")

        return lines

    def increment_episode(self) -> None:
        """Call this on episode reset."""
        self._episode_count += 1

    def reset_count(self) -> None:
        """Reset episode counter."""
        self._episode_count = 0


@dataclass
class DOFPositionItemConfig(HUDItemConfig):
    """Config for DOF position display."""

    joint_names: tuple[str, ...] | None = None  # None = show all
    precision: int = 3
    show_header: bool = True
    header_text: str = "Joint Positions"
    skip_base_joint: bool = True  # Skip joint 0 (free joint)


class DOFPositionItem(HUDItem):
    """Displays joint positions."""

    def __init__(self, config: DOFPositionItemConfig | None = None):
        super().__init__(config or DOFPositionItemConfig())
        self.config: DOFPositionItemConfig = self.config

    @property
    def name(self) -> str:
        return "joint_position"

    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        robot = env.scene_manager.entities.get("robot")
        if robot is None:
            return []

        dof_pos = tensor_to_array(robot.get_dofs_position())[env_idx]
        joints = robot.joints

        lines: list[str | dict] = []

        if self.config.show_header:
            lines.append(f"=== {self.config.header_text} ===")

        precision = self.config.precision
        dof_idx = 0

        for i, joint in enumerate(joints):
            # Skip base joint
            if self.config.skip_base_joint and i == 0:
                dof_idx += joint.n_dofs
                continue

            # Filter by name
            if self.config.joint_names is not None and joint.name not in self.config.joint_names:
                dof_idx += joint.n_dofs
                continue

            n_dofs = joint.n_dofs

            if n_dofs == 1:
                val = float(dof_pos[dof_idx])
                lines.append(f"{joint.name}: {val:.{precision}f}")
            else:
                vals = dof_pos[dof_idx : dof_idx + n_dofs]
                val_str = ", ".join(f"{v:.{precision}f}" for v in vals)
                lines.append(f"{joint.name}: ({val_str})")

            dof_idx += n_dofs

        return lines


@dataclass
class DOFVelocityItemConfig(HUDItemConfig):
    """Config for DOF velocity display."""

    dof_names: tuple[str, ...] | None = None
    precision: int = 3
    show_header: bool = True
    header_text: str = "DOF Velocities"


class DOFVelocityItem(HUDItem):
    """Displays DOF (joint) velocities."""

    def __init__(self, config: DOFVelocityItemConfig | None = None):
        super().__init__(config or DOFVelocityItemConfig())
        self.config: DOFVelocityItemConfig = self.config

    @property
    def name(self) -> str:
        return "dof_velocity"

    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        robot = env.scene_manager.entities.get("robot")
        if robot is None:
            return []

        dof_vel = tensor_to_array(robot.get_dofs_velocity())
        dof_names = [dof.name for dof in robot.dofs]

        lines: list[str | dict] = []

        if self.config.show_header:
            lines.append(f"=== {self.config.header_text} ===")

        dof_vals = dof_vel[env_idx]
        precision = self.config.precision

        for i, val in enumerate(dof_vals):
            name = dof_names[i] if i < len(dof_names) else f"dof_{i}"

            if self.config.dof_names is not None and name not in self.config.dof_names:
                continue

            lines.append(f"{name}: {float(val):.{precision}f}")

        return lines


@dataclass
class BaseVelocityItemConfig(HUDItemConfig):
    """Config for base velocity display."""

    show_linear: bool = True
    show_angular: bool = True
    precision: int = 3


class BaseVelocityItem(HUDItem):
    """Displays robot base linear and angular velocity."""

    def __init__(self, config: BaseVelocityItemConfig | None = None):
        super().__init__(config or BaseVelocityItemConfig())
        self.config: BaseVelocityItemConfig = self.config

    @property
    def name(self) -> str:
        return "base_velocity"

    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        robot = env.scene_manager.entities.get("robot")
        if robot is None:
            return []

        lines = []
        precision = self.config.precision

        if self.config.show_linear:
            lin_vel = tensor_to_array(robot.get_vel())[env_idx]
            lines.append(
                f"Lin vel: ({lin_vel[0]:.{precision}f}, {lin_vel[1]:.{precision}f}, {lin_vel[2]:.{precision}f})"
            )

        if self.config.show_angular:
            ang_vel = tensor_to_array(robot.get_ang())[env_idx]
            lines.append(
                f"Ang vel: ({ang_vel[0]:.{precision}f}, {ang_vel[1]:.{precision}f}, {ang_vel[2]:.{precision}f})"
            )

        return lines


@dataclass
class LinkPositionItemConfig(HUDItemConfig):
    """Config for link position display."""

    link_patterns: tuple[str, ...] = ()  # Regex patterns for link names
    show_header: bool = True
    header_text: str = "Link Positions"
    precision: int = 3
    preserve_order: bool = True  # Match order of patterns


class LinkPositionItem(HUDItem):
    """Displays link positions matching regex patterns."""

    def __init__(self, config: LinkPositionItemConfig | None = None):
        super().__init__(config or LinkPositionItemConfig())
        self.config: LinkPositionItemConfig = self.config
        self._link_ids: list[int] | None = None
        self._link_names: list[str] | None = None

    @property
    def name(self) -> str:
        return "link_position"

    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        robot = env.scene_manager.entities.get("robot")
        if robot is None or not self.config.link_patterns:
            return []

        # Cache link indices on first call
        if self._link_ids is None:
            try:
                from rlworld.rl.utils import entity_utils as eu

                self._link_ids, self._link_names = eu.find_links(
                    robot,
                    list(self.config.link_patterns),
                    global_ids=False,
                    preserve_order=self.config.preserve_order,
                )
            except ValueError:
                return []

        # Get link positions
        try:
            link_pos = tensor_to_array(robot.get_links_pos(links_idx_local=self._link_ids))[env_idx]
        except Exception:
            return []

        lines: list[str | dict] = []
        precision = self.config.precision

        if self.config.show_header:
            lines.append(f"=== {self.config.header_text} ===")

        for i, link_name in enumerate(self._link_names):
            pos = link_pos[i]
            lines.append(f"{link_name}: ({pos[0]:.{precision}f}, {pos[1]:.{precision}f}, {pos[2]:.{precision}f})")

        return lines


@dataclass
class CustomItemConfig(HUDItemConfig):
    """Config for custom display item."""

    item_name: str = "custom"
    label: str = ""
    precision: int = 3


class CustomItem(HUDItem):
    """
    Custom item with user-provided data fetch function.

    Example:
        def get_reward(env, env_idx):
            return env.reward_manager.total_reward[env_idx].item()

        item = CustomItem(
            fetch_fn=get_reward,
            config=CustomItemConfig(item_name="reward", label="Reward")
        )
    """

    def __init__(
        self,
        fetch_fn: callable,
        config: CustomItemConfig | None = None,
    ):
        super().__init__(config or CustomItemConfig())
        self.config: CustomItemConfig = self.config
        self._fetch_fn = fetch_fn

    @property
    def name(self) -> str:
        return self.config.item_name

    def build_lines(self, env: "GenesisEnv", env_idx: int) -> list[str | dict]:
        try:
            value = self._fetch_fn(env, env_idx)
        except Exception:
            return []

        label = self.config.label or self.config.item_name
        precision = self.config.precision

        if isinstance(value, (list, tuple, np.ndarray)):
            lines = []
            for i, v in enumerate(value):
                lines.append(f"{label}_{i}: {float(v):.{precision}f}")
            return lines
        else:
            return [f"{label}: {float(value):.{precision}f}"]
