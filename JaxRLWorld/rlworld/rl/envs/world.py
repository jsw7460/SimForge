from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import torch
from gymnasium import spaces

from rlworld.rl.envs.lifecycle import LifecycleEvent, LifecycleManager

if TYPE_CHECKING:
    from rlworld.rl.envs.robot_data import RobotData


class World(ABC):
    """Abstract base class for all RL environments."""

    sim_name: str = "World"
    sim_type: str = "world"  # lowercase key for ManagerRegistry ("genesis", "newton", "mujoco")

    # Required attributes (set in subclass __init__)
    num_envs: int
    device: torch.device
    seed: int

    # Timing
    physics_dt: float
    decimation: int
    control_dt: float

    # Managers (set in subclass _setup_environment)
    scene_manager: Any
    obs_manager: Any
    act_manager: Any
    reward_manager: Any
    termination_manager: Any
    command_manager: Any
    event_manager: Any
    contact_manager: Any

    def __init__(self):
        super().__init__()
        self._cache_generation = 0
        self._env_step_counter = 0
        self.lifecycle = LifecycleManager()

    def _init_buffers(self) -> None:
        """Initialize common buffers. Call after setting num_envs and device."""
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums = defaultdict(
            lambda: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        )
        self.rew_buf_per_type = defaultdict(
            lambda: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        )
        self.extras = {}

    def _invalidate_cache(self) -> None:
        """Invalidate observation cache."""
        self._cache_generation += 1

    def _update_num_step_calls(self) -> None:
        self._env_step_counter += 1

    @property
    def env_step_counter(self) -> int:
        """Number of step() calls on this environment instance."""
        return self._env_step_counter

    # ========== Properties ==========

    @property
    def task_name(self):
        return self.env_cfg.task_name

    @property
    def action_low(self):
        return self.act_manager._clip_low

    @property
    def action_high(self):
        return self.act_manager._clip_high

    @property
    def num_actions(self) -> int:
        return self.act_manager.total_action_dim

    @property
    def max_episode_length(self) -> int:
        return self.termination_manager.max_episode_length

    @property
    def reset_buf(self) -> torch.Tensor:
        return self.termination_manager.reset_buf

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.termination_manager.episode_length_buf

    @property
    @abstractmethod
    def robot(self) -> Any:
        """Get the main robot entity/model."""
        pass

    @abstractmethod
    def get_robot_data(self, entity_name: str = "robot") -> "RobotData":
        """Get the RobotData interface for a named entity.

        Args:
            entity_name: Name of the entity in the scene (default: "robot").

        Returns:
            An object satisfying the ``RobotData`` protocol.
        """
        pass

    @property
    def robot_data(self) -> "RobotData":
        """Shortcut for ``get_robot_data("robot")``."""
        return self.get_robot_data("robot")

    @property
    def heading_w(self) -> torch.Tensor:
        """Get the heading (yaw angle) of the robot in world frame.

        Returns:
            Tensor of shape [num_envs] in radians.
        """
        return self.robot_data.heading_w

    def calculate_obs_dim(self) -> dict[str, int]:
        return self.obs_manager.calculate_obs_dim()

    @property
    def action_space(self) -> spaces.Box:
        """Get the action space (gymnasium-style).

        Returns:
            spaces.Box: Continuous action space with shape (num_actions,)
        """
        num_actions = self.act_manager.total_action_dim
        # Get clip range from action manager if available
        if hasattr(self.act_manager, 'clip') and self.act_manager.clip is not None:
            low, high = self.act_manager.clip
        elif hasattr(self.act_manager, 'clip_actions') and self.act_manager.clip_actions is not None:
            low, high = self.act_manager.clip_actions
        else:
            low, high = -np.inf, np.inf

        return spaces.Box(
            low=np.float32(low),
            high=np.float32(high),
            shape=(num_actions,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> Dict[str, spaces.Box]:
        """Get the observation space (gymnasium-style).

        Returns:
            Dict[str, spaces.Box]: Dictionary of observation spaces for each group
        """
        obs_dims = self.obs_manager.calculate_obs_dim()
        obs_spaces = {}

        for group_name, dim in obs_dims.items():
            obs_spaces[group_name] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dim,),
                dtype=np.float32,
            )

        return obs_spaces

    # ========== Environment Setup (phased lifecycle) ==========

    def _setup_environment(self) -> None:
        """Initialize all managers in a structured, phased sequence.

        Subclasses implement the abstract hooks (_build_scene,
        _build_sim_managers) and optionally override _post_setup for
        customisation.  The ManagerRegistry resolves backend-specific
        classes (including reward manager) automatically via sim_type.
        Lifecycle events fire at well-defined points so external code
        can hook in.
        """
        # Register backend-specific managers in the registry
        from rlworld.rl.envs.managers._registrations import register_all_for
        register_all_for(self.sim_type)

        # Phase 1 — Build physics scene (simulator-specific)
        self._build_scene()
        self.lifecycle.dispatch(LifecycleEvent.SCENE_BUILT)

        # Phase 2 — Create managers
        self._build_sim_managers()
        self._build_common_managers()
        self.lifecycle.dispatch(LifecycleEvent.MANAGERS_READY)

        # Phase 3 — Simulator-specific finalization
        self._post_setup()

        # Phase 4 — Startup events
        if hasattr(self, "event_manager") and self.event_manager is not None:
            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")

        self.lifecycle.dispatch(LifecycleEvent.ENV_READY)

        # Pretty print environment summary
        from rlworld.rl.utils.pretty import print_env_summary
        print_env_summary(self)

        # Print joint mapping for debugging cross-simulator consistency
        # if hasattr(self, "act_manager") and hasattr(self.act_manager, "print_joint_mapping"):
        #     self.act_manager.print_joint_mapping()

    @abstractmethod
    def _build_scene(self) -> None:
        """Create the scene manager and build the physics world.

        This is the first setup phase.  After this method returns the
        scene manager must be assigned to ``self.scene_manager`` and the
        scene must be fully built (entities registered, simulation ready).
        """

    @abstractmethod
    def _build_sim_managers(self) -> None:
        """Create simulator-specific managers.

        At minimum this must set ``self.act_manager``,
        ``self.obs_manager``, and ``self.contact_manager``.
        Visualization managers are also created here.
        """

    def _build_common_managers(self) -> None:
        """Create simulator-agnostic managers (command, reward, termination, event).

        The reward manager class is resolved via ManagerRegistry so that
        backends like MuJoCo automatically get MujocoRewardManager without
        needing a subclass hook.
        """
        from rlworld.rl.envs.managers import (
            CommandManager, CommandManagerConfig,
            RewardManagerConfig,
            TerminationManager, TerminationConfig,
            EventManager, EventManagerConfig,
        )
        from rlworld.rl.envs.managers.registry import ManagerRegistry

        self.command_manager = CommandManager(
            env=self,
            config=CommandManagerConfig(terms=self.command_cfg.terms),
        )

        reward_cls = ManagerRegistry.get_class(self.sim_type, "reward")
        self.reward_manager = reward_cls(
            env=self,
            config=self.reward_cfg,
        )

        self.termination_manager = TerminationManager(
            env=self,
            config=self.env_cfg.terminations,
            episode_length_s=self.env_cfg.episode_length_s,
        )

        self.event_manager = EventManager(
            env=self,
            config=self.event_cfg,
        )

    def _post_setup(self) -> None:
        """Simulator-specific finalization after all managers are created.

        Examples: Newton captures CUDA graphs, MuJoCo expands model fields
        for domain randomisation.  Called before startup events.
        """

    @abstractmethod
    def _step_physics(self) -> None:
        """Execute physics step(s). Implement in subclass."""
        pass

    # ========== Common Implementation ==========

    def get_observation(self):
        return self.obs_manager.get_observation()

    def step(self, actions: torch.Tensor) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute one environment step."""
        # Process and apply actions
        processed_actions = self.act_manager.process_actions(actions)
        self._apply_actions(processed_actions)

        # Step physics (simulator-specific)
        self._step_physics()

        self._invalidate_cache()

        # Update contact info
        self.contact_manager.advance()

        # Apply interval events
        if hasattr(self, 'event_manager') and self.event_manager is not None:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.control_dt)

        # Pre-reward hook (e.g., gait advance that rewards depend on)
        self._pre_reward_hook()

        # Compute rewards
        self.rew_buf[:] = 0.0
        self.reward_manager.set_rewards(
            reward_buffer=self.rew_buf,
            episode_sums=self.episode_sums,
            reward_buffer_per_type=self.rew_buf_per_type
        )

        # Pre-termination hook
        self._pre_termination_hook()

        # Check termination
        terminated, truncated = self.termination_manager.check_termination()
        reset_buf = terminated | truncated
        reset_env_ids = reset_buf.nonzero(as_tuple=False).flatten()

        # Handle terminal observations
        final_observation = None
        final_info = None
        if len(reset_env_ids) > 0:
            self.obs_manager.process_observations(update_history=True)
            final_observation = {
                key: obs.clone() for key, obs in self.obs_manager.obs_dict.items()
            }
            final_info = {
                "episode_reward_sums": deepcopy(self.episode_sums),
            }

        # Advance commands (timer-based resampling + per-step post-processing)
        self.command_manager.compute(self.control_dt)

        # Reset terminated environments
        self._reset_idx(reset_env_ids)
        self._invalidate_cache()

        # Advance managers
        self._advance_managers()
        self._update_num_step_calls()

        # Build extras
        self.extras = {
            "final_observation": final_observation,
            "final_info": final_info,
            "terminal_env_ids": reset_env_ids if len(reset_env_ids) > 0 else None,
            "rewards_per_type": self.rew_buf_per_type,
            "episode_reward_sums": deepcopy(self.episode_sums),
            **self.obs_manager.extras,
            **self.termination_manager.extras,
        }

        return self.obs_manager.get_observation(), self.rew_buf, terminated, truncated, self.extras

    def _apply_actions(self, processed_actions: torch.Tensor) -> None:
        """Apply processed actions. Override in subclass if needed."""
        if hasattr(self.act_manager, 'apply_actions'):
            self.act_manager.apply_actions(processed_actions)
        elif hasattr(self.act_manager, 'apply_dofs_position'):
            self.act_manager.apply_dofs_position(processed_actions)

    def _pre_reward_hook(self) -> None:
        """Override in subclass for logic that must run before reward computation.

        Example: advancing gait manager so desired_contact_states are available
        for gait-tracking rewards.
        """
        pass

    def _pre_termination_hook(self) -> None:
        """Override in subclass for pre-termination logic."""
        pass

    def _advance_managers(self) -> None:
        """Advance all managers. Override to add custom managers."""
        self.obs_manager.advance()
        self.reward_manager.advance()
        self.termination_manager.advance()
        self.act_manager.advance()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        # State initialization via event manager
        if hasattr(self, 'event_manager') and self.event_manager is not None:
            if "reset" in self.event_manager.available_modes:
                self.event_manager.apply(mode="reset", env_ids=env_ids)

        self.termination_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        self.act_manager.reset(env_ids)
        self.obs_manager.reset(env_ids)
        self.contact_manager.reset(env_ids)
        self.reward_manager.reset(env_ids)

        # Reset episode sums
        keys = list(self.episode_sums.keys())
        for key in keys:
            self.episode_sums[key].index_fill_(0, env_ids, 0.0)

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Reset all environments."""
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_env_ids)
        self.obs_manager.advance()

        self.extras = {
            "time_outs": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "terminal_observations": None,
            "terminal_env_ids": None,
            "rewards_per_type": self.rew_buf_per_type,
        }

        return self.obs_manager.get_observation(), self.extras

    def get_robot_state(self) -> torch.Tensor | None:
        """Get current robot state."""
        return self.obs_manager.get_robot_state()

    def get_observation_dims(self) -> dict[str, int]:
        """Get observation dimensions."""
        return self.obs_manager.calculate_obs_dim()

    def __str__(self) -> str:
        """Pretty print environment summary with all manager information."""
        from rlworld.rl.utils.pretty import (
            create_env_panel, panel_to_string, get_console
        )
        from io import StringIO
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=100)

        # Environment header panel
        env_rows = [
            ("Simulator", self.sim_name),
            ("Seed", str(self.seed)),
            ("Num Envs", str(self.num_envs)),
            ("Device", str(self.device)),
            ("Physics dt", f"{self.physics_dt:.4f}s"),
            ("Control dt", f"{self.control_dt:.4f}s"),
        ]

        if hasattr(self, 'decimation'):
            env_rows.append(("Decimation", str(self.decimation)))

        if hasattr(self, 'task_name') and self.task_name:
            env_rows.append(("Task", str(self.task_name)))

        panel = create_env_panel(
            title=f"{self.sim_name} Environment",
            rows=env_rows,
            border_style="blue",
        )
        console.print(panel)
        console.print()

        # Print each manager
        managers = [
            ("obs_manager", "Observation Manager"),
            ("act_manager", "Action Manager"),
            ("reward_manager", "Reward Manager"),
            ("termination_manager", "Termination Manager"),
            ("contact_manager", "Contact Manager"),
            ("command_manager", "Command Manager"),
            ("event_manager", "Event Manager"),
        ]

        for attr_name, _ in managers:
            if hasattr(self, attr_name):
                manager = getattr(self, attr_name)
                if manager is not None:
                    try:
                        manager_str = str(manager)
                        if manager_str.strip():
                            console.print(manager_str)
                    except Exception:
                        pass

        return output.getvalue()
