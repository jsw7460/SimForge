"""JAX-native World base class for Newton environments.

Replaces torch buffers and operations with JAX equivalents.
All step() outputs are JAX arrays — no torch→jax conversion needed in runners.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from rlworld.rl.envs.utils import NumStepCallsObserver


class JaxWorld(ABC, NumStepCallsObserver):
    """Abstract base class for JAX-native RL environments."""

    sim_name: str = "JaxWorld"

    # Required attributes (set in subclass __init__)
    num_envs: int
    device: Any  # kept for compatibility, but data is JAX
    seed: int

    # Timing
    physics_dt: float
    decimation: int
    control_dt: float

    # Managers
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

    def _init_buffers(self) -> None:
        """Initialize common buffers as JAX arrays."""
        self.rew_buf = jnp.zeros(self.num_envs)
        self.episode_sums: dict[str, jax.Array] = {}
        self.rew_buf_per_type: dict[str, jax.Array] = {}
        self.extras = {}

    def _invalidate_cache(self) -> None:
        self._cache_generation += 1

    def _update_num_step_calls(self) -> None:
        self._env_step_counter += 1
        NumStepCallsObserver.on_env_step_counter_update(self._env_step_counter)

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
    def reset_buf(self) -> jax.Array:
        return self.termination_manager.reset_buf

    @property
    def episode_length_buf(self) -> jax.Array:
        return self.termination_manager.episode_length_buf

    @property
    @abstractmethod
    def robot(self) -> Any:
        pass

    @property
    @abstractmethod
    def heading_w(self) -> jax.Array:
        pass

    def calculate_obs_dim(self) -> dict[str, int]:
        return self.obs_manager.calculate_obs_dim()

    @property
    def action_space(self) -> spaces.Box:
        num_actions = self.act_manager.total_action_dim
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

    # ========== Abstract Methods ==========

    @abstractmethod
    def _setup_environment(self) -> None:
        pass

    @abstractmethod
    def _step_physics(self) -> None:
        pass

    # ========== Common Implementation ==========

    def get_observation(self):
        return self.obs_manager.get_observation()

    def step(self, actions: jax.Array) -> Tuple[
        Dict[str, jax.Array], jax.Array, jax.Array, jax.Array, Dict[str, Any]]:
        """Execute one environment step. All outputs are JAX arrays."""
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

        # Compute rewards (JAX: immutable, so set_rewards returns new buffer)
        self.rew_buf = jnp.zeros(self.num_envs)
        self.rew_buf, self.episode_sums = self.reward_manager.set_rewards(
            reward_buffer=self.rew_buf,
            episode_sums=self.episode_sums,
            reward_buffer_per_type=self.rew_buf_per_type
        )

        # Pre-termination hook
        self._pre_termination_hook()

        # Check termination
        terminated, truncated = self.termination_manager.check_termination()
        reset_buf = terminated | truncated
        reset_env_ids = jnp.where(reset_buf)[0]

        # Handle terminal observations
        final_observation = None
        final_info = None
        if len(reset_env_ids) > 0:
            self.obs_manager.process_observations(update_history=True)
            final_observation = {
                key: jnp.array(obs) for key, obs in self.obs_manager.obs_dict.items()
            }
            final_info = {
                "episode_reward_sums": deepcopy(self.episode_sums),
            }

        # Update commands
        self.command_manager.update_commands(self.episode_length_buf)

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

    def _apply_actions(self, processed_actions: jax.Array) -> None:
        if hasattr(self.act_manager, 'apply_actions'):
            self.act_manager.apply_actions(processed_actions)
        elif hasattr(self.act_manager, 'apply_dofs_position'):
            self.act_manager.apply_dofs_position(processed_actions)

    def _pre_termination_hook(self) -> None:
        pass

    def _advance_managers(self) -> None:
        self.obs_manager.advance()
        self.reward_manager.advance()
        self.termination_manager.advance()
        self.act_manager.advance()

    def _reset_idx(self, env_ids) -> None:
        if len(env_ids) == 0:
            return

        # State initialization via event manager
        if hasattr(self, 'event_manager') and self.event_manager is not None:
            if "reset" in self.event_manager.available_modes:
                self.event_manager.apply(mode="reset", env_ids=env_ids)

        self.termination_manager.reset(env_ids)
        self.command_manager.resample_commands(env_ids)
        self.act_manager.reset(env_ids)
        self.obs_manager.reset(env_ids)
        self.contact_manager.reset(env_ids)
        self.reward_manager.reset(env_ids)

        # Reset episode sums (JAX immutable)
        for key in list(self.episode_sums.keys()):
            self.episode_sums[key] = self.episode_sums[key].at[env_ids].set(0.0)

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, jax.Array], Dict[str, Any]]:
        """Reset all environments."""
        all_env_ids = jnp.arange(self.num_envs)
        self._reset_idx(all_env_ids)
        self.obs_manager.advance()

        self.extras = {
            "time_outs": jnp.zeros(self.num_envs, dtype=jnp.bool_),
            "terminal_observations": None,
            "terminal_env_ids": None,
            "rewards_per_type": self.rew_buf_per_type,
        }

        return self.obs_manager.get_observation(), self.extras

    def get_robot_state(self) -> jax.Array | None:
        return self.obs_manager.get_robot_state()

    def get_observation_dims(self) -> dict[str, int]:
        return self.obs_manager.calculate_obs_dim()

    @property
    def is_jax_native(self) -> bool:
        """Flag for runners to detect JAX-native environments."""
        return True

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import (
            create_env_panel, panel_to_string, get_console
        )
        from io import StringIO
        from rich.console import Console

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=100)

        env_rows = [
            ("Simulator", f"{self.sim_name} (JAX-native)"),
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
            title=f"{self.sim_name} Environment (JAX-native)",
            rows=env_rows,
            border_style="blue",
        )
        console.print(panel)
        console.print()

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