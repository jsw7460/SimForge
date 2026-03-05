"""JAX-native observation manager for Newton environments."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.storages import CircularBuffer

if TYPE_CHECKING:
    from rlworld.rl.envs import World

from dataclasses import dataclass


@dataclass
class ObsManagerConfig:
    num_envs: int
    obs_group: dict[str, list[ObservationTermConfig]]


class JaxObservationManager(BaseManager):
    """JAX-native observation manager."""

    def __init__(self, env: "World", config: ObsManagerConfig):
        BaseManager.__init__(self, env=env)
        self.config = config
        self.obs_dict = {}
        self.extras = {}
        self._group_obs_term_history_buffer: dict[str, dict[str, CircularBuffer]] = {}
        self._group_term_indices: dict[str, dict[str, tuple[int, int]]] = {}
        self._is_term_indices_built = False
        self._initialize_history_buffers()

    def _initialize_history_buffers(self) -> None:
        for group_name, terms in self.config.obs_group.items():
            self._group_obs_term_history_buffer[group_name] = {}
            for term_idx, obs_term in enumerate(terms):
                if obs_term.history_length > 0:
                    term_name = getattr(obs_term, 'name', f"term_{term_idx}")
                    self._group_obs_term_history_buffer[group_name][term_name] = CircularBuffer(
                        max_len=obs_term.history_length,
                        batch_size=self.config.num_envs,
                        device=self.env.device
                    )

    def _build_term_indices(self) -> None:
        for group_name, terms in self.config.obs_group.items():
            self._group_term_indices[group_name] = {}
            current_idx = 0
            for term_idx, obs_term in enumerate(terms):
                term_name = getattr(obs_term.func, '__name__', f"term_{term_idx}")
                dummy_value = obs_term.func(self.env, **obs_term.params)
                base_dim = dummy_value.shape[-1]
                history_length = getattr(obs_term, 'history_length', 0)
                flatten_history = getattr(obs_term, 'flatten_history_dim', True)
                if history_length > 0 and flatten_history:
                    term_dim = base_dim * history_length
                else:
                    term_dim = base_dim
                self._group_term_indices[group_name][term_name] = (current_idx, current_idx + term_dim)
                current_idx += term_dim

    def calculate_obs_dim(self) -> dict[str, int]:
        if not self._is_term_indices_built:
            self._build_term_indices()
        self.process_observations(update_history=False)
        return defaultdict(int, {group: tensor.shape[-1] for group, tensor in self.obs_dict.items()})

    def get_observation(self) -> dict[str, jax.Array]:
        return self.obs_dict

    def get_robot_state(self) -> jax.Array | None:
        return self.extras.get("robot_state", None)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            import torch
            env_ids = torch.arange(self.config.num_envs, device=self.env.device)
        for group_name, history_buffers in self._group_obs_term_history_buffer.items():
            for term_name, buffer in history_buffers.items():
                buffer.reset(batch_ids=env_ids)

    def extract_term(self, group_name: str, term_name: str, observations=None) -> jax.Array:
        if observations is None:
            observations = self.obs_dict[group_name]
        start_idx, end_idx = self._group_term_indices[group_name][term_name]
        return observations[:, start_idx:end_idx]

    def get_raw_term(self, term_func: callable, **params) -> jax.Array:
        return term_func(self.env, **params)

    def process_observations(self, update_history: bool = True) -> None:
        self.obs_dict = {}

        for group_name, terms in self.config.obs_group.items():
            obs_list = []

            for term_idx, obs_term in enumerate(terms):
                func = obs_term.func
                scale = obs_term.scale

                obs_value = func(self.env, **obs_term.params)

                # Apply noise if configured
                if obs_term.noise is not None:
                    from rlworld.rl.configs.observations.noise import apply_noise
                    obs_value = apply_noise(obs_value, obs_term.noise)

                # Clip
                if obs_term.clip is not None:
                    obs_value = jnp.clip(obs_value, obs_term.clip[0], obs_term.clip[1])

                obs_value = obs_value * scale

                # Handle history
                term_name = getattr(obs_term, 'name', f"term_{term_idx}")
                history_length = getattr(obs_term, 'history_length', 0)

                if history_length > 0:
                    circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                    if update_history:
                        circular_buffer.append(obs_value)
                    elif circular_buffer._buffer is None:
                        circular_buffer.append(obs_value)
                    flatten_history = getattr(obs_term, 'flatten_history_dim', True)
                    if flatten_history:
                        obs_with_history = circular_buffer.buffer.reshape(self.config.num_envs, -1)
                    else:
                        obs_with_history = circular_buffer.buffer
                    obs_list.append(obs_with_history)
                else:
                    obs_list.append(obs_value)

            self.obs_dict[group_name] = jnp.concatenate(obs_list, axis=-1)

    def advance(self) -> None:
        self.process_observations(update_history=True)

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string, format_shape

        output_parts = []
        for group_name, terms in self.config.obs_group.items():
            rows = []
            total_dim = 0
            for idx, obs_term in enumerate(terms):
                func_name = getattr(obs_term.func, '__name__', f"term_{idx}")
                try:
                    dummy = obs_term.func(self.env, **obs_term.params)
                    base_dim = dummy.shape[-1]
                except Exception:
                    base_dim = "?"
                history_str = "-"
                display_dim = base_dim
                if obs_term.history_length > 0:
                    mode = "flatten" if obs_term.flatten_history_dim else "stack"
                    history_str = f"{obs_term.history_length} ({mode})"
                    if obs_term.flatten_history_dim and isinstance(base_dim, int):
                        display_dim = base_dim * obs_term.history_length
                if isinstance(display_dim, int):
                    total_dim += display_dim
                scale_str = f"{obs_term.scale}" if obs_term.scale != 1.0 else "1.0"
                noise_str = "-"
                if obs_term.noise is not None:
                    noise_str = type(obs_term.noise).__name__
                rows.append([idx, func_name, format_shape(base_dim), scale_str, history_str, noise_str])
            table = create_manager_table(
                title=f"Observation Space ({group_name})",
                columns=["Idx", "Name", "Shape", "Scale", "History", "Noise"],
                rows=rows,
                footer=f"Total: {total_dim} dims" if isinstance(total_dim, int) else None
            )
            output_parts.append(table_to_string(table))
        return "\n".join(output_parts)
