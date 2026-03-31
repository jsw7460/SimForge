from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.base_config import iter_terms
from rlworld.rl.configs.common_config_classes import ObservationGroupConfig
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.storages import CircularBuffer
from rlworld.rl.configs.observations.noise import apply_noise

if TYPE_CHECKING:
    from rlworld.rl.envs import World

# Backward-compatible alias
ObsManagerConfig = None  # deprecated


class ObservationManager(BaseManager):
    """Manages observation generation and processing for RL environments.

    Uses the IsaacLab named-attribute pattern:
    - Groups are discovered as ``ObservationGroupConfig`` attributes on the config.
    - Terms are discovered as ``ObservationTermConfig`` attributes on each group.
    """

    def __init__(self, env: "World", config):
        BaseManager.__init__(self, env=env)
        self.config = config

        # Observation buffers (populated during runtime)
        self.obs_dict = {}
        self.extras = {}

        # Discover groups: any attribute that is an ObservationGroupConfig
        self._groups: dict[str, ObservationGroupConfig] = {}
        for attr_name in dir(config):
            if attr_name.startswith("_"):
                continue
            val = getattr(config, attr_name)
            if isinstance(val, ObservationGroupConfig):
                self._groups[attr_name] = val

        # Discover terms within each group and resolve callables
        self._group_terms: dict[str, dict[str, ObservationTermConfig]] = {}
        self._resolved_fns: dict[str, dict[str, callable]] = {}
        for group_name, group_cfg in self._groups.items():
            terms = iter_terms(group_cfg, ObservationTermConfig)
            self._group_terms[group_name] = terms
            self._resolved_fns[group_name] = {
                name: t.resolved_func for name, t in terms.items()
            }

        # History buffers
        self._group_obs_term_history_buffer: dict[str, dict[str, CircularBuffer]] = {}
        self._initialize_history_buffers()

        # Term index mapping for extraction
        self._group_term_indices: dict[str, dict[str, tuple[int, int]]] = {}
        self._is_term_indices_built = False

    @property
    def enable_noise(self) -> bool:
        return getattr(self.config, "enable_noise", True)

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    # ========== Initialization ==========

    def _initialize_history_buffers(self) -> None:
        for group_name, terms in self._group_terms.items():
            self._group_obs_term_history_buffer[group_name] = {}
            for term_name, obs_term in terms.items():
                if obs_term.history_length > 0:
                    self._group_obs_term_history_buffer[group_name][term_name] = CircularBuffer(
                        max_len=obs_term.history_length,
                        batch_size=self.num_envs,
                        device=self.env.device,
                    )

    def _build_term_indices(self) -> None:
        for group_name, terms in self._group_terms.items():
            self._group_term_indices[group_name] = {}
            current_idx = 0
            for term_name, obs_term in terms.items():
                resolved_fn = self._resolved_fns[group_name][term_name]
                dummy_value = resolved_fn(self.env, **obs_term.params)
                base_dim = dummy_value.shape[-1]

                history_length = obs_term.history_length
                flatten_history = obs_term.flatten_history_dim

                if history_length > 0 and flatten_history:
                    term_dim = base_dim * history_length
                else:
                    term_dim = base_dim

                self._group_term_indices[group_name][term_name] = (current_idx, current_idx + term_dim)
                current_idx += term_dim

    # ========== Public API ==========

    def calculate_obs_dim(self) -> dict[str, int]:
        if not self._is_term_indices_built:
            self._build_term_indices()
        self.process_observations(update_history=False)
        return defaultdict(int, {group: tensor.shape[-1] for group, tensor in self.obs_dict.items()})

    def get_observation(self) -> dict[str, torch.Tensor]:
        return self.obs_dict

    def get_robot_state(self) -> torch.Tensor | None:
        return self.extras.get("robot_state", None)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.env.device)
        for group_name, history_buffers in self._group_obs_term_history_buffer.items():
            for term_name, buffer in history_buffers.items():
                buffer.reset(batch_ids=env_ids)

    def extract_term(
        self,
        group_name: str,
        term_name: str,
        observations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if observations is None:
            observations = self.obs_dict[group_name]
        with torch.no_grad():
            start_idx, end_idx = self._group_term_indices[group_name][term_name]
            result = observations[:, start_idx:end_idx]
        return result

    def get_raw_term(self, term_func: callable, **params) -> torch.Tensor:
        return term_func(self.env, **params)

    # ========== Core Processing ==========

    def process_observations(self, update_history: bool = True) -> None:
        self.obs_dict = {}

        for group_name, terms in self._group_terms.items():
            obs_list = []

            for term_name, obs_term in terms.items():
                func = self._resolved_fns[group_name][term_name]
                scale = obs_term.scale

                obs_value = func(self.env, **obs_term.params)

                if self.enable_noise and obs_term.noise is not None:
                    obs_value = apply_noise(obs_value, obs_term.noise)

                if obs_term.clip is not None:
                    clip = obs_term.clip
                    if isinstance(clip, list):
                        clip = tuple(clip)
                    obs_value = obs_value.clip_(min=clip[0], max=clip[1])

                obs_value = obs_value * scale

                # Handle history
                history_length = obs_term.history_length
                if history_length > 0:
                    circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]

                    if update_history:
                        circular_buffer.append(obs_value)
                    elif circular_buffer._buffer is None:
                        circular_buffer.append(obs_value)

                    flatten_history = obs_term.flatten_history_dim
                    if flatten_history:
                        obs_with_history = circular_buffer.buffer.reshape(self.num_envs, -1)
                    else:
                        obs_with_history = circular_buffer.buffer

                    obs_list.append(obs_with_history)
                else:
                    obs_list.append(obs_value)

            self.obs_dict[group_name] = torch.concat(obs_list, dim=-1)

    def advance(self) -> None:
        self.process_observations(update_history=True)

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string, format_shape

        output_parts = []

        for group_name, terms in self._group_terms.items():
            rows = []
            total_dim = 0

            for idx, (term_name, obs_term) in enumerate(terms.items()):
                resolved_fn = self._resolved_fns[group_name][term_name]
                func_name = getattr(resolved_fn, '__name__', term_name)

                try:
                    dummy = resolved_fn(self.env, **obs_term.params)
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
                    if self.enable_noise:
                        noise_str = type(obs_term.noise).__name__
                    else:
                        noise_str = f"{type(obs_term.noise).__name__} (off)"

                rows.append([idx, func_name, format_shape(base_dim), scale_str, history_str, noise_str])

            table = create_manager_table(
                title=f"Observation Space ({group_name})",
                columns=["Idx", "Name", "Shape", "Scale", "History", "Noise"],
                rows=rows,
                footer=f"Total: {total_dim} dims" if isinstance(total_dim, int) else None,
            )
            output_parts.append(table_to_string(table))

        return "\n".join(output_parts)
