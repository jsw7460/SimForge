from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.base_config import iter_terms
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.mdp.configs import TerminationTermConfig, TerminationResult

if TYPE_CHECKING:
    from rlworld.rl.envs import World
    from rlworld.rl.configs.common_config_classes import TerminationsConfig

# Backward-compatible alias
TerminationConfig = None  # will be cleaned up later


class TerminationManager(BaseManager):
    """Manages termination conditions for the environment.

    Terms are discovered via :func:`iter_terms` on the config instance.
    """

    def __init__(self, env: "World", config: "TerminationsConfig", episode_length_s: float):
        super().__init__(env=env)
        self.config = config
        self._episode_length_s = episode_length_s

        # Discover named terms
        self._all_terms: dict[str, TerminationTermConfig] = iter_terms(config, TerminationTermConfig)
        self._resolved_fns: dict[str, callable] = {
            name: term.resolved_func for name, term in self._all_terms.items()
        }

        self.reset_buf = torch.ones(env.num_envs, device=self.device, dtype=torch.bool)
        self.episode_count = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(env.num_envs, device=self.device, dtype=torch.long)

        self.extras = {}

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self._episode_length_s / self.env.control_dt)

    def advance(self) -> None:
        self.episode_length_buf += 1

    def check_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        for name, term_config in self._all_terms.items():
            result: TerminationResult = self._resolved_fns[name](self.env, **term_config.params)

            if result.is_timeout:
                truncated |= result.reset
            else:
                terminated |= result.reset

            if result.extras:
                self.extras.update(result.extras)

        self.reset_buf = terminated | truncated
        return terminated, truncated

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            return
        self.episode_count[env_ids] += 1
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True

    def __str__(self) -> str:
        """Pretty print termination manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._all_terms:
            return ""

        rows = []
        for idx, (name, term) in enumerate(self._all_terms.items()):
            func_name = getattr(self._resolved_fns[name], '__name__', name)

            if "timeout" in func_name.lower() or "time_out" in func_name.lower():
                type_str = "Truncation (timeout)"
            else:
                type_str = "Termination"

            params_str = "-"
            if term.params:
                param_items = [f"{k}={v}" for k, v in list(term.params.items())[:2]]
                params_str = ", ".join(param_items)

            rows.append([idx, func_name, type_str, params_str])

        table = create_manager_table(
            title="Termination Criteria",
            columns=["Idx", "Name", "Type", "Params"],
            rows=rows,
            footer=f"Max Episode: {self.max_episode_length} steps ({self._episode_length_s}s)"
        )
        return table_to_string(table)
