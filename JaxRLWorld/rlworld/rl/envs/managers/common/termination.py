from dataclasses import dataclass
from typing import TYPE_CHECKING
import math

import torch

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.mdp.configs import TerminationTermConfig, TerminationResult

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class TerminationConfig:
    """Termination manager configuration."""
    num_envs: int
    termination_criteria: list[TerminationTermConfig]
    episode_length_s: float


class TerminationManager(BaseManager):
    """Manages termination conditions for the environment."""

    def __init__(self, env: "World", config: TerminationConfig):
        super().__init__(env=env)
        self.config = config
        self.termination_criteria = config.termination_criteria

        self.reset_buf = torch.ones(config.num_envs, device=self.device, dtype=torch.bool)
        self.episode_count = torch.zeros(config.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(config.num_envs, device=self.device, dtype=torch.long)

        self.extras = {}

    @property
    def max_episode_length(self) -> int:
        return math.ceil(self.config.episode_length_s / self.env.control_dt)

    def advance(self) -> None:
        self.episode_length_buf += 1

    def check_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        for term_config in self.termination_criteria:
            result: TerminationResult = term_config.func(self.env, **term_config.params)

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

        if not self.termination_criteria:
            return ""

        rows = []
        for idx, term in enumerate(self.termination_criteria):
            func_name = getattr(term.func, '__name__', f"term_{idx}")

            # Determine type from function name or params
            if "timeout" in func_name.lower() or "time_out" in func_name.lower():
                type_str = "Truncation (timeout)"
            elif any(key in func_name.lower() for key in ["contact", "collision"]):
                type_str = "Termination"
            else:
                type_str = "Termination"

            # Format params
            params_str = "-"
            if term.params:
                param_items = [f"{k}={v}" for k, v in list(term.params.items())[:2]]
                params_str = ", ".join(param_items)

            rows.append([idx, func_name, type_str, params_str])

        table = create_manager_table(
            title="Termination Criteria",
            columns=["Idx", "Name", "Type", "Params"],
            rows=rows,
            footer=f"Max Episode: {self.max_episode_length} steps ({self.config.episode_length_s}s)"
        )
        return table_to_string(table)