from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.managers.common.command_term import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class CommandManagerConfig:
    """Configuration for CommandManager."""
    terms: dict[str, CommandTermCfg] = field(default_factory=dict)


class CommandManager(BaseManager):
    """Manages command generation via pluggable CommandTerm objects.

    Each term independently samples, resamples on its own timer,
    and applies per-step post-processing.

    Access patterns::

        command_manager.get_command("velocity")   # → [num_envs, 3]
        command_manager.get_term("velocity")      # → VelocityCommandTerm
        command_manager.lin_vel_x                 # → [num_envs] (shortcut)
        command_manager.get_commands_tensor()      # → all terms concatenated
    """

    def __init__(self, env: "World", config: CommandManagerConfig):
        super().__init__(env)
        self.config = config

        self._terms: dict[str, CommandTerm] = {}

        # Maps column name → (term_name, column_index).
        #
        # Each CommandTerm declares column_names for its command dimensions.
        # For example, VelocityCommandTerm.column_names = ("lin_vel_x", "lin_vel_y", "ang_vel")
        # so _column_map becomes:
        #   {"lin_vel_x": ("velocity", 0), "lin_vel_y": ("velocity", 1), "ang_vel": ("velocity", 2)}
        #
        # This allows `command_manager.lin_vel_x` to resolve to
        # `_terms["velocity"].command[:, 0]` via __getattr__.
        self._column_map: dict[str, tuple[str, int]] = {}

        for name, term_cfg in config.terms.items():
            term = term_cfg.build(env)
            self._terms[name] = term
            for col_idx, col_name in enumerate(term.column_names):
                self._column_map[col_name] = (name, col_idx)

    def get_command(self, name: str) -> torch.Tensor:
        """Get full command tensor for a named term."""
        return self._terms[name].command

    def get_term(self, name: str) -> CommandTerm:
        """Get the CommandTerm object."""
        return self._terms[name]

    def get_commands_tensor(self) -> torch.Tensor:
        """Concatenate all term commands into a single tensor.

        Returns:
            [num_envs, total_command_dim] with terms in insertion order.
        """
        if not self._terms:
            return torch.zeros(self.env.num_envs, 0, device=self.device)
        return torch.cat(
            [term.command for term in self._terms.values()], dim=1
        )

    @property
    def num_commands(self) -> int:
        return sum(term.command.shape[1] for term in self._terms.values())

    def compute(self, dt: float) -> None:
        """Advance all terms (timer-based resampling + post-processing)."""
        for term in self._terms.values():
            term.compute(dt)

    def set_commands(
        self, env_ids: torch.Tensor, **kwargs: torch.Tensor
    ) -> None:
        """Override commands by term name for the given environments.

        Disables auto-resampling for the injected envs until
        :meth:`release_commands` or episode reset. Useful for
        teleoperation or external controllers::

            env.command_manager.set_commands(
                env_ids=torch.tensor([0]),
                velocity=torch.tensor([[0.5, 0.0, 0.3]]),
            )

        Args:
            env_ids: Environment indices to override.
            **kwargs: ``term_name=values`` pairs. Each ``values``
                tensor has shape ``(len(env_ids), term_command_dim)``.
        """
        for name, values in kwargs.items():
            self._terms[name].set_command(env_ids, values)

    def release_commands(
        self, env_ids: torch.Tensor, *term_names: str
    ) -> None:
        """Release external control and return to auto-resampling.

        If no ``term_names`` are given, releases ALL terms for the
        specified envs. Otherwise only the listed terms are released.
        """
        targets = (
            [self._terms[n] for n in term_names]
            if term_names
            else list(self._terms.values())
        )
        for term in targets:
            term.release_command(env_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Force resample all terms for the given environments.

        Also clears any external-control state set by
        :meth:`set_commands`.
        """
        for term in self._terms.values():
            term.reset(env_ids)

    def __getattr__(self, name: str):
        """Shortcut access for command columns and term attributes.

        Enables ``command_manager.lin_vel_x`` instead of
        ``command_manager.get_command("velocity")[:, 0]``.

        Lookup order:
          1. _column_map: name matches a declared column → return that slice.
          2. Term attributes: name matches a property on one of the terms
             (e.g., is_standing_env on VelocityCommandTerm).
          3. Fall through to normal AttributeError.
        """
        # Guard: avoid infinite recursion during __init__ before _column_map exists.
        if name.startswith("_") or name in ("config", "env", "device"):
            return object.__getattribute__(self, name)

        try:
            column_map = object.__getattribute__(self, "_column_map")
        except AttributeError:
            # __init__ not finished yet — _column_map doesn't exist.
            return object.__getattribute__(self, name)

        # 1. Column lookup: "lin_vel_x" → _terms["velocity"].command[:, 0]
        if name in column_map:
            term_name, col_idx = column_map[name]
            terms = object.__getattribute__(self, "_terms")
            return terms[term_name].command[:, col_idx]

        # 2. Term attribute lookup: "is_standing_env" → VelocityCommandTerm.is_standing_env
        terms = object.__getattribute__(self, "_terms")
        for term in terms.values():
            if hasattr(term, name):
                return getattr(term, name)

        return object.__getattribute__(self, name)

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._terms:
            return ""

        rows = []
        for name, term in self._terms.items():
            dim = term.command.shape[1]
            cols = ", ".join(term.column_names) if term.column_names else "-"
            resample = (
                f"{term.cfg.resampling_time_range[0]}"
                f"-{term.cfg.resampling_time_range[1]}s"
            )
            rows.append([name, str(dim), cols, resample])

        table = create_manager_table(
            title="Command Terms",
            columns=["Term", "Dim", "Columns", "Resample"],
            rows=rows,
            footer=f"{len(self._terms)} terms, {self.num_commands} total dims",
        )
        return table_to_string(table)
