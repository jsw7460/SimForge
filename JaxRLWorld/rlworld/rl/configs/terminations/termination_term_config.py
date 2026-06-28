from dataclasses import dataclass, field
from typing import Callable

import torch

from rlworld.rl.utils.resolve import resolve_callable


@dataclass
class TerminationResult:
    """Result from a termination check."""

    reset: torch.Tensor  # Which envs to reset
    is_timeout: bool = False  # Whether this is a timeout termination
    extras: dict = None  # Additional logging info


@dataclass
class TerminationTermConfig:
    """Configuration for a termination term.

    ``func`` is a ``"module.path:attr_name"`` string reference.

    ``bootstrap_value`` declares whether this (non-timeout) termination is a
    *non-absorbing* early-stop whose terminal value should still be
    bootstrapped — e.g. reaching a goal in a dense-reward task, where the agent
    would keep earning if it continued. Default ``False`` = absorbing failure
    (fall / unrecoverable), so the terminal value is treated as 0 (no
    bootstrap), which is the correct and historical behaviour for locomotion.
    Ignored for timeout terms (they are truncations and always bootstrap).
    """

    func: Callable | str
    params: dict = field(default_factory=dict)
    bootstrap_value: bool = False

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        return resolve_callable(self.func)
