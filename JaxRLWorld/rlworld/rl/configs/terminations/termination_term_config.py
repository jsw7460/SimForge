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
    """

    func: Callable | str
    params: dict = field(default_factory=dict)

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        return resolve_callable(self.func)
