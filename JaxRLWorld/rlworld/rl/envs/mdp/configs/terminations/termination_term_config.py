from dataclasses import dataclass, field
import torch
from typing import Callable


@dataclass
class TerminationResult:
    """Result from a termination check."""
    reset: torch.Tensor  # Which envs to reset
    is_timeout: bool = False  # Whether this is a timeout termination
    extras: dict = None  # Additional logging info


@dataclass
class TerminationTermConfig:
    """Configuration for a reward term."""

    func: Callable[..., TerminationResult]
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
    """

    params: dict = field(default_factory=dict)
