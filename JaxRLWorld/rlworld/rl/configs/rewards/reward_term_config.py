from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import torch


class WeightSchedule(ABC):
    """Base class for reward weight scheduling."""

    @abstractmethod
    def __call__(self, step: int) -> float:
        pass


class ConstantWeight(WeightSchedule):
    """Constant weight throughout training."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, step: int) -> float:
        return self.value


class LinearSchedule(WeightSchedule):
    """Linear interpolation from initial to final value."""

    def __init__(self, initial: float, final: float, total_steps: int):
        self.initial = initial
        self.final = final
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        ratio = min(step / self.total_steps, 1.0)
        return self.initial + (self.final - self.initial) * ratio


class ExponentialDecay(WeightSchedule):
    """Exponential decay: initial * (decay_rate ** step)."""

    def __init__(self, initial: float, decay_rate: float, min_value: float = 0.0):
        self.initial = initial
        self.decay_rate = decay_rate
        self.min_value = min_value

    def __call__(self, step: int) -> float:
        return max(self.initial * (self.decay_rate ** step), self.min_value)


class StepSchedule(WeightSchedule):
    """Step-wise schedule with predefined milestones."""

    def __init__(self, milestones: dict[int, float], default: float = 0.0):
        self.milestones = sorted(milestones.items())
        self.default = default

    def __call__(self, step: int) -> float:
        value = self.default
        for milestone_step, milestone_value in self.milestones:
            if step >= milestone_step:
                value = milestone_value
            else:
                break
        return value


def get_weight_value(weight: float | WeightSchedule, step: int) -> float:
    """Utility function to resolve weight value."""
    if isinstance(weight, WeightSchedule):
        return weight(step)
    return weight


@dataclass
class RewardTermConfig:
    """Configuration for a reward term.

    ``func`` is a string reference in ``"module.path:attr_name"`` format.
    """

    func: Callable | str
    weight: float | WeightSchedule = 0.0
    params: dict = field(default_factory=dict)
    exp_shaping: bool = False
    """When reward_mode is 'exponential', this term goes inside exp().
    When reward_mode is 'exponential_auto', this field is ignored."""

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        from rlworld.rl.utils.resolve import resolve_callable
        return resolve_callable(self.func)