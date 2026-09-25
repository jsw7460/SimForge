from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


class WeightSchedule(ABC):
    """Base class for reward weight scheduling.

    A schedule is a plain object, so config serialization writes it as
    ``{"_type": <class name>, **constructor kwargs}`` (:meth:`to_dict`) and
    :func:`weight_schedule_from_dict` rebuilds it; each subclass's
    :meth:`to_dict` returns exactly its ``__init__`` keyword arguments.
    """

    @abstractmethod
    def __call__(self, step: int) -> float:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


class ConstantWeight(WeightSchedule):
    """Constant weight throughout training."""

    def __init__(self, value: float):
        self.value = value

    def __call__(self, step: int) -> float:
        return self.value

    def to_dict(self) -> dict:
        return {"_type": type(self).__name__, "value": self.value}


class LinearSchedule(WeightSchedule):
    """Linear interpolation from initial to final value."""

    def __init__(self, initial: float, final: float, total_steps: int):
        self.initial = initial
        self.final = final
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        ratio = min(step / self.total_steps, 1.0)
        return self.initial + (self.final - self.initial) * ratio

    def to_dict(self) -> dict:
        return {
            "_type": type(self).__name__,
            "initial": self.initial,
            "final": self.final,
            "total_steps": self.total_steps,
        }


class ExponentialDecay(WeightSchedule):
    """Exponential decay: initial * (decay_rate ** step)."""

    def __init__(self, initial: float, decay_rate: float, min_value: float = 0.0):
        self.initial = initial
        self.decay_rate = decay_rate
        self.min_value = min_value

    def __call__(self, step: int) -> float:
        return max(self.initial * (self.decay_rate**step), self.min_value)

    def to_dict(self) -> dict:
        return {
            "_type": type(self).__name__,
            "initial": self.initial,
            "decay_rate": self.decay_rate,
            "min_value": self.min_value,
        }


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

    def to_dict(self) -> dict:
        return {"_type": type(self).__name__, "milestones": dict(self.milestones), "default": self.default}


_SCHEDULE_TYPES: dict[str, type[WeightSchedule]] = {
    cls.__name__: cls for cls in (ConstantWeight, LinearSchedule, ExponentialDecay, StepSchedule)
}


def weight_schedule_from_dict(data: dict) -> WeightSchedule:
    """Rebuild a schedule from its :meth:`WeightSchedule.to_dict` form."""
    kwargs = dict(data)
    type_name = kwargs.pop("_type")
    if type_name not in _SCHEDULE_TYPES:
        raise ValueError(f"Unknown weight schedule type {type_name!r}; known: {sorted(_SCHEDULE_TYPES)}")
    return _SCHEDULE_TYPES[type_name](**kwargs)


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

    def __post_init__(self) -> None:
        # A serialized schedule (config restore, YAML) arrives as a dict.
        if isinstance(self.weight, dict):
            self.weight = weight_schedule_from_dict(self.weight)

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        from jaxrlworld.rl.utils.resolve import resolve_callable

        return resolve_callable(self.func)
