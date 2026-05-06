from dataclasses import dataclass, field
from typing import Callable

from rlworld.rl.utils.resolve import resolve_callable


@dataclass
class CommandTermConfig:
    """``func`` accepts a callable or ``"module.path:attr_name"`` string."""

    func: Callable | str
    params: dict = field(default_factory=dict)
    scale: float = 1.0

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        return resolve_callable(self.func)
