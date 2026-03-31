from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from rlworld.rl.utils.resolve import resolve_callable


@dataclass
class EventTermConfig:
    """``func`` accepts a callable or ``"module.path:attr_name"`` string."""
    func: Callable | str
    mode: Literal["startup", "reset", "interval"]
    params: dict[str, Any] = field(default_factory=dict)
    interval_range_s: tuple[float, float] | None = None  # for interval mode

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        return resolve_callable(self.func)