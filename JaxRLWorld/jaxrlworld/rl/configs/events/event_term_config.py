from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from jaxrlworld.rl.utils.resolve import resolve_callable


@dataclass
class EventTermConfig:
    """``func`` accepts a callable or ``"module.path:attr_name"`` string."""

    func: Callable | str
    mode: Literal["startup", "reset", "reset_dr", "interval", "interval_dr"]
    params: dict[str, Any] = field(default_factory=dict)
    interval_range_s: tuple[float, float] | None = None  # for interval mode
    # For interval_dr mode: all interval_dr terms fire together on ONE global
    # timer every ``interval_dr_period_s`` seconds (all envs at once, single
    # deferred recompute). Unlike ``interval`` (per-env async timers), this
    # keeps the recompute cost to one flush per period instead of per reset.
    interval_dr_period_s: float | None = None

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        return resolve_callable(self.func)
