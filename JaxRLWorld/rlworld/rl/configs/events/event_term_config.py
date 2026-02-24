from dataclasses import dataclass, field
from typing import Callable, Literal, Any


@dataclass
class EventTermConfig:
    func: Callable
    mode: Literal["startup", "reset", "interval"]
    params: dict[str, Any] = field(default_factory=dict)
    interval_range_s: tuple[float, float] | None = None  # for interval mode