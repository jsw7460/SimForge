from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CommandTermConfig:
    func: Callable
    params: dict = field(default_factory=dict)
    scale: float = 1.0
