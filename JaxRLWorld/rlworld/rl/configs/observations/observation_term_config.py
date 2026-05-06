from dataclasses import dataclass, field
from typing import Any, Callable

from rlworld.rl.configs.observations.noise import NoiseConfig
from rlworld.rl.utils.resolve import resolve_callable


@dataclass
class ObservationTermConfig:
    """Configuration for an observation term.

    ``func`` accepts a callable or a ``"module.path:attr_name"`` string.
    In presets, use callables directly for IDE support.
    Strings are used after YAML deserialization.
    """

    func: Callable | str
    history_length: int = 0
    flatten_history_dim: bool = True
    clip: tuple[float, float] | None = None
    scale: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)
    noise: NoiseConfig | None = None

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        return resolve_callable(self.func)
