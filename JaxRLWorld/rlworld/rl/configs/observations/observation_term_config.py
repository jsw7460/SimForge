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

    # Stochastic observation delay (mjlab-compatible field set). The
    # processing order matches mjlab: compute -> noise -> clip -> scale
    # -> delay -> history. Lags are in CONTROL steps.
    delay_min_lag: int = 0
    """Minimum lag for delayed observations. Lag is sampled uniformly
    from ``[delay_min_lag, delay_max_lag]``; use min == max for a
    constant delay."""
    delay_max_lag: int = 0
    """Maximum lag for delayed observations. 0 disables the delay
    stage entirely."""
    delay_per_env: bool = True
    """Sample an independent lag per environment; ``False`` shares one
    sampled lag across the batch."""
    delay_hold_prob: float = 0.0
    """Probability of keeping the previous lag when a resample is due
    (temporal correlation in the delay pattern)."""
    delay_update_period: int = 0
    """Resample lags every N control steps per env; 0 resamples every
    step."""
    delay_per_env_phase: bool = True
    """With ``delay_update_period > 0``, stagger the resample steps
    across envs via a random per-env phase offset."""

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        return resolve_callable(self.func)
