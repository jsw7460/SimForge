from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def _stack_scalars(values: tuple) -> jnp.ndarray:
    """Gather scalars into one array inside a single dispatch.

    Doing the stack eagerly would trade one blocking transfer per value
    for one op dispatch per value, which on a fast link is no better.
    Under ``jit`` the whole gather is one program and one launch.
    """
    return jnp.stack([jnp.asarray(value, dtype=jnp.float32).reshape(()) for value in values])


def host_scalars(values: Dict[str, Any]) -> Dict[str, float]:
    """Materialise a whole metric set in ONE device-to-host transfer.

    ``float()`` on a device scalar is a blocking transfer: it drains the
    queue and stalls the pipeline. A metric set has a couple of dozen of
    them, so building it one ``float()`` at a time costs a couple of
    dozen stalls per update.

    That is invisible in an on-policy run, where one update covers a
    whole rollout, and dominant in an off-policy one, where an update
    follows every single environment step. PPO removed its own copy of
    this cost by fusing the metrics into one jitted program; stacking the
    scalars first achieves the same thing without touching the update.

    Cheaper still is not building the set at all on iterations that will
    not log it — callers that update far more often than they log should
    skip the call rather than optimise it.

    Plain Python floats are accepted alongside device scalars — branches
    that skip a network (a delayed policy update) pass literals, and both
    cross in the same transfer.
    """
    names = list(values)
    stacked = _stack_scalars(tuple(values[name] for name in names))
    return {name: float(value) for name, value in zip(names, np.asarray(stacked))}


class MetricType(Enum):
    """Type of metric for formatting."""

    LOSS = auto()
    VALUE = auto()
    ENTROPY = auto()
    COEFFICIENT = auto()
    RATIO = auto()
    COUNT = auto()
    STRING = auto()


@dataclass
class ConsoleMetric:
    """Single console metric with display info."""

    display_name: str
    metric_type: MetricType
    value: Union[float, str] = 0.0


@dataclass
class BaseMetrics:
    """Base class for algorithm metrics."""

    def get_console_metrics(self) -> list[ConsoleMetric]:
        """Return metrics with display info for console. Override in subclass."""
        raise NotImplementedError

    def to_wandb_dict(self) -> dict[str, float]:
        """Return all metrics for wandb (flat, with prefixes)."""
        raise NotImplementedError

    def to_full_dict(self) -> dict[str, Any]:
        """Return full dict for backward compatibility."""
        return {
            "wandb_extra": self.to_wandb_dict(),
        }


# ==================== Shared Metrics ====================


@dataclass
class ActorMetrics:
    """Common actor metrics for all algorithms."""

    loss: float = 0.0
    entropy: float = 0.0
    std: float = 0.0

    def to_wandb_dict(self, prefix: str = "actor") -> dict[str, float]:
        """Convert to wandb dict with prefix."""
        return {
            f"{prefix}/loss": self.loss,
            f"{prefix}/entropy": self.entropy,
            f"{prefix}/std": self.std,
        }


@dataclass
class BatchMetrics:
    """Common batch statistics."""

    return_mean: float = 0.0
    return_std: float = 0.0
    return_min: float = 0.0
    return_max: float = 0.0
    action_mean: float = 0.0
    action_std: float = 0.0

    def to_wandb_dict(self, prefix: str = "batch") -> dict[str, float]:
        """Convert to wandb dict with prefix."""
        return {
            f"{prefix}/return_mean": self.return_mean,
            f"{prefix}/return_std": self.return_std,
            f"{prefix}/return_min": self.return_min,
            f"{prefix}/return_max": self.return_max,
            f"{prefix}/action_mean": self.action_mean,
            f"{prefix}/action_std": self.action_std,
        }
