from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Union


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
