from dataclasses import dataclass, field
from enum import Enum
from typing import List


class MetricType(Enum):
    """Type of metric for console display styling."""
    LOSS = "loss"
    ENTROPY = "entropy"
    COEFFICIENT = "coef"
    VALUE = "value"
    RATE = "rate"


@dataclass
class MetricSpec:
    """Specification for a single metric."""
    key: str
    label: str
    type: MetricType = MetricType.LOSS


@dataclass
class ConsoleLoggingSpec:
    """Specification for console logging output."""
    metrics: List[MetricSpec] = field(default_factory=list)

    def add(
        self,
        key: str,
        label: str,
        type: MetricType = MetricType.LOSS,
    ) -> "ConsoleLoggingSpec":
        """Add a metric specification. Returns self for chaining."""
        self.metrics.append(MetricSpec(key, label, type))
        return self