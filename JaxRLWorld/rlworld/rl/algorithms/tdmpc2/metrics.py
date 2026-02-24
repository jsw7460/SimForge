"""TD-MPC2 training metrics."""

from dataclasses import dataclass, field

from rlworld.rl.algorithms.metrics.base import (
    BaseMetrics,
    BatchMetrics,
    MetricType,
    ConsoleMetric,
)


@dataclass
class TDMPC2WorldModelMetrics:
    """World model training losses."""
    consistency_loss: float = 0.0
    reward_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    grad_norm: float = 0.0

    def to_wandb_dict(self, prefix: str = "world_model") -> dict[str, float]:
        return {
            f"{prefix}/consistency_loss": self.consistency_loss,
            f"{prefix}/reward_loss": self.reward_loss,
            f"{prefix}/value_loss": self.value_loss,
            f"{prefix}/total_loss": self.total_loss,
            f"{prefix}/grad_norm": self.grad_norm,
        }


@dataclass
class TDMPC2PolicyMetrics:
    """Policy update metrics."""
    pi_loss: float = 0.0
    pi_entropy: float = 0.0
    pi_scaled_entropy: float = 0.0
    pi_grad_norm: float = 0.0
    pi_scale: float = 0.0

    def to_wandb_dict(self, prefix: str = "policy") -> dict[str, float]:
        return {
            f"{prefix}/pi_loss": self.pi_loss,
            f"{prefix}/pi_entropy": self.pi_entropy,
            f"{prefix}/pi_scaled_entropy": self.pi_scaled_entropy,
            f"{prefix}/pi_grad_norm": self.pi_grad_norm,
            f"{prefix}/pi_scale": self.pi_scale,
        }


@dataclass
class TDMPC2QMetrics:
    """Q-value statistics."""
    mean: float = 0.0
    std: float = 0.0
    p05: float = 0.0
    p95: float = 0.0

    def to_wandb_dict(self, prefix: str = "q") -> dict[str, float]:
        return {
            f"{prefix}/mean": self.mean,
            f"{prefix}/std": self.std,
            f"{prefix}/p05": self.p05,
            f"{prefix}/p95": self.p95,
        }


@dataclass
class TDMPC2Metrics(BaseMetrics):
    """Complete TD-MPC2 training metrics."""
    world_model: TDMPC2WorldModelMetrics = field(default_factory=TDMPC2WorldModelMetrics)
    policy: TDMPC2PolicyMetrics = field(default_factory=TDMPC2PolicyMetrics)
    q: TDMPC2QMetrics = field(default_factory=TDMPC2QMetrics)
    batch: BatchMetrics = field(default_factory=BatchMetrics)
    total_updates: int = 0

    def get_console_metrics(self) -> list[ConsoleMetric]:
        return [
            ConsoleMetric("Name", MetricType.VALUE, "TD-MPC2"),
            ConsoleMetric("Consistency", MetricType.LOSS, self.world_model.consistency_loss),
            ConsoleMetric("Reward", MetricType.LOSS, self.world_model.reward_loss),
            ConsoleMetric("Value", MetricType.LOSS, self.world_model.value_loss),
            ConsoleMetric("Total", MetricType.LOSS, self.world_model.total_loss),
            ConsoleMetric("Pi Loss", MetricType.LOSS, self.policy.pi_loss),
            ConsoleMetric("Entropy", MetricType.ENTROPY, self.policy.pi_entropy),
            ConsoleMetric("Scaled Ent", MetricType.ENTROPY, self.policy.pi_scaled_entropy),
            ConsoleMetric("Pi Scale", MetricType.VALUE, self.policy.pi_scale),
            ConsoleMetric("Q Mean", MetricType.VALUE, self.q.mean),
            ConsoleMetric("Pi Grad", MetricType.VALUE, self.policy.pi_grad_norm),
            ConsoleMetric("WM Grad", MetricType.VALUE, self.world_model.grad_norm),
        ]

    def to_wandb_dict(self) -> dict[str, float]:
        result = {}
        result.update(self.world_model.to_wandb_dict())
        result.update(self.policy.to_wandb_dict())
        result.update(self.q.to_wandb_dict())
        result.update(self.batch.to_wandb_dict())
        result["total_updates"] = self.total_updates
        return result