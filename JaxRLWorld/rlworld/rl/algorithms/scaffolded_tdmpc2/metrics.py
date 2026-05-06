"""Scaffolded TD-MPC2 training metrics.

Follows rlworld.rl.algorithms.tdmpc2.metrics pattern:
- Dataclasses with to_wandb_dict()
- BaseMetrics with get_console_metrics(), to_full_dict()
"""

from dataclasses import dataclass, field

from rlworld.rl.algorithms.metrics.base import (
    BaseMetrics,
    BatchMetrics,
    ConsoleMetric,
    MetricType,
)


@dataclass
class ScaffoldedWorldModelMetrics:
    """Target world model training losses."""

    consistency_loss: float = 0.0
    reward_loss: float = 0.0
    value_loss: float = 0.0
    ortho_loss: float = 0.0
    total_loss: float = 0.0
    grad_norm: float = 0.0

    def to_wandb_dict(self, prefix: str = "target_wm") -> dict[str, float]:
        return {
            f"{prefix}/consistency_loss": self.consistency_loss,
            f"{prefix}/reward_loss": self.reward_loss,
            f"{prefix}/value_loss": self.value_loss,
            f"{prefix}/ortho_loss": self.ortho_loss,
            f"{prefix}/total_loss": self.total_loss,
            f"{prefix}/grad_norm": self.grad_norm,
        }


@dataclass
class ScaffoldedPolicyMetrics:
    """Target policy update metrics."""

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
class ExplorationPolicyMetrics:
    """Scaffolded exploration policy metrics."""

    pi_loss: float = 0.0
    entropy: float = 0.0

    def to_wandb_dict(self, prefix: str = "explore") -> dict[str, float]:
        return {
            f"{prefix}/pi_loss": self.pi_loss,
            f"{prefix}/entropy": self.entropy,
        }


@dataclass
class QMetrics:
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
class ScaffoldedTDMPC2Metrics(BaseMetrics):
    """Complete scaffolded TD-MPC2 training metrics."""

    target_wm: ScaffoldedWorldModelMetrics = field(default_factory=ScaffoldedWorldModelMetrics)
    scaff_wm: ScaffoldedWorldModelMetrics = field(default_factory=ScaffoldedWorldModelMetrics)
    policy: ScaffoldedPolicyMetrics = field(default_factory=ScaffoldedPolicyMetrics)
    explore: ExplorationPolicyMetrics = field(default_factory=ExplorationPolicyMetrics)
    q: QMetrics = field(default_factory=QMetrics)
    target_q: QMetrics = field(default_factory=QMetrics)
    batch: BatchMetrics = field(default_factory=BatchMetrics)
    total_updates: int = 0

    def get_console_metrics(self) -> list[ConsoleMetric]:
        return [
            ConsoleMetric("Name", MetricType.VALUE, "Scaff-TDMPC2"),
            # Target WM
            ConsoleMetric("T-Consist", MetricType.LOSS, self.target_wm.consistency_loss),
            ConsoleMetric("T-Reward", MetricType.LOSS, self.target_wm.reward_loss),
            ConsoleMetric("T-Value", MetricType.LOSS, self.target_wm.value_loss),
            ConsoleMetric("T-Ortho", MetricType.LOSS, self.target_wm.ortho_loss),
            ConsoleMetric("T-Total", MetricType.LOSS, self.target_wm.total_loss),
            # Scaffolded WM
            ConsoleMetric("S-Consist", MetricType.LOSS, self.scaff_wm.consistency_loss),
            ConsoleMetric("S-Total", MetricType.LOSS, self.scaff_wm.total_loss),
            # Policy
            ConsoleMetric("Pi Loss", MetricType.LOSS, self.policy.pi_loss),
            ConsoleMetric("Entropy", MetricType.ENTROPY, self.policy.pi_entropy),
            ConsoleMetric("Pi Scale", MetricType.VALUE, self.policy.pi_scale),
            # Explore
            ConsoleMetric("Exp Loss", MetricType.LOSS, self.explore.pi_loss),
            ConsoleMetric("Exp Ent", MetricType.ENTROPY, self.explore.entropy),
            # Grads
            ConsoleMetric("T-WM Grad", MetricType.VALUE, self.target_wm.grad_norm),
            ConsoleMetric("Pi Grad", MetricType.VALUE, self.policy.pi_grad_norm),
        ]

    def to_wandb_dict(self) -> dict[str, float]:
        result = {}
        result.update(self.target_wm.to_wandb_dict("target_wm"))
        result.update(self.scaff_wm.to_wandb_dict("scaff_wm"))
        result.update(self.policy.to_wandb_dict())
        result.update(self.explore.to_wandb_dict())
        result.update(self.q.to_wandb_dict("scaff_q"))
        result.update(self.target_q.to_wandb_dict("target_q"))
        result.update(self.batch.to_wandb_dict())
        result["total_updates"] = self.total_updates
        return result
