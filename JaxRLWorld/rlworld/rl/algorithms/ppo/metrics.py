from dataclasses import dataclass, field

from rlworld.rl.algorithms.metrics.base import (
    ActorMetrics,
    BaseMetrics,
    BatchMetrics,
    ConsoleMetric,
    MetricType,
)


@dataclass
class PPOCriticMetrics:
    """PPO critic metrics."""

    value_loss: float = 0.0

    def to_wandb_dict(self, prefix: str = "critic") -> dict[str, float]:
        return {
            f"{prefix}/value_loss": self.value_loss,
        }


@dataclass
class PPOActorMetrics(ActorMetrics):
    """PPO actor metrics (extends base)."""

    policy_loss: float = 0.0

    def to_wandb_dict(self, prefix: str = "actor") -> dict[str, float]:
        return {
            f"{prefix}/policy_loss": self.policy_loss,
            f"{prefix}/entropy": self.entropy,
            f"{prefix}/std": self.std,
        }


@dataclass
class PPOKLMetrics:
    """PPO KL divergence and clipping metrics."""

    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    early_stop_ratio: float = 0.0
    actual_updates: int = 0
    expected_updates: int = 0

    def to_wandb_dict(self, prefix: str = "kl") -> dict[str, float]:
        return {
            f"{prefix}/approx_kl": self.approx_kl,
            f"{prefix}/clip_fraction": self.clip_fraction,
            f"{prefix}/early_stop_ratio": self.early_stop_ratio,
            f"{prefix}/actual_updates": self.actual_updates,
            f"{prefix}/expected_updates": self.expected_updates,
        }


@dataclass
class PPOMetrics(BaseMetrics):
    """Complete PPO training metrics."""

    critic: PPOCriticMetrics = field(default_factory=PPOCriticMetrics)
    actor: PPOActorMetrics = field(default_factory=PPOActorMetrics)
    kl: PPOKLMetrics = field(default_factory=PPOKLMetrics)
    batch: BatchMetrics = field(default_factory=BatchMetrics)
    learning_rate: float = 0.0

    def get_console_metrics(self) -> list[ConsoleMetric]:
        """Return metrics with display info for console."""
        return [
            ConsoleMetric("Name", MetricType.VALUE, "PPO"),
            ConsoleMetric("Value Loss", MetricType.LOSS, self.critic.value_loss),
            ConsoleMetric("Surrogate Loss", MetricType.LOSS, self.actor.policy_loss),
            ConsoleMetric("Entropy", MetricType.ENTROPY, self.actor.entropy),
            ConsoleMetric("LR", MetricType.COEFFICIENT, self.learning_rate),
            ConsoleMetric("KL", MetricType.VALUE, self.kl.approx_kl),
            ConsoleMetric("Clip Frac", MetricType.RATIO, self.kl.clip_fraction),
            ConsoleMetric("Std", MetricType.VALUE, self.actor.std),
        ]

    def to_wandb_dict(self) -> dict[str, float]:
        """Return all metrics for wandb."""
        result = {}
        result.update(self.critic.to_wandb_dict())
        result.update(self.actor.to_wandb_dict())
        result.update(self.kl.to_wandb_dict())
        result.update(self.batch.to_wandb_dict())
        return result
