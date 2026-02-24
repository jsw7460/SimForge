from dataclasses import dataclass, field

from rlworld.rl.algorithms.metrics.base import (
    BaseMetrics,
    ActorMetrics,
    BatchMetrics,
    MetricType,
    ConsoleMetric,
)


@dataclass
class PPODR3CriticMetrics:
    """PPO-DR3 critic metrics."""
    value_loss: float = 0.0
    dr3_loss: float = 0.0
    feature_dot_product: float = 0.0
    feature_cosine_similarity: float = 0.0
    feature_norm: float = 0.0

    def to_wandb_dict(self, prefix: str = "critic") -> dict[str, float]:
        return {
            f"{prefix}/value_loss": self.value_loss,
            f"{prefix}/dr3_loss": self.dr3_loss,
            f"{prefix}/feature_dot_product": self.feature_dot_product,
            f"{prefix}/feature_cosine_similarity": self.feature_cosine_similarity,
            f"{prefix}/feature_norm": self.feature_norm,
        }


@dataclass
class PPODR3ActorMetrics(ActorMetrics):
    """PPO-DR3 actor metrics (extends base)."""
    policy_loss: float = 0.0

    def to_wandb_dict(self, prefix: str = "actor") -> dict[str, float]:
        return {
            f"{prefix}/policy_loss": self.policy_loss,
            f"{prefix}/entropy": self.entropy,
            f"{prefix}/std": self.std,
        }


@dataclass
class PPODR3KLMetrics:
    """PPO-DR3 KL divergence and clipping metrics."""
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
class PPODR3Metrics(BaseMetrics):
    """Complete PPO-DR3 training metrics."""
    critic: PPODR3CriticMetrics = field(default_factory=PPODR3CriticMetrics)
    actor: PPODR3ActorMetrics = field(default_factory=PPODR3ActorMetrics)
    kl: PPODR3KLMetrics = field(default_factory=PPODR3KLMetrics)
    batch: BatchMetrics = field(default_factory=BatchMetrics)
    learning_rate: float = 0.0

    def get_console_metrics(self) -> list[ConsoleMetric]:
        """Return metrics with display info for console."""
        return [
            ConsoleMetric("Name", MetricType.VALUE, "PPO-DR3"),
            ConsoleMetric("Value Loss", MetricType.LOSS, self.critic.value_loss),
            ConsoleMetric("DR3 Loss", MetricType.LOSS, self.critic.dr3_loss),
            ConsoleMetric("Cos Sim", MetricType.VALUE, self.critic.feature_cosine_similarity),
            ConsoleMetric("Dot Prod", MetricType.VALUE, self.critic.feature_dot_product),
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
        result["learning_rate"] = self.learning_rate
        return result