from dataclasses import dataclass, field

from rlworld.rl.algorithms.metrics.base import (
    BaseMetrics,
    ActorMetrics,
    MetricType,
    ConsoleMetric,
)


@dataclass
class SACCriticMetrics:
    """SAC twin critic metrics."""
    loss: float = 0.0
    critic1_loss: float = 0.0
    critic2_loss: float = 0.0
    q1_mean: float = 0.0
    q2_mean: float = 0.0
    q1_std: float = 0.0
    q2_std: float = 0.0
    q_target_mean: float = 0.0

    def to_wandb_dict(self, prefix: str = "critic") -> dict[str, float]:
        return {
            f"{prefix}/loss": self.loss,
            f"{prefix}/critic1_loss": self.critic1_loss,
            f"{prefix}/critic2_loss": self.critic2_loss,
            f"{prefix}/q1_mean": self.q1_mean,
            f"{prefix}/q2_mean": self.q2_mean,
            f"{prefix}/q1_std": self.q1_std,
            f"{prefix}/q2_std": self.q2_std,
            f"{prefix}/q_target_mean": self.q_target_mean,
            f"{prefix}/q_diff": self.q1_mean - self.q2_mean,
        }


@dataclass
class SACAlphaMetrics:
    """SAC entropy coefficient metrics."""
    value: float = 0.0
    loss: float = 0.0
    target_entropy: float = 0.0
    entropy_gap: float = 0.0

    def to_wandb_dict(self, prefix: str = "alpha") -> dict[str, float]:
        return {
            f"{prefix}/value": self.value,
            f"{prefix}/loss": self.loss,
            f"{prefix}/target_entropy": self.target_entropy,
            f"{prefix}/entropy_gap": self.entropy_gap,
        }


@dataclass
class SACBatchMetrics:
    """SAC batch statistics (uses actual rewards, not returns)."""
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    action_mean: float = 0.0
    action_std: float = 0.0
    terminated_ratio: float = 0.0

    def to_wandb_dict(self, prefix: str = "batch") -> dict[str, float]:
        return {
            f"{prefix}/reward_mean": self.reward_mean,
            f"{prefix}/reward_std": self.reward_std,
            f"{prefix}/reward_min": self.reward_min,
            f"{prefix}/reward_max": self.reward_max,
            f"{prefix}/action_mean": self.action_mean,
            f"{prefix}/action_std": self.action_std,
            f"{prefix}/terminated_ratio": self.terminated_ratio,
        }


@dataclass
class SACMetrics(BaseMetrics):
    """Complete SAC training metrics."""
    critic: SACCriticMetrics = field(default_factory=SACCriticMetrics)
    actor: ActorMetrics = field(default_factory=ActorMetrics)
    alpha: SACAlphaMetrics = field(default_factory=SACAlphaMetrics)
    batch: SACBatchMetrics = field(default_factory=SACBatchMetrics)
    total_updates: int = 0

    def get_console_metrics(self) -> list[ConsoleMetric]:
        """Return metrics with display info for console."""
        return [
            ConsoleMetric("Name", MetricType.VALUE, "SAC"),
            ConsoleMetric("Actor Loss", MetricType.LOSS, self.actor.loss),
            ConsoleMetric("Critic Loss", MetricType.LOSS, self.critic.loss),
            ConsoleMetric("Critic1 Loss", MetricType.LOSS, self.critic.critic1_loss),
            ConsoleMetric("Critic2 Loss", MetricType.LOSS, self.critic.critic2_loss),
            ConsoleMetric("Alpha Loss", MetricType.LOSS, self.alpha.loss),
            ConsoleMetric("Q1 Mean", MetricType.VALUE, self.critic.q1_mean),
            ConsoleMetric("Q2 Mean", MetricType.VALUE, self.critic.q2_mean),
            ConsoleMetric("Q Target", MetricType.VALUE, self.critic.q_target_mean),
            ConsoleMetric("Q1 Std", MetricType.VALUE, self.critic.q1_std),
            ConsoleMetric("Q2 Std", MetricType.VALUE, self.critic.q2_std),
            ConsoleMetric("Entropy", MetricType.ENTROPY, self.actor.entropy),
            ConsoleMetric("Alpha", MetricType.COEFFICIENT, self.alpha.value),
            ConsoleMetric("Target Ent", MetricType.VALUE, self.alpha.target_entropy),
        ]

    def to_wandb_dict(self) -> dict[str, float]:
        """Return all metrics for wandb."""
        result = {
            "actor/loss": self.actor.loss,
            "actor/entropy": self.actor.entropy,
            "train/total_updates": self.total_updates,
        }
        result.update(self.critic.to_wandb_dict())
        result.update(self.alpha.to_wandb_dict())
        result.update(self.batch.to_wandb_dict())
        return result