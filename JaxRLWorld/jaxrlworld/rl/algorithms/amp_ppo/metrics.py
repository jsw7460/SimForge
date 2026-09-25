from dataclasses import dataclass, field

from jaxrlworld.rl.algorithms.metrics.base import ConsoleMetric, MetricType
from jaxrlworld.rl.algorithms.ppo.metrics import PPOMetrics


@dataclass
class AmpMetrics:
    """Discriminator and style-reward statistics of one iteration."""

    disc_loss: float = 0.0
    grad_penalty: float = 0.0
    policy_pred: float = 0.0
    expert_pred: float = 0.0
    accuracy_policy: float = 0.0
    accuracy_expert: float = 0.0
    style_reward: float = 0.0
    """Mean per-step style term as blended in (``w * dt * r_style``)."""
    task_reward: float = 0.0
    """Mean per-step task term as blended in (``(1 - w) * r_task``)."""
    style_reward_raw: float = 0.0
    """Mean discriminator reward before weighting."""
    style_weight: float = 0.0
    discriminator_lr: float = 0.0

    def to_wandb_dict(self, prefix: str = "amp") -> dict[str, float]:
        return {
            f"{prefix}/disc_loss": self.disc_loss,
            f"{prefix}/grad_penalty": self.grad_penalty,
            f"{prefix}/policy_pred": self.policy_pred,
            f"{prefix}/expert_pred": self.expert_pred,
            f"{prefix}/accuracy_policy": self.accuracy_policy,
            f"{prefix}/accuracy_expert": self.accuracy_expert,
            f"{prefix}/style_reward": self.style_reward,
            f"{prefix}/task_reward": self.task_reward,
            f"{prefix}/style_reward_raw": self.style_reward_raw,
            f"{prefix}/style_weight": self.style_weight,
            f"{prefix}/discriminator_lr": self.discriminator_lr,
        }


@dataclass
class AmpPPOMetrics(PPOMetrics):
    """PPO metrics plus the motion prior's."""

    amp: AmpMetrics = field(default_factory=AmpMetrics)

    def get_console_metrics(self) -> list[ConsoleMetric]:
        base = super().get_console_metrics()
        base[0] = ConsoleMetric("Name", MetricType.VALUE, "AMP-PPO")
        return base + [
            ConsoleMetric("D Loss", MetricType.LOSS, self.amp.disc_loss),
            ConsoleMetric("D GP", MetricType.LOSS, self.amp.grad_penalty),
            ConsoleMetric("D Acc Pol", MetricType.RATIO, self.amp.accuracy_policy),
            ConsoleMetric("D Acc Exp", MetricType.RATIO, self.amp.accuracy_expert),
            ConsoleMetric("Style R", MetricType.VALUE, self.amp.style_reward),
            ConsoleMetric("Task R", MetricType.VALUE, self.amp.task_reward),
        ]

    def to_wandb_dict(self) -> dict[str, float]:
        result = super().to_wandb_dict()
        result.update(self.amp.to_wandb_dict())
        return result
