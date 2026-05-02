"""NPMP distillation metrics for console + wandb logging.

Mirrors the :class:`PPOMetrics` pattern from
``rlworld.rl.algorithms.ppo.metrics`` so that ``ConsoleWriter`` and
``WandbLogger`` from the RL stack can consume distillation iteration
data without any custom-formatting glue.

Fields are populated from :class:`NPMPLossInfo` plus a few aggregate
statistics (decoder log-std spread, encoder posterior log-std mean)
that are cheap to compute and useful for diagnosing collapse vs blow-
up of the latent posterior during early training.
"""
from __future__ import annotations

from dataclasses import dataclass

from rlworld.rl.algorithms.metrics.base import (
    BaseMetrics,
    ConsoleMetric,
    MetricType,
)


__all__ = ["NPMPMetrics"]


@dataclass
class NPMPMetrics(BaseMetrics):
    """Per-iteration distillation metrics."""

    # Loss decomposition.
    loss: float
    recon: float
    kl: float
    beta: float

    # Decoder action log-std (per-action-dim, learnable, state-independent).
    decoder_log_std_mean: float
    decoder_log_std_min: float
    decoder_log_std_max: float

    # Encoder posterior log-std (averaged over batch x time x latent).
    encoder_q_log_std_mean: float

    def get_console_metrics(self) -> list[ConsoleMetric]:
        return [
            ConsoleMetric("Total Loss",            MetricType.LOSS,        self.loss),
            ConsoleMetric("Recon NLL",             MetricType.LOSS,        self.recon),
            ConsoleMetric("KL(q || p)",            MetricType.LOSS,        self.kl),
            ConsoleMetric("beta",                  MetricType.COEFFICIENT, self.beta),
            ConsoleMetric("Decoder log_std mean",  MetricType.VALUE,       self.decoder_log_std_mean),
            ConsoleMetric("Decoder log_std min",   MetricType.VALUE,       self.decoder_log_std_min),
            ConsoleMetric("Decoder log_std max",   MetricType.VALUE,       self.decoder_log_std_max),
            ConsoleMetric("Encoder q log_std",     MetricType.VALUE,       self.encoder_q_log_std_mean),
        ]

    def to_wandb_dict(self) -> dict[str, float]:
        return {
            "Loss/total": self.loss,
            "Loss/recon": self.recon,
            "Loss/kl": self.kl,
            "Loss/beta": self.beta,
            "Decoder/log_std_mean": self.decoder_log_std_mean,
            "Decoder/log_std_min": self.decoder_log_std_min,
            "Decoder/log_std_max": self.decoder_log_std_max,
            "Encoder/q_log_std_mean": self.encoder_q_log_std_mean,
        }
