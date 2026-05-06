"""Typed dataclasses for training iteration data.

Replaces the untyped Dict[str, Any] that was previously passed between
runners, loggers, and console writers.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeStats:
    """Typed container for episode-level statistics."""

    return_buffer: list[float]
    length_buffer: list[float]
    reward_stats: dict[str, dict[str, float]]
    success_rate: float | None = None
    per_sim_stats: dict[str, EpisodeStats] | None = None

    @property
    def mean_return(self) -> float:
        if not self.return_buffer:
            return 0.0
        return statistics.mean(self.return_buffer)

    @property
    def mean_episode_length(self) -> float:
        if not self.length_buffer:
            return 0.0
        return statistics.mean(self.length_buffer)

    def to_wandb_dict(self) -> dict[str, float]:
        d: dict[str, float] = {}
        if self.return_buffer:
            d["Train/mean_return"] = self.mean_return
        if self.length_buffer:
            d["Train/mean_episode_length"] = self.mean_episode_length
        if self.success_rate is not None:
            d["Train/success_rate"] = self.success_rate

        for reward_name, stats in self.reward_stats.items():
            for category, val in stats.items():
                d[f"Rewards/{category}/{reward_name}"] = val

        # Per-sim stats
        if self.per_sim_stats:
            for sim_name, sim_stats in self.per_sim_stats.items():
                if sim_stats.return_buffer:
                    d[f"PerSim/{sim_name}/mean_return"] = sim_stats.mean_return
                if sim_stats.length_buffer:
                    d[f"PerSim/{sim_name}/mean_episode_length"] = sim_stats.mean_episode_length
                if sim_stats.success_rate is not None:
                    d[f"PerSim/{sim_name}/success_rate"] = sim_stats.success_rate
                for reward_name, stats in sim_stats.reward_stats.items():
                    for category, val in stats.items():
                        d[f"PerSim/{sim_name}/Rewards/{category}/{reward_name}"] = val

        return d


@dataclass
class IterationData:
    """Typed container for a single training iteration's data."""

    collection_time: float
    learning_time: float
    episode_stats: EpisodeStats
    fps: float = 0.0
    metrics: Any = None  # BaseMetrics subclass (PPOMetrics, SACMetrics, etc.)
    last_obs: Any = None  # runner-specific, not logged
    action_distribution: dict[str, Any] = field(default_factory=dict)
    buffer_size: int | None = None
    iteration: int = 0
    total_timesteps: int = 0
    total_time: float = 0.0

    def to_wandb_dict(self) -> dict[str, float]:
        d: dict[str, float] = {}

        # Episode stats
        d.update(self.episode_stats.to_wandb_dict())

        # Algorithm metrics
        if self.metrics is not None and hasattr(self.metrics, "to_wandb_dict"):
            d.update(self.metrics.to_wandb_dict())

        # Performance
        d["Performance/fps"] = self.fps
        d["Performance/collection_time"] = self.collection_time
        d["Performance/learning_time"] = self.learning_time
        d["total_timesteps"] = self.total_timesteps
        d["iteration"] = self.iteration

        if self.buffer_size is not None:
            d["Performance/buffer_size"] = self.buffer_size

        return d
