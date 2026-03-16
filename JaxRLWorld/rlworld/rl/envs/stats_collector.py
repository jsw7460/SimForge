from collections import deque, defaultdict
from typing import Optional

import torch


class OnlineStats:
    """Efficient online statistics calculator using vectorized Welford's algorithm."""

    def __init__(self, device: str):
        self.device = device
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_from_stats(
        self,
        n: int,
        batch_mean: float,
        batch_var: float,
    ):
        """
        Update statistics from pre-computed batch statistics.

        Args:
            n: Number of samples in batch
            batch_mean: Pre-computed mean
            batch_var: Pre-computed variance (unbiased=False)
        """
        if n == 0:
            return

        if self.count == 0:
            self.mean = batch_mean
            self.M2 = batch_var * n
            self.count = n
        else:
            new_count = self.count + n
            delta = batch_mean - self.mean
            self.mean += delta * n / new_count
            self.M2 += batch_var * n + delta * delta * self.count * n / new_count
            self.count = new_count

    def reset(self):
        """Reset all statistics."""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def get_stats(self) -> dict[str, float]:
        """Get current statistics."""
        if self.count == 0:
            return {"mean": 0.0, "std": 0.0, "var": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        variance = self.M2 / self.count if self.count > 1 else 0.0
        return {
            "mean": float(self.mean),
            "std": float(variance ** 0.5),
            "var": float(variance),
            "count": self.count
        }



class EpisodeStatsCollector:
    """
    Collects and manages episode-level statistics for vectorized environments.
    """

    def __init__(
        self,
        num_envs: int,
        max_episode_length: int,
        device: torch.device,
        gamma: float,
        window_size: int = 100,
    ):
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.window_size = window_size
        self.device = device
        self.gamma = gamma

        # Current episode tracking (on GPU)
        self.current_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.episode_discounted_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # Per-reward-type returns (current episode)
        self.episode_returns_per_type = defaultdict(
            lambda: torch.zeros(num_envs, dtype=torch.float32, device=device)
        )

        # Historical data (CPU)
        self.return_history = deque(maxlen=window_size)
        self.discounted_return_history = deque(maxlen=window_size)
        self.episode_length_history = deque(maxlen=window_size)
        self.return_history_per_type = defaultdict(lambda: deque(maxlen=window_size))

        # Online statistics
        self.reward_stats = defaultdict(lambda: OnlineStats(device))

        # For success/fail tasks
        self.episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success_history = deque(maxlen=window_size)

    def update(
        self,
        reward_info: dict[str, torch.Tensor],
        dones: torch.Tensor,
        success: torch.Tensor = None
    ):
        """Update episode statistics with new step data."""
        assert "total_reward" in reward_info
        assert reward_info["total_reward"].shape[0] == self.num_envs
        assert dones.shape[0] == self.num_envs

        # Update current episode returns
        self.episode_returns += reward_info["total_reward"]

        # Update discounted returns
        discount_factor = self.gamma ** self.current_step
        self.episode_discounted_returns += discount_factor * reward_info["total_reward"]

        # Batch compute all reward stats on GPU (single sync)
        reward_types = list(reward_info.keys())
        reward_tensors = torch.stack([reward_info[k] for k in reward_types])  # (num_types, num_envs)

        all_means = reward_tensors.mean(dim=1)
        all_vars = reward_tensors.var(dim=1, unbiased=False)

        # Single GPU→CPU transfer
        all_stats = torch.stack([all_means, all_vars]).cpu().numpy()

        # Update per-type returns and statistics
        for i, reward_type in enumerate(reward_types):
            self.episode_returns_per_type[reward_type] += reward_info[reward_type]
            self.reward_stats[reward_type].update_from_stats(
                n=self.num_envs,
                batch_mean=all_stats[0, i],
                batch_var=all_stats[1, i],
            )

        if success is not None:
            self.episode_success = success

        self.current_step += 1

        # Process completed episodes
        if torch.any(dones):
            self._process_completed_episodes(dones)

    def _process_completed_episodes(self, dones: torch.Tensor):
        """Process and store statistics for completed episodes."""
        done_indices = dones.nonzero(as_tuple=True)[0]

        if len(done_indices) == 0:
            return
        # Batch indexing for CPU transfer
        tensors_to_transfer = [
            self.episode_returns[done_indices],
            self.episode_discounted_returns[done_indices],
            self.episode_success[done_indices].float(),
            self.current_step[done_indices].float(),
        ]
        reward_types = list(self.episode_returns_per_type.keys())
        for reward_type in reward_types:
            tensors_to_transfer.append(self.episode_returns_per_type[reward_type][done_indices])

        # Single GPU→CPU transfer
        stacked = torch.stack(tensors_to_transfer).cpu().numpy()

        self.return_history.extend(stacked[0])
        self.discounted_return_history.extend(stacked[1])
        self.success_history.extend(stacked[2].astype(bool))
        self.episode_length_history.extend(stacked[3].astype(int))
        for i, reward_type in enumerate(reward_types):
            self.return_history_per_type[reward_type].extend(stacked[4 + i])

        # Reset using boolean mask (faster than index assignment)
        self.episode_returns.masked_fill_(dones, 0.0)
        self.episode_discounted_returns.masked_fill_(dones, 0.0)
        self.episode_success.masked_fill_(dones, False)
        self.current_step.masked_fill_(dones, 0)
        for returns in self.episode_returns_per_type.values():
            returns.masked_fill_(dones, 0.0)

    def reset(self):
        """Reset current episode tracking (keeps history)."""
        self.current_step.zero_()
        self.episode_returns.zero_()
        self.episode_discounted_returns.zero_()

        for returns in self.episode_returns_per_type.values():
            returns.zero_()

        for stats in self.reward_stats.values():
            stats.reset()

    def reset_all(self):
        """Reset everything including history."""
        self.reset()
        self.return_history.clear()
        self.discounted_return_history.clear()
        self.episode_length_history.clear()
        self.return_history_per_type.clear()

    # ==================== Getters ====================

    def get_episode_returns(self) -> torch.Tensor:
        return self.episode_returns

    def get_discounted_return_history(self) -> list[float]:
        return list(self.discounted_return_history)

    def get_success_rate(self):
        if not self.success_history:
            return None
        return sum(self.success_history) / len(self.success_history)

    def get_mean_discounted_return(self) -> float:
        if not self.discounted_return_history:
            return 0.0
        return sum(self.discounted_return_history) / len(self.discounted_return_history)

    def get_episode_lengths(self) -> torch.Tensor:
        return self.current_step

    def get_return_history(self) -> list[float]:
        return list(self.return_history)

    def get_length_history(self) -> list[int]:
        return list(self.episode_length_history)

    def get_return_history_per_type(self, reward_type: str) -> list[float]:
        return list(self.return_history_per_type.get(reward_type, []))

    def get_reward_stats(self, reward_type: str) -> dict[str, float]:
        if reward_type not in self.reward_stats:
            return {"mean": 0.0, "std": 0.0, "var": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return self.reward_stats[reward_type].get_stats()

    def get_all_reward_stats(self) -> dict[str, dict[str, float]]:
        return {k: v.get_stats() for k, v in self.reward_stats.items()}

    def get_mean_episode_return(self) -> float:
        if not self.return_history:
            return 0.0
        return sum(self.return_history) / len(self.return_history)

    def get_mean_episode_length(self) -> float:
        if not self.episode_length_history:
            return 0.0
        return sum(self.episode_length_history) / len(self.episode_length_history)

    def get_summary(self) -> dict:
        return {
            "mean_return": self.get_mean_episode_return(),
            "mean_length": self.get_mean_episode_length(),
            "num_episodes": len(self.return_history),
            "reward_stats": self.get_all_reward_stats()
        }

    # ==================== Legacy API ====================

    def get_current_returns(self, dones: Optional[torch.Tensor] = None) -> torch.Tensor:
        if dones is not None and torch.any(dones):
            return self.episode_returns[dones]
        return self.episode_returns

    def get_returns_buffer(self) -> deque:
        return self.return_history

    def get_length_buffer(self) -> deque:
        return self.episode_length_history

    def get_returns_buffer_per_type(self) -> dict[str, deque]:
        return dict(self.return_history_per_type)

    def get_reward_stats_per_type(self) -> dict[str, dict[str, float]]:
        return self.get_all_reward_stats()

    def snapshot(self) -> "EpisodeStats":
        """Create a typed EpisodeStats snapshot of current statistics."""
        from rlworld.rl.runners.iteration_data import EpisodeStats
        return EpisodeStats(
            return_buffer=list(self.return_history),
            length_buffer=list(self.episode_length_history),
            reward_stats=self.get_all_reward_stats(),
            success_rate=self.get_success_rate(),
        )