from collections import deque
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxrlworld.rl.runners.iteration_data import EpisodeStats


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
        return {"mean": float(self.mean), "std": float(variance**0.5), "var": float(variance), "count": self.count}


class EpisodeStatsCollector:
    """Episode-level statistics for vectorized environments.

    The per-step ``update()`` is pure device work — no ``.cpu()``, no
    ``.item()``, no data-dependent Python branch — so it never stalls the
    CUDA pipeline mid-rollout (the previous version cost 2-3 full queue
    drains per collect step).  Completed episodes are scattered into
    device-side ring buffers with a cumsum-position trick (non-done lanes
    write to a dummy slot), and everything crosses to the host in ONE
    batched transfer per read burst (``_drain_to_host``), i.e. once per
    training iteration.
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
        self.episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Reward-type layout is fixed on the first update() (the reward
        # manager's term set is immutable after build).
        self._reward_types: list[str] | None = None
        self._episode_returns_per_type: torch.Tensor | None = None  # (n_types, num_envs)
        self._acc_sum: torch.Tensor | None = None  # (n_types,)
        self._acc_sumsq: torch.Tensor | None = None  # (n_types,)
        self._acc_steps = 0

        # Completed-episode ring buffers.  Slot ``window_size`` is a dummy
        # sink: every env writes each step, non-done lanes land there, so
        # the scatter needs no data-dependent indexing (= no host sync).
        w = window_size + 1
        self._ring_returns = torch.zeros(w, dtype=torch.float32, device=device)
        self._ring_discounted = torch.zeros(w, dtype=torch.float32, device=device)
        self._ring_success = torch.zeros(w, dtype=torch.float32, device=device)
        self._ring_lengths = torch.zeros(w, dtype=torch.float32, device=device)
        self._ring_per_type: torch.Tensor | None = None  # (n_types, w)
        self._write_ptr = torch.zeros((), dtype=torch.long, device=device)
        self._total_done = torch.zeros((), dtype=torch.long, device=device)
        self._dummy_slot = torch.full((), window_size, dtype=torch.long, device=device)

        # Host-side mirrors, rebuilt by _drain_to_host().
        self.return_history: deque = deque(maxlen=window_size)
        self.discounted_return_history: deque = deque(maxlen=window_size)
        self.episode_length_history: deque = deque(maxlen=window_size)
        self.return_history_per_type: dict[str, deque] = {}
        self.success_history: deque = deque(maxlen=window_size)
        self.reward_stats: dict[str, OnlineStats] = {}
        self._dirty = False

    # ------------------------------------------------------------ update

    def _init_type_layout(self, reward_info: dict[str, torch.Tensor]) -> None:
        self._reward_types = list(reward_info.keys())
        n = len(self._reward_types)
        self._episode_returns_per_type = torch.zeros(n, self.num_envs, dtype=torch.float32, device=self.device)
        self._acc_sum = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._acc_sumsq = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._ring_per_type = torch.zeros(n, self.window_size + 1, dtype=torch.float32, device=self.device)
        for name in self._reward_types:
            self.return_history_per_type[name] = deque(maxlen=self.window_size)
            self.reward_stats[name] = OnlineStats(self.device)

    def update(self, reward_info: dict[str, torch.Tensor], dones: torch.Tensor, success: torch.Tensor = None):
        """Update episode statistics with new step data.  Device-only."""
        assert "total_reward" in reward_info
        assert reward_info["total_reward"].shape[0] == self.num_envs
        assert dones.shape[0] == self.num_envs

        if self._reward_types is None:
            self._init_type_layout(reward_info)
        elif len(reward_info) != len(self._reward_types):
            raise RuntimeError(
                f"Reward-type set changed after first update: {sorted(reward_info)} vs {sorted(self._reward_types)}"
            )

        total = reward_info["total_reward"]
        self.episode_returns += total
        self.episode_discounted_returns += (self.gamma**self.current_step) * total

        vals = torch.stack([reward_info[k] for k in self._reward_types])  # (n_types, num_envs)
        self._episode_returns_per_type += vals
        self._acc_sum += vals.sum(dim=1)
        self._acc_sumsq += (vals * vals).sum(dim=1)
        self._acc_steps += 1

        if success is not None:
            self.episode_success = success

        self.current_step += 1

        # Completed episodes: scatter into the rings without ever asking
        # the host "did anything finish?".  Done lanes get consecutive
        # ring positions (matching the old sequential-extend order);
        # everyone else writes to the dummy slot.
        done_count_prefix = dones.long().cumsum(dim=0)
        n_done = done_count_prefix[-1]
        pos = (self._write_ptr + done_count_prefix - 1) % self.window_size
        pos = torch.where(dones, pos, self._dummy_slot)
        self._ring_returns.scatter_(0, pos, self.episode_returns)
        self._ring_discounted.scatter_(0, pos, self.episode_discounted_returns)
        self._ring_success.scatter_(0, pos, self.episode_success.float())
        self._ring_lengths.scatter_(0, pos, self.current_step.float())
        self._ring_per_type.scatter_(
            1, pos.unsqueeze(0).expand_as(self._episode_returns_per_type), self._episode_returns_per_type
        )
        self._write_ptr = (self._write_ptr + n_done) % self.window_size
        self._total_done = self._total_done + n_done

        # Reset finished episodes' accumulators.
        self.episode_returns.masked_fill_(dones, 0.0)
        self.episode_discounted_returns.masked_fill_(dones, 0.0)
        self.episode_success.masked_fill_(dones, False)
        self.current_step.masked_fill_(dones, 0)
        self._episode_returns_per_type.masked_fill_(dones.unsqueeze(0), 0.0)

        self._dirty = True

    # ------------------------------------------------------------- drain

    def _drain_to_host(self) -> None:
        """One batched device->host transfer; rebuilds the host mirrors.

        Called lazily by every host-reading getter, so it costs one sync
        per read burst (in training: once per iteration, at logging time).
        """
        if not self._dirty:
            return
        self._dirty = False

        w = self.window_size
        floats = torch.cat(
            [
                self._ring_returns[:w],
                self._ring_discounted[:w],
                self._ring_success[:w],
                self._ring_lengths[:w],
                self._ring_per_type[:, :w].reshape(-1),
                self._acc_sum,
                self._acc_sumsq,
            ]
        ).cpu()
        counters = torch.stack([self._write_ptr, self._total_done]).cpu()

        n_types = len(self._reward_types)
        rings = floats[: 4 * w].view(4, w).numpy()
        off = 4 * w
        per_type = floats[off : off + n_types * w].view(n_types, w).numpy()
        off += n_types * w
        acc_sum = floats[off : off + n_types].numpy()
        acc_sumsq = floats[off + n_types :].numpy()
        write_ptr, total_done = int(counters[0]), int(counters[1])

        # Rebuild the window deques oldest -> newest.
        valid = min(total_done, w)
        order = [(write_ptr - valid + i) % w for i in range(valid)]
        self.return_history = deque((float(rings[0, j]) for j in order), maxlen=w)
        self.discounted_return_history = deque((float(rings[1, j]) for j in order), maxlen=w)
        self.success_history = deque((bool(rings[2, j] > 0.5) for j in order), maxlen=w)
        self.episode_length_history = deque((int(rings[3, j]) for j in order), maxlen=w)
        for i, name in enumerate(self._reward_types):
            self.return_history_per_type[name] = deque((float(per_type[i, j]) for j in order), maxlen=w)

        # Fold the accumulated per-step batches into the running stats.
        count = self._acc_steps * self.num_envs
        if count > 0:
            for i, name in enumerate(self._reward_types):
                mean = float(acc_sum[i]) / count
                var = max(float(acc_sumsq[i]) / count - mean * mean, 0.0)
                self.reward_stats[name].update_from_stats(n=count, batch_mean=mean, batch_var=var)
        self._acc_sum.zero_()
        self._acc_sumsq.zero_()
        self._acc_steps = 0

    # ------------------------------------------------------------- reset

    def reset(self):
        """Reset current episode tracking (keeps history)."""
        self.current_step.zero_()
        self.episode_returns.zero_()
        self.episode_discounted_returns.zero_()
        self.episode_success.zero_()
        if self._episode_returns_per_type is not None:
            self._episode_returns_per_type.zero_()
            self._acc_sum.zero_()
            self._acc_sumsq.zero_()
        self._acc_steps = 0
        for stats in self.reward_stats.values():
            stats.reset()

    def reset_all(self):
        """Reset everything including history."""
        self.reset()
        self._ring_returns.zero_()
        self._ring_discounted.zero_()
        self._ring_success.zero_()
        self._ring_lengths.zero_()
        if self._ring_per_type is not None:
            self._ring_per_type.zero_()
        self._write_ptr.zero_()
        self._total_done.zero_()
        self._dirty = False
        self.return_history.clear()
        self.discounted_return_history.clear()
        self.episode_length_history.clear()
        self.success_history.clear()
        for dq in self.return_history_per_type.values():
            dq.clear()

    # ==================== Getters ====================

    def get_episode_returns(self) -> torch.Tensor:
        return self.episode_returns

    def get_discounted_return_history(self) -> list[float]:
        self._drain_to_host()
        return list(self.discounted_return_history)

    def get_success_rate(self):
        self._drain_to_host()
        if not self.success_history:
            return None
        return sum(self.success_history) / len(self.success_history)

    def get_mean_discounted_return(self) -> float:
        self._drain_to_host()
        if not self.discounted_return_history:
            return 0.0
        return sum(self.discounted_return_history) / len(self.discounted_return_history)

    def get_episode_lengths(self) -> torch.Tensor:
        return self.current_step

    def get_return_history(self) -> list[float]:
        self._drain_to_host()
        return list(self.return_history)

    def get_length_history(self) -> list[int]:
        self._drain_to_host()
        return list(self.episode_length_history)

    def get_return_history_per_type(self, reward_type: str) -> list[float]:
        self._drain_to_host()
        return list(self.return_history_per_type.get(reward_type, []))

    def get_reward_stats(self, reward_type: str) -> dict[str, float]:
        self._drain_to_host()
        if reward_type not in self.reward_stats:
            return {"mean": 0.0, "std": 0.0, "var": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return self.reward_stats[reward_type].get_stats()

    def get_all_reward_stats(self) -> dict[str, dict[str, float]]:
        self._drain_to_host()
        return {k: v.get_stats() for k, v in self.reward_stats.items()}

    def get_mean_episode_return(self) -> float:
        self._drain_to_host()
        if not self.return_history:
            return 0.0
        return sum(self.return_history) / len(self.return_history)

    def get_mean_episode_length(self) -> float:
        self._drain_to_host()
        if not self.episode_length_history:
            return 0.0
        return sum(self.episode_length_history) / len(self.episode_length_history)

    def get_summary(self) -> dict:
        return {
            "mean_return": self.get_mean_episode_return(),
            "mean_length": self.get_mean_episode_length(),
            "num_episodes": len(self.return_history),
            "reward_stats": self.get_all_reward_stats(),
        }

    # ==================== Legacy API ====================

    def get_current_returns(self, dones: torch.Tensor | None = None) -> torch.Tensor:
        if dones is not None and torch.any(dones):
            return self.episode_returns[dones]
        return self.episode_returns

    def get_returns_buffer(self) -> deque:
        self._drain_to_host()
        return self.return_history

    def get_length_buffer(self) -> deque:
        self._drain_to_host()
        return self.episode_length_history

    def get_returns_buffer_per_type(self) -> dict[str, deque]:
        self._drain_to_host()
        return dict(self.return_history_per_type)

    def get_reward_stats_per_type(self) -> dict[str, dict[str, float]]:
        return self.get_all_reward_stats()

    def snapshot(self) -> "EpisodeStats":
        """Create a typed EpisodeStats snapshot of current statistics."""
        from jaxrlworld.rl.runners.iteration_data import EpisodeStats

        self._drain_to_host()
        return EpisodeStats(
            return_buffer=list(self.return_history),
            length_buffer=list(self.episode_length_history),
            reward_stats=self.get_all_reward_stats(),
            success_rate=self.get_success_rate(),
        )
