"""Delay buffer for stochastically delayed observations.

Faithful port of mjlab's ``utils/buffers/delay_buffer.py`` semantics,
self-contained over an internal ring that reproduces the parts of
mjlab's CircularBuffer the delay logic relies on:

- per-env valid-frame counting (``current_length``) with lag clamping,
- BACKFILL: the first append after a reset copies that frame into every
  history slot of the reset rows, so any lag reads a valid value,
- reset rows read zeros until their next append.

At each ``append`` + ``compute`` pair (one per control step):

1. the new frame is appended to the ring,
2. the lag policy updates (see below) and samples per-env lags in
   ``[min_lag, max_lag]``,
3. the frame from ``lag`` steps ago is returned (clamped to the
   available history).

Lag update policy (mirrors mjlab exactly):

- ``update_period == 0``: every step may resample.
- ``update_period == N``: env ``i`` resamples only when
  ``(step_count + phase_offset_i) % N == 0``; with ``per_env_phase``
  each env draws a random phase in ``[0, N)`` so refreshes stagger
  across the batch.
- ``hold_prob``: even when a resample is due, keep the previous lag
  with this probability (temporal correlation).
- ``per_env=False``: one shared lag for the whole batch per resample.

Beyond the mjlab surface, :meth:`rollback_last` supports this
framework's terminal-observation capture contract (see
``World.step``): the capture runs a full observation pass and then
rewinds the buffers so the regular per-step pass is the only one that
counts. Rolling back restores the ring write head, the per-env push
counters, the lag-policy state AND the sampler's RNG state, so a step
with a capture consumes exactly the same random stream as a step
without one. The frame data overwritten in the ring by the captured
append is left in place — the immediate re-append of the same step
overwrites that slot again, the same caveat the history buffers'
``rollback_last`` accepts.
"""

from __future__ import annotations

import torch


class DelayBuffer:
    """Serve stochastically delayed frames from a rolling history.

    Args:
        min_lag: Minimum lag in steps (inclusive, >= 0).
        max_lag: Maximum lag in steps (inclusive, >= ``min_lag``).
        batch_size: Number of parallel environments.
        device: Torch device for storage and sampling.
        per_env: Sample an independent lag per environment; otherwise one
            shared lag per resample.
        hold_prob: Probability in ``[0, 1]`` of keeping the previous lag
            when a resample is due.
        update_period: Resample lags every N steps per env (0 = every step).
        per_env_phase: Stagger the periodic resamples with a random
            per-env phase offset in ``[0, update_period)``.
        generator: RNG for lag/phase sampling. Seeding it identically
            across simulator backends keeps the lag streams aligned.
    """

    def __init__(
        self,
        min_lag: int = 0,
        max_lag: int = 3,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        per_env: bool = True,
        hold_prob: float = 0.0,
        update_period: int = 0,
        per_env_phase: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if min_lag < 0:
            raise ValueError(f"min_lag must be >= 0, got {min_lag}")
        if max_lag < min_lag:
            raise ValueError(f"max_lag ({max_lag}) must be >= min_lag ({min_lag})")
        if not 0.0 <= hold_prob <= 1.0:
            raise ValueError(f"hold_prob must be in [0, 1], got {hold_prob}")
        if update_period < 0:
            raise ValueError(f"update_period must be >= 0, got {update_period}")

        self.min_lag = min_lag
        self.max_lag = max_lag
        self.batch_size = batch_size
        self.device = device
        self.per_env = per_env
        self.hold_prob = hold_prob
        self.update_period = update_period
        self.per_env_phase = per_env_phase
        self.generator = generator

        self._max_len = max_lag + 1 if max_lag > 0 else 1
        self._ring: torch.Tensor | None = None  # (max_len, batch, ...)
        self._pointer = -1
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._all_indices = torch.arange(batch_size, device=device)
        self._current_lags = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._step_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        if update_period > 0 and per_env_phase:
            self._phase_offsets = torch.randint(
                0, update_period, (batch_size,), dtype=torch.long, device=device, generator=generator
            )
        else:
            self._phase_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Snapshot taken at the top of append(); see rollback_last().
        self._snapshot: tuple | None = None

    @property
    def is_initialized(self) -> bool:
        return self._ring is not None

    @property
    def current_lags(self) -> torch.Tensor:
        """Current lag per environment. Shape ``(batch_size,)``."""
        return self._current_lags

    def reset(self, batch_ids: torch.Tensor | None = None) -> None:
        """Reset the given envs: zero their history, lag and step counters.

        Their next :meth:`append` backfills the whole ring row with the
        first new frame; until then :meth:`compute`/:meth:`peek` return
        zeros for those rows. With periodic updates and per-env phases,
        the reset rows draw fresh phase offsets (mjlab semantics).
        """
        idx = slice(None) if batch_ids is None else batch_ids
        self._num_pushes[idx] = 0
        if self._ring is not None:
            self._ring[:, idx] = 0.0
        self._current_lags[idx] = 0
        self._step_count[idx] = 0
        if self.update_period > 0 and self.per_env_phase:
            new_phases = torch.randint(
                0,
                self.update_period,
                (self.batch_size,),
                dtype=torch.long,
                device=self.device,
                generator=self.generator,
            )
            self._phase_offsets[idx] = new_phases[idx]

    def append(self, data: torch.Tensor) -> None:
        """Append one frame ``(batch_size, ...)`` and snapshot for rollback."""
        if self._ring is None:
            self._pointer = -1
            self._ring = torch.zeros((self._max_len, *data.shape), dtype=data.dtype, device=self.device)

        self._snapshot = (
            self._pointer,
            self._num_pushes.clone(),
            self._current_lags.clone(),
            self._step_count.clone(),
            self.generator.get_state() if self.generator is not None else None,
        )

        self._pointer = (self._pointer + 1) % self._max_len
        self._ring[self._pointer] = data

        # Backfill the whole row with the first frame after a reset so
        # every lag reads a valid value instead of zeros.
        is_first_push = self._num_pushes == 0
        condition = is_first_push.view(1, self.batch_size, *([1] * (data.ndim - 1)))
        torch.where(condition, data.unsqueeze(0), self._ring, out=self._ring)

        self._num_pushes += 1

    def compute(self) -> torch.Tensor:
        """Advance the lag policy and return the delayed frame."""
        if self._ring is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")
        self._update_lags()
        return self._gather(self._current_lags)

    def peek(self) -> torch.Tensor:
        """Return the delayed frame WITHOUT advancing the lag policy.

        Used by observation passes that must not consume delay state
        (e.g. dimension probing), mirroring mjlab's cached-compute
        behavior of serving the last result on such calls.
        """
        if self._ring is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")
        return self._gather(self._current_lags)

    def rollback_last(self) -> None:
        """Undo the most recent append()+compute() pair.

        Restores the write head, push counters, lag-policy state and the
        sampler's RNG state captured at the top of the last append().
        No-op when nothing was appended since construction.
        """
        if self._snapshot is None:
            return
        pointer, num_pushes, lags, step_count, gen_state = self._snapshot
        self._pointer = pointer
        self._num_pushes = num_pushes
        self._current_lags = lags
        self._step_count = step_count
        if gen_state is not None:
            self.generator.set_state(gen_state)
        self._snapshot = None

    def _gather(self, lags: torch.Tensor) -> torch.Tensor:
        # Clamp to the available history: a freshly reset row has one
        # (backfilled) valid frame, so any lag resolves to it; rows that
        # never appended since reset read the zeroed slots.
        pushes = self._num_pushes.clamp(min=1)
        valid = torch.minimum(lags, torch.minimum(pushes, torch.full_like(pushes, self._max_len)) - 1).clamp(min=0)
        idx = torch.remainder(self._pointer - valid, self._max_len)
        return self._ring[idx, self._all_indices]

    def _update_lags(self) -> None:
        if self.update_period > 0:
            phase_adjusted = (self._step_count + self._phase_offsets) % self.update_period
            should_update = phase_adjusted == 0
        else:
            should_update = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        new_lags = self._sample_lags(should_update)
        self._current_lags = torch.where(should_update, new_lags, self._current_lags)
        self._step_count += 1

    def _sample_lags(self, mask: torch.Tensor) -> torch.Tensor:
        if self.per_env:
            candidate = torch.randint(
                self.min_lag,
                self.max_lag + 1,
                (self.batch_size,),
                dtype=torch.long,
                device=self.device,
                generator=self.generator,
            )
        else:
            shared = torch.randint(
                self.min_lag,
                self.max_lag + 1,
                (1,),
                dtype=torch.long,
                device=self.device,
                generator=self.generator,
            )
            candidate = shared.expand(self.batch_size)

        if self.hold_prob > 0.0:
            keep_sampling = (
                torch.rand(self.batch_size, dtype=torch.float32, device=self.device, generator=self.generator)
                >= self.hold_prob
            )
            update_mask = mask & keep_sampling
        else:
            update_mask = mask
        return torch.where(update_mask, candidate, self._current_lags)
