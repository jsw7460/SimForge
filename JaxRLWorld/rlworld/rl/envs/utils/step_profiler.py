"""Per-section wall-clock timing for ``World.step()``.

Disabled by default — every call is a cheap no-op unless the env var
``JAXRLWORLD_PROFILE_STEP`` is truthy (``1`` / ``true`` / ...).  When
enabled it accumulates time per named section across steps and prints an
averaged breakdown (ms/step and % of the step) every
``JAXRLWORLD_PROFILE_STEP_INTERVAL`` steps (default 200), then resets.

CUDA work is asynchronous, so when a CUDA device is present the profiler
calls ``torch.cuda.synchronize()`` at each section boundary — section
times then reflect real GPU work (at the cost of serializing the
pipeline, so absolute step time grows a bit while profiling).

Usage in a step body::

    prof = self._step_profiler
    with prof.section("step_physics"):
        self._step_physics()
    ...
    prof.step_done()

The section names + insertion order are stable, so a Genesis vs Newton
vs MuJoCo comparison lines up row-for-row.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import torch


def _truthy(value: str) -> bool:
    return value.strip().lower() not in ("", "0", "false", "no", "off")


class StepProfiler:
    """Accumulates per-section step timings and prints periodic breakdowns."""

    def __init__(self, label: str = "World") -> None:
        self.label = label
        self.enabled = _truthy(os.environ.get("JAXRLWORLD_PROFILE_STEP", ""))
        try:
            self.interval = max(1, int(os.environ.get("JAXRLWORLD_PROFILE_STEP_INTERVAL", "200")))
        except ValueError:
            self.interval = 200
        self._sync = self.enabled and torch.cuda.is_available()
        self._totals: dict[str, float] = {}
        self._order: list[str] = []
        self._n_steps = 0

    def _now(self) -> float:
        if self._sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = self._now()
        try:
            yield
        finally:
            dt = self._now() - t0
            if name not in self._totals:
                self._totals[name] = 0.0
                self._order.append(name)
            self._totals[name] += dt

    def step_done(self) -> None:
        if not self.enabled:
            return
        self._n_steps += 1
        if self._n_steps >= self.interval:
            self._print()
            self._totals.clear()
            self._order.clear()
            self._n_steps = 0

    def _print(self) -> None:
        n = self._n_steps
        total = sum(self._totals.values()) or 1e-12
        print(f"\n[step profiler] {self.label} — avg over {n} step(s) ({total / n * 1e3:.3f} ms/step total):")
        for name in self._order:
            t = self._totals[name]
            print(f"  {name:<28s} {t / n * 1e3:9.4f} ms  {t / total * 100:5.1f}%")
