"""Shared utilities for domain randomization terms.

Provides:
- ``DefaultCache`` — caches simulator default values so that ``scale`` and
  ``add`` operations always reference the original, preventing drift.
- ``sample`` — unified sampling with pluggable distributions.
- ``resolve_patterns`` — regex-based name-to-index resolution.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import torch


class DefaultCache:
    """Caches original parameter values to prevent accumulation.

    When using ``operation="scale"`` or ``"add"``, the random perturbation
    must be applied to the *original* value, not the already-perturbed one.
    This cache stores the first-seen value for each key and always returns it.
    """

    def __init__(self) -> None:
        self._store: dict[str, torch.Tensor] = {}

    def get_or_cache(self, key: str, current: torch.Tensor) -> torch.Tensor:
        """Return cached default; on first call, cache *current*."""
        if key not in self._store:
            self._store[key] = current.clone()
        return self._store[key]

    def clear(self, key: str | None = None) -> None:
        """Drop one key or all cached defaults."""
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)


def sample(
    shape: tuple[int, ...],
    lo: float,
    hi: float,
    device: torch.device,
    distribution: str = "uniform",
) -> torch.Tensor:
    """Sample a tensor of the given *shape* from *distribution*.

    Args:
        shape: Output shape.
        lo: Lower bound (or lower parameter for gaussian).
        hi: Upper bound (or upper parameter for gaussian).
        device: Torch device.
        distribution: One of ``"uniform"``, ``"log_uniform"``, ``"gaussian"``.

    Returns:
        Sampled tensor on *device*.
    """
    if distribution == "uniform":
        return torch.empty(shape, device=device).uniform_(lo, hi)
    elif distribution == "log_uniform":
        log_lo = math.log(max(lo, 1e-8))
        log_hi = math.log(max(hi, 1e-8))
        return torch.empty(shape, device=device).uniform_(log_lo, log_hi).exp()
    elif distribution == "gaussian":
        mean = (lo + hi) / 2.0
        std = (hi - lo) / 4.0  # 95% within [lo, hi]
        return torch.empty(shape, device=device).normal_(mean, std).clamp_(lo, hi)
    raise ValueError(
        f"Unknown distribution {distribution!r}. "
        "Choose from 'uniform', 'log_uniform', 'gaussian'."
    )


def apply_operation(
    defaults: torch.Tensor,
    sampled: torch.Tensor,
    operation: str,
) -> torch.Tensor:
    """Combine *defaults* with *sampled* values according to *operation*.

    Args:
        defaults: Original (cached) values.
        sampled: Freshly sampled random values.
        operation: One of ``"abs"`` (replace), ``"scale"`` (multiply),
            ``"add"`` (offset).

    Returns:
        New parameter values.
    """
    if operation == "abs":
        return sampled
    elif operation == "scale":
        return defaults * sampled
    elif operation == "add":
        return defaults + sampled
    raise ValueError(
        f"Unknown operation {operation!r}. Choose from 'abs', 'scale', 'add'."
    )


def resolve_patterns(
    patterns: str | list[str],
    all_names: Sequence[str],
) -> list[int]:
    """Match regex *patterns* against *all_names*, returning matched indices.

    Args:
        patterns: A single pattern string or list of regex patterns.
        all_names: Ordered names to match against.

    Returns:
        Deduplicated list of matched indices (preserving pattern order).

    Raises:
        ValueError: If any pattern matches nothing.
    """
    if isinstance(patterns, str):
        patterns = [patterns]

    seen: set[int] = set()
    result: list[int] = []
    for pat in patterns:
        matched = [i for i, name in enumerate(all_names) if re.fullmatch(pat, name)]
        if not matched:
            raise ValueError(
                f"Pattern {pat!r} matched no names. "
                f"Available: {list(all_names)}"
            )
        for idx in matched:
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
    return result
