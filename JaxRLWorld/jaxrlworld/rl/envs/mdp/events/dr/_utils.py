"""Shared utilities for domain randomization terms.

Provides:
- ``sample`` — unified sampling with pluggable distributions.
- ``apply_operation`` — combine cached defaults with sampled values.
- ``resolve_patterns`` — regex-based name-to-index resolution.

Per-simulator baseline storage now lives next to each simulator's env
class (e.g. Newton uses ``jaxrlworld.rl.envs.utils.newton.dr_baselines``,
mjlab uses ``env.sim.get_default_field``).
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import torch


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
    raise ValueError(f"Unknown distribution {distribution!r}. Choose from 'uniform', 'log_uniform', 'gaussian'.")


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
    raise ValueError(f"Unknown operation {operation!r}. Choose from 'abs', 'scale', 'add'.")


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
            raise ValueError(f"Pattern {pat!r} matched no names. Available: {list(all_names)}")
        for idx in matched:
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
    return result
