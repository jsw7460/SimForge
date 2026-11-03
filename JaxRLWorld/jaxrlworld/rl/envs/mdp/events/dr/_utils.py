"""Shared utilities for domain randomization terms.

Provides:
- ``apply_operation`` — combine cached defaults with sampled values.
- ``resolve_patterns`` — regex-based name-to-index resolution.

Per-simulator baseline storage now lives next to each simulator's env
class (e.g. Newton uses ``jaxrlworld.rl.envs.utils.newton.dr_baselines``,
mjlab uses ``env.sim.get_default_field``).
"""

from __future__ import annotations

import re
from typing import Sequence

import torch


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
