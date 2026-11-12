"""Preset builders for simulator backends that live outside this package.

A unified preset (``Go2FlatConfig`` ...) picks its per-simulator builder
module by ``sim_type``. The built-in simulators are listed in the preset
itself; a backend from another package adds its builders for a preset
here, from the callable its ``jaxrlworld.backends`` entry point runs (see
:mod:`jaxrlworld.rl.envs.backends`)::

    register_sim_builders(
        "go2",
        "mysim",
        SimBuilders(
            module="mypkg.presets.go2._mysim_builders",
            timing={"dt": 0.005, "substeps": 1, "decimation": 4},
            default_run_name="Go2_MySim",
        ),
    )

The builder module must satisfy the preset's builder protocol
(``presets/_sim_builder_protocol.py``).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from jaxrlworld.rl.envs.backends import load_plugins


@dataclass(frozen=True)
class SimBuilders:
    """One external simulator's builders for one preset."""

    module: str
    """Importable module exporting the preset's builder functions."""

    timing: dict[str, Any] = field(default_factory=dict)
    """The preset's ``_SIM_TIMINGS`` entry for this simulator."""

    default_run_name: str = ""
    """The preset's ``_SIM_DEFAULT_RUN_NAMES`` entry for this simulator."""

    runner_overrides: dict[str, Any] = field(default_factory=dict)
    """The preset's ``_SIM_RUNNER_OVERRIDES`` entry for this simulator, for
    presets that have that table (G1 29-DOF)."""

    def load(self) -> ModuleType:
        return importlib.import_module(self.module)


_REGISTRY: dict[tuple[str, str], SimBuilders] = {}


def register_sim_builders(preset: str, sim_type: str, builders: SimBuilders) -> None:
    """Register ``builders`` as preset ``preset``'s builders for ``sim_type``."""
    key = (preset, sim_type)
    existing = _REGISTRY.get(key)
    if existing is not None and existing != builders:
        raise ValueError(f"Preset {preset!r} already has {sim_type!r} builders: {existing}")
    _REGISTRY[key] = builders


def get_sim_builders(preset: str, sim_type: str, builtin: tuple[str, ...]) -> SimBuilders:
    """Preset ``preset``'s builders for the external ``sim_type``.

    ``builtin`` is the preset's own simulator list, quoted in the error
    for an unknown ``sim_type``.
    """
    key = (preset, sim_type)
    if key not in _REGISTRY:
        load_plugins()
    if key not in _REGISTRY:
        registered = sorted(s for p, s in _REGISTRY if p == preset)
        raise ValueError(
            f"Unknown sim_type: {sim_type!r}. Expected one of {sorted(builtin)} "
            f"or a registered external simulator {registered} for preset {preset!r}."
        )
    return _REGISTRY[key]
