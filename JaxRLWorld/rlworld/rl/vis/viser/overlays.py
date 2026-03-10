"""Overlay managers for Viser viewer orchestration.

Coordinate *when* higher-level updates happen (env switches, paused/running),
while leaving render handle lifecycle in scene.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import viser

from .term_plotter import ViserTermPlotter


class _EnvProtocol(Protocol):
    """Minimal env interface for overlays."""

    rew_buf_per_type: dict[str, Any]


class _SceneProtocol(Protocol):
    env_idx: int
    needs_update: bool

    def clear_debug(self) -> None: ...


def _get_reward_term_names(env: Any) -> list[str]:
    """Extract reward term names from the reward manager config."""
    rm = getattr(env, "reward_manager", None)
    if rm is None:
        return []
    terms = getattr(rm, "reward_terms", None)
    if not terms:
        return []
    names = []
    for idx, term in enumerate(terms):
        func = term.func
        # Stateful reward classes use their name attribute.
        if isinstance(func, type):
            name = getattr(func, "name", None) or func.__name__
        else:
            name = getattr(func, "__name__", f"reward_{idx}")
        names.append(name)
    return names


@dataclass
class ViserTermOverlays:
    """Manage reward term plot tabs."""

    server: viser.ViserServer
    env: _EnvProtocol
    scene: _SceneProtocol
    reward_plotter: ViserTermPlotter | None = None
    _initialized: bool = False

    def setup_tabs(self, tabs: Any) -> None:
        """Create rewards tab. Gets names from reward config (not rew_buf_per_type)."""
        reward_names = _get_reward_term_names(self.env)
        if not reward_names:
            # Defer — will try lazy init on first update.
            self._tabs = tabs
            return

        self._create_plotter(tabs, reward_names)

    def _create_plotter(self, tabs: Any, reward_names: list[str]) -> None:
        with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
            self.reward_plotter = ViserTermPlotter(
                self.server,
                reward_names,
                name="Reward",
                env_idx=self.scene.env_idx,
            )
        self._initialized = True

    def on_env_switch(self) -> None:
        """Clear histories when active environment changes."""
        env_idx = self.scene.env_idx
        if self.reward_plotter:
            self.reward_plotter.clear_histories()
            self.reward_plotter.update_env_idx(env_idx)

    def update(self) -> None:
        """Update term plots from the selected environment."""
        # Lazy init if reward names weren't available at setup time.
        if not self._initialized and hasattr(self, "_tabs"):
            names = list(self.env.rew_buf_per_type.keys())
            if names:
                self._create_plotter(self._tabs, names)

        if self.reward_plotter is None:
            return

        env_idx = self.scene.env_idx
        terms = {}
        for name, tensor in self.env.rew_buf_per_type.items():
            try:
                terms[name] = float(tensor[env_idx])
            except (IndexError, TypeError):
                continue

        self.reward_plotter.update(terms)

    def cleanup(self) -> None:
        if self.reward_plotter:
            self.reward_plotter.cleanup()


@dataclass
class ViserDebugOverlays:
    """Manage debug visualization queueing."""

    env: _EnvProtocol
    scene: _SceneProtocol

    def queue(self) -> None:
        """Queue environment debug visualizers for the current frame."""
        self.scene.clear_debug()
        if hasattr(self.env, "update_visualizers"):
            self.env.update_visualizers(self.scene)

    def on_env_switch(self) -> None:
        """Reset debug visuals when switching environment."""
        self.scene.clear_debug()
