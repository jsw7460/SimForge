"""Lifecycle event system for declarative manager initialization ordering.

Instead of hard-coding the setup sequence in each Env subclass, managers
and components register for lifecycle events and get called in priority
order when the event is dispatched.

Example:
    lifecycle = LifecycleManager()

    # Register callbacks with priority (lower = earlier)
    lifecycle.on(LifecycleEvent.SCENE_BUILT, contact_mgr.register_sensors, order=10)
    lifecycle.on(LifecycleEvent.SCENE_BUILT, vis_mgr.setup, order=20)

    # Dispatch after scene is built
    lifecycle.dispatch(LifecycleEvent.SCENE_BUILT)
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Callable


class LifecycleEvent(Enum):
    """Events dispatched during environment initialization and runtime."""

    SCENE_BUILT = "scene_built"
    """Fired after scene_manager.build_scene() completes.
    At this point the physics world exists and entities are registered."""

    MANAGERS_READY = "managers_ready"
    """Fired after all managers have been created.
    Safe to perform cross-manager initialization here."""

    ENV_READY = "env_ready"
    """Fired after the environment is fully initialized,
    including startup events and any performance captures."""


class LifecycleManager:
    """Manages lifecycle event callbacks with priority ordering."""

    def __init__(self):
        self._callbacks: dict[LifecycleEvent, list[tuple[int, str, Callable]]] = defaultdict(list)
        self._dispatched: set[LifecycleEvent] = set()

    def on(
        self,
        event: LifecycleEvent,
        callback: Callable[[], None],
        order: int = 0,
        name: str | None = None,
    ) -> None:
        """Register a callback for a lifecycle event.

        Args:
            event: The lifecycle event to listen for.
            callback: Zero-argument callable invoked when the event fires.
            order: Priority — lower values run first. Default 0.
            name: Optional label for debugging / logging.
        """
        label = name or getattr(callback, "__qualname__", str(callback))
        self._callbacks[event].append((order, label, callback))

    def dispatch(self, event: LifecycleEvent) -> None:
        """Fire a lifecycle event, invoking callbacks in priority order."""
        self._dispatched.add(event)
        for _, _, cb in sorted(self._callbacks[event], key=lambda x: x[0]):
            cb()

    def was_dispatched(self, event: LifecycleEvent) -> bool:
        """Check whether an event has already been dispatched."""
        return event in self._dispatched

    def reset(self) -> None:
        """Clear all callbacks and dispatch history."""
        self._callbacks.clear()
        self._dispatched.clear()
