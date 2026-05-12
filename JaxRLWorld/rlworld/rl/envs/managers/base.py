from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class BaseManager:
    """Base class for all managers."""

    def __init__(self, env: World):
        self.env = env
        self.device = env.device

    @property
    def env_step_calls(self) -> int:
        """Number of step() calls on the parent environment."""
        return self.env._env_step_counter

    # ------------------------------------------------------------------ #
    #  Setup-time SceneEntitySelector resolution                          #
    # ------------------------------------------------------------------ #

    def _selector_overrides(self, func, params: dict) -> dict:
        """Compute the ``{param_name: ResolvedEntity}`` overrides for a term.

        Two cases handled:

        1. **User-provided selector** in ``params`` (e.g.
           ``params["asset_cfg"] = SceneEntitySelector(...)``) — resolved
           via ``env.resolve_selector``.
        2. **Function default selector** the preset did NOT override —
           discovered via :func:`inspect.signature` on ``func`` (or its
           ``__init__`` for class-based terms) and added so term functions
           can declare ``def f(env, asset_cfg = _DEFAULT_SELECTOR)``
           without every preset repeating the selector.

        Returns a (possibly empty) dict that callers merge over ``params``
        at term-invocation time: ``func(env, **{**params, **overrides})``.

        Crucially this does **not** mutate ``params`` — the config tree
        (which holds the same ``params`` dicts) stays free of
        :class:`ResolvedEntity` so it remains deep-copyable (ResolvedEntity
        carries sim-native ctypes-backed handles that cannot be pickled).
        Parameters whose value/default is **not** a
        :class:`SceneEntitySelector` are ignored, so legacy terms (e.g.
        ``entity_name="robot"``) keep working until migrated.
        """
        overrides: dict = {}

        # Case 1: user-provided selectors.
        for key, value in params.items():
            if isinstance(value, SceneEntitySelector):
                overrides[key] = self.env.resolve_selector(value)

        # Case 2: function-default selectors not supplied by the preset.
        target = func.__init__ if isinstance(func, type) else func
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            return overrides
        for param_name, param in sig.parameters.items():
            if param_name in params or param_name in overrides:
                continue
            if isinstance(param.default, SceneEntitySelector):
                overrides[param_name] = self.env.resolve_selector(param.default)
        return overrides
