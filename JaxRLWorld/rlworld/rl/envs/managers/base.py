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

    def _resolve_term_selectors(self, func, params: dict) -> None:
        """Replace SceneEntitySelector entries in ``params`` with their
        resolved :class:`ResolvedEntity`, **in place**.

        Two cases:

        1. User supplied a selector in ``params`` (e.g.
           ``params["asset_cfg"] = SceneEntitySelector(...)``) — resolved
           via ``env.resolve_selector`` and swapped.
        2. The preset omitted ``asset_cfg`` but the term function declares
           a selector-valued default (``def f(env, asset_cfg=_DEFAULT_SELECTOR)``)
           — discovered via :func:`inspect.signature` on ``func`` (or its
           ``__init__`` for class-based terms) and injected.

        Mutating ``params`` is safe: ``ResolvedEntity`` holds only a name,
        index tensors and matched-name lists — no sim-native handle — so a
        config carrying it still deep-copies cleanly (the config is cloned
        for the eval env, which then re-uses these resolved indices because
        it's the same robot model).  Parameters whose value/default is not
        a :class:`SceneEntitySelector` are left untouched, so legacy terms
        (``entity_name="robot"``) keep working until migrated.
        """
        # Case 1: user-provided selectors.
        for key, value in list(params.items()):
            if isinstance(value, SceneEntitySelector):
                params[key] = self.env.resolve_selector(value)

        # Case 2: function-default selectors not supplied by the preset.
        target = func.__init__ if isinstance(func, type) else func
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            return
        for param_name, param in sig.parameters.items():
            if param_name in params:
                continue
            if isinstance(param.default, SceneEntitySelector):
                params[param_name] = self.env.resolve_selector(param.default)
