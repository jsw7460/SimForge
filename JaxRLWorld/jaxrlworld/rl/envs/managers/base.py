from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING

from jaxrlworld.rl.configs.scene.entity_selector import SceneEntitySelector

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import World


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
    #  Term ownership and setup-time SceneEntitySelector resolution       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _own_terms(terms: dict) -> dict:
        """Per-manager copies of the config's term objects.

        Named terms are class attributes of their config, so every instance
        of that config class, and every ``deepcopy`` of it (the in-training
        eval config), shares one term object per name. A manager writes
        into its terms, resolving selectors into ``params`` and, through
        ``get_term_cfg``, letting a curriculum move weights and params, so
        it works on copies: the term is shallow-copied and given its own
        ``params`` dict. The config keeps the declarative term untouched,
        and a second env built from the same config resolves afresh.
        """
        owned = {}
        for name, term in terms.items():
            own = copy.copy(term)
            own.params = dict(term.params)
            owned[name] = own
        return owned

    def _resolve_term_selectors(self, func, params: dict) -> None:
        """Replace SceneEntitySelector entries in ``params`` with their
        resolved :class:`ResolvedEntity`, **in place**.

        ``params`` is a manager-owned dict (see :meth:`_own_terms`), never
        the config's. Two cases:

        1. User supplied a selector in ``params`` (e.g.
           ``params["asset_cfg"] = SceneEntitySelector(...)``) — resolved
           via ``env.resolve_selector`` and swapped.
        2. The preset omitted ``asset_cfg`` but the term function declares
           a selector-valued default (``def f(env, asset_cfg=_DEFAULT_SELECTOR)``)
           — discovered via :func:`inspect.signature` on ``func`` (or its
           ``__init__`` for class-based terms) and injected.

        Parameters whose value/default is not a :class:`SceneEntitySelector`
        are left untouched, so legacy terms (``entity_name="robot"``) keep
        working until migrated.
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
