from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from rlworld.rl.utils.resolve import resolve_callable


@dataclass
class CurriculumTermConfig:
    """Configuration for a curriculum term.

    Mirrors mjlab's ``CurriculumTermCfg``. The ``func`` may be:
      - A plain callable ``(env, env_ids, **params) -> dict`` that runs
        every :meth:`CurriculumManager.compute` call, or
      - A class whose ``__init__(env, cfg)`` resolves target term refs
        once and whose ``__call__(env, env_ids, **params) -> dict``
        runs every step.

    The returned dict is used for logging; it should contain scalar
    values that describe the current curriculum state (e.g.
    ``{"weight": -0.05}``).

    Args:
        func: Curriculum function or class (see above), or a
            ``"module.path:attr"`` string reference.
        params: Kwargs passed to ``func`` on every call.
    """

    func: Callable | str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_func(self) -> Callable:
        if callable(self.func):
            return self.func
        return resolve_callable(self.func)


@dataclass
class CurriculumManagerConfig:
    """Config discovered by :class:`CurriculumManager` via ``iter_terms``.

    Preset authors typically subclass this with named
    :class:`CurriculumTermConfig` attributes, one per curriculum term.
    """

    pass
