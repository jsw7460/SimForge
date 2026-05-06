"""Curriculum manager — mjlab-style step-based schedule system.

Ported from ``mjlab/managers/curriculum_manager.py``. The manager holds
a dict of :class:`CurriculumTermConfig` entries; each entry's
``func(env, env_ids, **params)`` is called every environment step, and
the function may mutate *other* managers (reward, termination) in
place based on the current training progress.

The two built-in curriculum functions in
``rlworld.rl.envs.mdp.curriculums`` — :class:`reward_curriculum` and
:class:`termination_curriculum` — are the typical users: they look up
a target term via ``env.reward_manager.get_term_cfg(name)`` or
``env.termination_manager.get_term_cfg(name)`` once at init, then on
each step call ``_apply_stages`` to set ``weight`` / ``params``
fields according to the current ``env.env_step_counter``.

The curriculum manager itself is side-effect-only — its ``compute()``
return value is a dict of logged values (one per active term) that
callers can forward to wandb / training logs. No reward is produced
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.base_config import iter_terms
from rlworld.rl.configs.curriculums import (
    CurriculumManagerConfig,
    CurriculumTermConfig,
)
from rlworld.rl.envs.managers.base import BaseManager

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


class CurriculumManager(BaseManager):
    """Manages step-based curriculum updates for reward / termination terms.

    Discovers terms via :func:`iter_terms` on the config instance.
    Each term is either:
      - a plain callable ``func(env, env_ids, **params) -> dict``, or
      - a class ``func(env, cfg)`` whose ``__init__`` resolves targets
        once and whose ``__call__(env, env_ids, **params) -> dict`` is
        invoked each step (mjlab convention, matches
        :class:`reward_curriculum` / :class:`termination_curriculum`).

    Terms may optionally implement ``reset(env_ids)`` for stateful
    curriculum logic (e.g. per-env level tracking); stateful terms are
    registered in ``_class_term_cfgs`` and automatically reset when
    the env's reset hook fires.
    """

    def __init__(self, env: World, config: CurriculumManagerConfig):
        super().__init__(env=env)
        self.config = config

        self.curriculum_terms: dict[str, CurriculumTermConfig] = iter_terms(config, CurriculumTermConfig)

        # Instantiate class-based terms once with (env, cfg) so they can
        # resolve target term refs (e.g. reward_manager.get_term_cfg).
        self._instances: dict[str, object] = {}
        self._stateful_names: list[str] = []
        for name, term in self.curriculum_terms.items():
            func = term.resolved_func
            if isinstance(func, type):
                inst = func(env=self.env, cfg=term)
                self._instances[name] = inst
                if hasattr(inst, "reset") and callable(inst.reset):
                    self._stateful_names.append(name)

        # Last computed state (for logging).
        self._state: dict[str, dict] = {name: {} for name in self.curriculum_terms}

    @property
    def active_terms(self) -> list[str]:
        return list(self.curriculum_terms)

    @property
    def state(self) -> dict[str, dict]:
        """Snapshot of the last values applied by each curriculum term.

        Keys are ``{term_name}`` and values are dicts of field_name →
        scalar. Suitable for wandb logging with a ``Curriculum/`` prefix.
        """
        return self._state

    def compute(self, env_ids: torch.Tensor | None = None) -> dict[str, dict]:
        """Run every curriculum term once; returns a logging snapshot."""
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.device)

        snapshot: dict[str, dict] = {}
        for name, term in self.curriculum_terms.items():
            if name in self._instances:
                result = self._instances[name](self.env, env_ids, **term.params)
            else:
                result = term.resolved_func(self.env, env_ids, **term.params)
            if isinstance(result, dict):
                snapshot[name] = result
            else:
                snapshot[name] = {}
        self._state = snapshot
        return snapshot

    def reset(self, env_ids: torch.Tensor) -> None:
        """Forward reset to stateful curriculum terms."""
        for name in self._stateful_names:
            self._instances[name].reset(env_ids)

    def __str__(self) -> str:
        if not self.curriculum_terms:
            return "<CurriculumManager> (no active terms)"
        lines = [f"<CurriculumManager> {len(self.curriculum_terms)} active terms:"]
        for name in self.curriculum_terms:
            lines.append(f"  - {name}")
        return "\n".join(lines)
