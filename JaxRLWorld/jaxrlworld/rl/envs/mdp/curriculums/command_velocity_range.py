"""Curriculum term driving the adaptive velocity command's range.

Runs on every reset batch (the curriculum manager fires in
``_reset_idx``) and asks the :class:`AdaptiveVelocityCommandTerm` to
widen its active ``lin_vel_x`` range when the frontier bin is tracked
reliably. The expansion decision itself is fully on-device; the
returned logging dict (``Curriculum/<term>/...`` in wandb) is refreshed
only every ``log_interval`` calls to avoid per-reset host syncs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.envs.mdp.commands.adaptive_velocity import AdaptiveVelocityCommandTerm

if TYPE_CHECKING:
    from jaxrlworld.rl.configs.curriculums.curriculum_term_config import (
        CurriculumTermConfig,
    )
    from jaxrlworld.rl.envs.world import World


class command_velocity_range:
    """Performance-gated widening of the adaptive velocity range.

    Params (via :class:`CurriculumTermConfig.params`):
        command_name: Name of the command term (required at init).
        promote_threshold: Frontier-bin success EMA needed to expand.
        min_tours: Minimum tours the frontier bin needs before its EMA
            is trusted.
        log_interval: Refresh the wandb snapshot every N calls.
    """

    def __init__(self, env: World, cfg: CurriculumTermConfig):
        term = env.command_manager.get_term(cfg.params["command_name"])
        if not isinstance(term, AdaptiveVelocityCommandTerm):
            raise TypeError(
                f"command term {cfg.params['command_name']!r} is "
                f"{type(term).__name__}, expected AdaptiveVelocityCommandTerm"
            )
        self._term = term
        self._calls = 0
        self._snapshot: dict[str, float] = {}

    def __call__(
        self,
        env: World,
        env_ids: torch.Tensor,
        command_name: str,
        promote_threshold: float = 0.8,
        min_tours: float = 300.0,
        log_interval: int = 25,
    ) -> dict[str, float]:
        term = self._term
        term.maybe_expand(promote_threshold, min_tours)
        self._calls += 1
        if not self._snapshot or self._calls % log_interval == 0:
            self._snapshot = {
                "vx_max": float(term.active_max),
                "vx_min": float(term.active_min),
                "frontier_hi_success": float(term.frontier_hi_success),
                "frontier_lo_success": float(term.frontier_lo_success),
            }
        return dict(self._snapshot)
