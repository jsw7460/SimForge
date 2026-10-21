"""The one definition of evaluation mode for a run config.

Evaluation measures the policy, not the training-time stochasticity
layered on top of the plant: observation noise is off on every group, and
every event term that randomizes or disturbs the plant is removed, whether
it fires per reset (``reset_dr``), on the global timer (``interval_dr``)
or per env on its own timer (``interval``, pushes and the like). The
``startup`` and ``reset`` terms stay: they place the robot and the task.

The offline evaluator, the in-training evaluation env and the eval-path
determinism gate all go through :func:`apply_eval_mode`, so an evaluation
means the same thing wherever it is run.
"""

from __future__ import annotations

from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.common_config_classes import disable_corruption
from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig

__all__ = ["EVAL_DISABLED_EVENT_MODES", "apply_eval_mode"]

# Event modes that inject training-only stochasticity; the other two
# (``startup``, ``reset``) set the scene up and are kept in evaluation.
EVAL_DISABLED_EVENT_MODES: tuple[str, ...] = ("interval", "interval_dr", "reset_dr")


def apply_eval_mode(cfgs, *, disable_noise: bool = True, disable_events: bool = True) -> None:
    """Turn a run config into its evaluation-mode form, in place.

    ``disable_noise`` sets ``enable_corruption = False`` on every
    observation group; ``disable_events`` removes every event term whose
    mode is in :data:`EVAL_DISABLED_EVENT_MODES`. Both default on; the
    switches exist for the in-training evaluation's runner flags.
    """
    if disable_noise:
        disable_corruption(cfgs.observation)
    if disable_events:
        event_cfg = cfgs.event
        for name, term in iter_terms(event_cfg, EventTermConfig).items():
            if term.mode in EVAL_DISABLED_EVENT_MODES:
                setattr(event_cfg, name, None)
