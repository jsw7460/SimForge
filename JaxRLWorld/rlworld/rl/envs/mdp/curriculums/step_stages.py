"""Step-based curriculum functions — port of mjlab's reward_curriculum /
termination_curriculum.

Each function is implemented as a class so it can resolve the target
term reference (via ``env.reward_manager.get_term_cfg`` /
``env.termination_manager.get_term_cfg``) once at construction time.
The ``__call__`` then runs every
:meth:`CurriculumManager.compute` invocation, reading
``env.env_step_counter`` and applying whichever stages have fired.

Stage schema (dict):
    {
        "step":   int,            # required — env.step() threshold
        "weight": float,          # reward curriculum: weight override
        "params": dict[str, Any], # param overrides (reward or termination)
    }

See ``mjlab/envs/mdp/curriculums.py`` for the reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from rlworld.rl.configs.curriculums.curriculum_term_config import (
        CurriculumTermConfig,
    )
    from rlworld.rl.envs.world import World


# ── Stage type schemas ──────────────────────────────────────────────


class _RewardCurriculumStageOptional(TypedDict, total=False):
    weight: float
    params: dict[str, Any]


class RewardCurriculumStage(_RewardCurriculumStageOptional):
    step: int


class _TerminationCurriculumStageOptional(TypedDict, total=False):
    params: dict[str, Any]
    time_out: bool


class TerminationCurriculumStage(_TerminationCurriculumStageOptional):
    step: int


# ── Shared validation / apply engine ────────────────────────────────


_RESERVED_KEYS = {"step", "params"}


def _validate_stages(
    term_cfg: Any,
    term_name: str,
    stages: Sequence[Any],
) -> None:
    """Check stage ordering, field existence, and param keys."""
    for i in range(1, len(stages)):
        if stages[i]["step"] < stages[i - 1]["step"]:
            raise ValueError(
                f"Curriculum stages must be in nondecreasing step order,"
                f" but stage {i} has step"
                f" {stages[i]['step']} < {stages[i - 1]['step']}."
            )
    for stage in stages:
        for key in stage:
            if key not in _RESERVED_KEYS and not hasattr(term_cfg, key):
                raise AttributeError(f"Field {key!r} does not exist on the resolved term config for {term_name!r}.")
    for stage in stages:
        unknown = stage.get("params", {}).keys() - term_cfg.params.keys()
        if unknown:
            raise KeyError(
                f"Stage at step {stage['step']} sets unknown param(s) {unknown} on term {term_name!r}. Check for typos."
            )


def _apply_stages(
    term_cfg: Any,
    step_counter: int,
    stages: Sequence[Any],
) -> dict[str, Any]:
    """Apply every stage whose ``step`` threshold has been crossed.

    Later stages overwrite earlier ones, so the highest-step stage that
    has fired wins for each (field, param) key. Returns a dict of the
    *currently effective* values — one entry per distinct field/param
    name referenced by any stage — suitable for logging.
    """
    for stage in stages:
        if step_counter >= stage["step"]:
            for key, value in stage.items():
                if key not in _RESERVED_KEYS:
                    setattr(term_cfg, key, value)
            if "params" in stage:
                term_cfg.params.update(stage["params"])

    # Collect referenced field/param names for logging snapshot.
    logged_fields: set[str] = set()
    logged_params: set[str] = set()
    for stage in stages:
        for key in stage:
            if key not in _RESERVED_KEYS:
                logged_fields.add(key)
        for key in stage.get("params", {}):
            logged_params.add(key)
    result: dict[str, Any] = {}
    for key in logged_fields:
        result[key] = getattr(term_cfg, key)
    for key in logged_params:
        result[key] = term_cfg.params[key]
    return result


# ── Public wrappers (class-style; CurriculumManager instantiates with
#     (env, cfg) and then calls with (env, env_ids, **params)) ──────


class reward_curriculum:
    """Update a reward term's weight / params based on training steps.

    Mirrors ``mjlab.envs.mdp.reward_curriculum``. Resolves the target
    :class:`RewardTermConfig` once at init via
    ``env.reward_manager.get_term_cfg``, then applies stages on each
    :meth:`CurriculumManager.compute` using ``env.env_step_counter``.

    Example::

        CurriculumTermConfig(
            func=reward_curriculum,
            params={
                "reward_name": "raw_action_rate_l2",
                "stages": [
                    {"step": 0,     "weight":  0.01},
                    {"step": 14400, "weight":  0.05},
                    {"step": 28800, "weight":  0.10},
                ],
            },
        )
    """

    __name__ = "reward_curriculum"

    def __init__(self, env: World, cfg: CurriculumTermConfig) -> None:
        reward_name: str = cfg.params["reward_name"]
        stages: list[RewardCurriculumStage] = cfg.params["stages"]
        self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
        self._stages = stages
        _validate_stages(self._term_cfg, reward_name, self._stages)

    def __call__(
        self,
        env: World,
        env_ids,
        reward_name: str,
        stages: list[RewardCurriculumStage],
    ) -> dict[str, Any]:
        del env_ids, reward_name, stages
        return _apply_stages(self._term_cfg, env.env_step_counter, self._stages)


class termination_curriculum:
    """Update a termination term's params based on training steps.

    Mirrors ``mjlab.envs.mdp.termination_curriculum``. Resolves the
    target :class:`TerminationTermConfig` once at init via
    ``env.termination_manager.get_term_cfg``, then applies stages on
    each :meth:`CurriculumManager.compute` using
    ``env.env_step_counter``.

    Example::

        CurriculumTermConfig(
            func=termination_curriculum,
            params={
                "termination_name": "energy",
                "stages": [
                    {"step": 21600, "params": {"threshold": 3000.0}},
                    {"step": 28800, "params": {"threshold": 2000.0}},
                ],
            },
        )
    """

    __name__ = "termination_curriculum"

    def __init__(self, env: World, cfg: CurriculumTermConfig) -> None:
        termination_name: str = cfg.params["termination_name"]
        stages: list[TerminationCurriculumStage] = cfg.params["stages"]
        self._term_cfg = env.termination_manager.get_term_cfg(termination_name)
        self._stages = stages
        _validate_stages(self._term_cfg, termination_name, self._stages)

    def __call__(
        self,
        env: World,
        env_ids,
        termination_name: str,
        stages: list[TerminationCurriculumStage],
    ) -> dict[str, Any]:
        del env_ids, termination_name, stages
        return _apply_stages(self._term_cfg, env.env_step_counter, self._stages)
