from __future__ import annotations

import inspect
from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.common_config_classes import EventConfig
from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig
from jaxrlworld.rl.envs.managers.base import BaseManager
from jaxrlworld.rl.envs.mdp.events.keyed_sampler import (
    STAGE_IDS,
    STAGE_INTERVAL_TIMER,
    TermSampler,
)

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import World

# Backward-compatible alias
EventManagerConfig = EventConfig


class EventManager(BaseManager):
    """Manages event execution for domain randomization and disturbances.

    Terms are discovered via :func:`iter_terms` on the ``EventConfig`` instance.

    Randomness
    ----------
    A term that declares a ``sampler`` parameter receives a
    :class:`TermSampler` bound to ``(env.seed, term name, stage, counter)``,
    where the counter is this manager's per-env episode number for the
    reset stages, the term's per-env trigger count for ``interval``, and
    the global fire count for ``interval_dr``. Every draw is then a pure
    function of those coordinates (see ``keyed_sampler``), so the values an
    environment receives do not depend on the number of environments, on
    which others reset with it, or on the other terms. The interval timers
    are drawn the same way.
    """

    def __init__(self, env: World, config: EventConfig):
        super().__init__(env=env)
        self.config = config
        self.num_envs = env.num_envs

        # Discover named terms and resolve callables
        self._all_terms: dict[str, EventTermConfig] = self._own_terms(iter_terms(config, EventTermConfig))
        self._resolved_fns: dict[str, callable] = {name: term.resolved_func for name, term in self._all_terms.items()}
        # Terms opt into keyed randomness by taking ``sampler``; a term with
        # no random draw (a fixed-value setter) simply does not declare it.
        self._takes_sampler: dict[str, bool] = {
            name: "sampler" in inspect.signature(fn).parameters for name, fn in self._resolved_fns.items()
        }
        self._base_seed: int = int(env.seed)
        # Per-env episode number: 0 until the env's first reset, then the
        # count of resets it has been through. Advanced in :meth:`reset`,
        # which the world calls before the reset events of that reset.
        self._episode_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._startup_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Replace SceneEntitySelector params with their resolved ResolvedEntity.
        for term in self._all_terms.values():
            self._resolve_term_selectors(term.resolved_func, term.params)

        # Group by mode
        self._terms_by_mode: dict[str, list[tuple[str, EventTermConfig]]] = defaultdict(list)
        for name, term in self._all_terms.items():
            self._terms_by_mode[term.mode].append((name, term))

        # Set up interval timers. Each interval term has a per-env trigger
        # count: it keys the term's own draws and its next timer, so every
        # firing of an env draws fresh values while staying reproducible.
        self._interval_timers: dict[int, torch.Tensor] = {}
        self._interval_ranges: dict[int, tuple[float, float]] = {}
        self._interval_names: dict[int, str] = {}
        self._interval_fire_count: dict[int, torch.Tensor] = {}

        for local_idx, (name, term) in enumerate(self._terms_by_mode.get("interval", [])):
            if term.interval_range_s is None:
                raise ValueError(f"Interval event term '{name}' must have interval_range_s specified.")
            self._interval_ranges[local_idx] = term.interval_range_s
            self._interval_names[local_idx] = name
            self._interval_fire_count[local_idx] = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._interval_timers[local_idx] = self._sample_interval(
                local_idx, torch.arange(self.num_envs, device=self.device)
            )
        self._interval_dr_fire_count: int = 0

        # interval_dr: one GLOBAL timer shared by all interval_dr terms, so they
        # fire together (single deferred recompute per period). All such terms
        # must declare the same period. Timer starts at 0 -> fires on the first
        # step, seeding the initial DR draw.
        self._interval_dr_period: float | None = None
        self._interval_dr_timer: float = 0.0
        _interval_dr_terms = self._terms_by_mode.get("interval_dr", [])
        if _interval_dr_terms:
            periods = {t.interval_dr_period_s for _, t in _interval_dr_terms}
            if None in periods:
                raise ValueError("interval_dr terms require interval_dr_period_s.")
            if len(periods) != 1:
                raise ValueError(f"all interval_dr terms must share one period; got {sorted(periods)}.")
            self._interval_dr_period = periods.pop()

    @property
    def available_modes(self) -> list[str]:
        return list(self._terms_by_mode.keys())

    def apply(self, mode: str, env_ids: torch.Tensor | None = None, dt: float | None = None) -> None:
        if mode not in self._terms_by_mode:
            return

        if mode == "startup":
            self._apply_startup()
        elif mode == "reset":
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self._apply_reset(env_ids)
        elif mode == "reset_dr":
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            self._apply_reset_dr(env_ids)
        elif mode == "interval":
            if dt is None:
                raise ValueError("dt must be provided for interval mode")
            self._apply_interval(dt)
        elif mode == "interval_dr":
            if dt is None:
                raise ValueError("dt must be provided for interval_dr mode")
            self._apply_interval_dr(dt)

    @property
    def episode_count(self) -> torch.Tensor:
        """Per-env number of resets so far (``long``, on the device)."""
        return self._episode_count

    def reset(self, env_ids: torch.Tensor) -> None:
        self._episode_count[env_ids] += 1
        for idx in self._interval_timers:
            self._interval_fire_count[idx][env_ids] += 1
            self._interval_timers[idx][env_ids] = self._sample_interval(idx, env_ids)

    def _sampler(self, name: str, stage: str, counter: torch.Tensor) -> TermSampler:
        return TermSampler.bind(base_seed=self._base_seed, term_name=name, stage=STAGE_IDS[stage], counter=counter)

    def sampler_for(self, term_name: str, mode: str) -> TermSampler:
        """A sampler bound as the manager would bind it for a term named
        ``term_name`` running in ``mode``, for code that calls a term outside
        the manager (diagnostics, scripted resets). Reset modes key on the
        episode counter, ``interval_dr`` on its fire count, ``startup`` and
        ``interval`` on zeros."""
        if mode in ("reset", "reset_dr"):
            counter = self._episode_count
        elif mode == "interval_dr":
            counter = torch.full((self.num_envs,), self._interval_dr_fire_count, dtype=torch.long, device=self.device)
        elif mode in ("startup", "interval"):
            counter = self._startup_counter
        else:
            raise ValueError(f"unknown event mode {mode!r}")
        return self._sampler(term_name, mode, counter)

    def run_term(self, name: str, env_ids: torch.Tensor) -> None:
        """Run one registered term on ``env_ids`` with the sampler its mode
        implies, outside the mode's batch (no deferred recompute flush)."""
        term = self._all_terms[name]
        self._call_event_fn(name, term, env_ids, self.sampler_for(name, term.mode))

    def _call_event_fn(
        self, name: str, term: EventTermConfig, env_ids: torch.Tensor, sampler: TermSampler | None = None
    ) -> None:
        if self._takes_sampler[name]:
            if sampler is None:
                raise ValueError(f"event term {name!r} takes a sampler but none was bound for this call")
            self._resolved_fns[name](env=self.env, env_ids=env_ids, sampler=sampler, **term.params)
        else:
            self._resolved_fns[name](env=self.env, env_ids=env_ids, **term.params)

    def _apply_startup(self) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device)
        for name, term in self._terms_by_mode["startup"]:
            self._call_event_fn(name, term, env_ids, self._sampler(name, "startup", self._startup_counter))

    def _apply_reset(self, env_ids: torch.Tensor) -> None:
        for name, term in self._terms_by_mode["reset"]:
            self._call_event_fn(name, term, env_ids, self._sampler(name, "reset", self._episode_count))

    def _run_dr_batch(self, terms, env_ids: torch.Tensor, stage: str, counter: torch.Tensor) -> None:
        """Run a batch of DR terms with a SINGLE deferred recompute/notify flush.

        Newton: each ``solver.notify_model_changed`` triggers a non-trivial GPU
        model-state refresh; when multiple DR terms fire together (body inertia +
        joint friction + foot friction), per-term notifies stack (~5 ms each). The
        Newton DR backends OR their flag into ``env._dr_pending_notify_flags``
        instead of notifying immediately; we flush one combined notify after all
        terms. No-op for Genesis (its backends never touch the attributes).

        MuJoCo: symmetric. ``sim.recompute_constants`` is model-global and its
        most expensive level (set_const, for body_mass) dominates the reset path
        when several body_mass/body_com terms fire together. The MuJoCo DR
        backends defer their recompute into ``env._dr_pending_recompute_level``
        (keeping the MAX level — RecomputeLevel is an ordered IntEnum whose higher
        levels are a superset), and we flush ONE recompute after all terms.

        Shared by reset_dr (per-reset, env_ids = the reset envs) and interval_dr
        (global period, env_ids = all envs).
        """
        is_newton = self.env.sim_type == "newton"
        is_mujoco = self.env.sim_type == "mujoco"
        if is_newton:
            self.env._dr_pending_notify_flags = 0
        if is_mujoco:
            self.env._dr_pending_recompute_level = None
        for name, term in terms:
            self._call_event_fn(name, term, env_ids, self._sampler(name, stage, counter))
        if is_newton:
            flags = self.env._dr_pending_notify_flags
            del self.env._dr_pending_notify_flags
            if flags:
                self.env.scene_manager.solver.notify_model_changed(flags)
        if is_mujoco:
            level = self.env._dr_pending_recompute_level
            del self.env._dr_pending_recompute_level
            if level:  # None (no recompute term) or none(0) -> skip
                self.env.scene_manager.sim.recompute_constants(level)

    def _apply_reset_dr(self, env_ids: torch.Tensor) -> None:
        self._run_dr_batch(self._terms_by_mode["reset_dr"], env_ids, "reset_dr", self._episode_count)

    def _apply_interval_dr(self, dt: float) -> None:
        """Global-interval DR: all interval_dr terms fire together on one shared
        timer every ``interval_dr_period_s`` seconds, for ALL envs, with a single
        deferred recompute. Unlike per-episode reset_dr (a recompute per reset),
        this amortizes the recompute to one flush per period."""
        self._interval_dr_timer -= dt
        if self._interval_dr_timer > 0.0:
            return
        # Carry the remainder so the period doesn't drift; guard against a period
        # shorter than dt (would fire every step anyway).
        self._interval_dr_timer = max(self._interval_dr_timer + self._interval_dr_period, 0.0)
        self._interval_dr_fire_count += 1
        all_ids = torch.arange(self.num_envs, device=self.device)
        counter = torch.full((self.num_envs,), self._interval_dr_fire_count, dtype=torch.long, device=self.device)
        self._run_dr_batch(self._terms_by_mode["interval_dr"], all_ids, "interval_dr", counter)

    def _apply_interval(self, dt: float) -> None:
        for local_idx, (name, term) in enumerate(self._terms_by_mode["interval"]):
            self._interval_timers[local_idx] -= dt
            triggered_mask = self._interval_timers[local_idx] <= 0
            triggered_env_ids = triggered_mask.nonzero(as_tuple=False).flatten()

            if len(triggered_env_ids) > 0:
                fire_count = self._interval_fire_count[local_idx]
                fire_count[triggered_env_ids] += 1
                self._call_event_fn(name, term, triggered_env_ids, self._sampler(name, "interval", fire_count))
                self._interval_timers[local_idx][triggered_env_ids] = self._sample_interval(
                    local_idx, triggered_env_ids
                )

    def _sample_interval(self, term_idx: int, env_ids: torch.Tensor) -> torch.Tensor:
        """The next wait of each env in ``env_ids`` for one interval term, keyed
        on the term's trigger count so each firing draws a fresh, reproducible
        interval."""
        min_interval, max_interval = self._interval_ranges[term_idx]
        sampler = TermSampler.bind(
            base_seed=self._base_seed,
            term_name=self._interval_names[term_idx],
            stage=STAGE_INTERVAL_TIMER,
            counter=self._interval_fire_count[term_idx],
        )
        return sampler.draw(min_interval, max_interval, env_ids)

    def __str__(self) -> str:
        """Pretty print event manager configuration."""
        from jaxrlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._all_terms:
            return ""

        rows = []
        for idx, (name, term) in enumerate(self._all_terms.items()):
            func_name = getattr(self._resolved_fns[name], "__name__", name)
            mode_str = term.mode.capitalize()

            interval_str = "-"
            if term.interval_range_s is not None:
                min_t, max_t = term.interval_range_s
                interval_str = f"{min_t}-{max_t}s"

            params_str = "-"
            if term.params:
                param_items = [f"{k}={v}" for k, v in list(term.params.items())[:2]]
                params_str = ", ".join(param_items)
                if len(term.params) > 2:
                    params_str += ", ..."

            rows.append([idx, func_name, mode_str, interval_str, params_str])

        mode_counts = {mode: len(terms) for mode, terms in self._terms_by_mode.items()}
        footer = ", ".join(f"{mode}: {count}" for mode, count in mode_counts.items())

        table = create_manager_table(
            title="Event Terms", columns=["Idx", "Name", "Mode", "Interval", "Params"], rows=rows, footer=footer
        )
        return table_to_string(table)
