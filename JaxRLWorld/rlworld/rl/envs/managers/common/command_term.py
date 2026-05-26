"""CommandTerm base class and built-in implementations.

A CommandTerm encapsulates a group of related commands with their own
sampling logic, resampling timer, and per-step post-processing.

External control (e.g. an interactive viewer driving sliders) is
supported via :meth:`CommandTerm.set_command` / :meth:`release_command`,
which take an optional ``columns=`` selector. The base class enforces
column-wise locking: locked entries of the command tensor are
preserved across automatic resampling AND post-processing. Unlocked
columns of the same env are free to evolve normally — so e.g. locking
``lin_vel_x`` does not interfere with the velocity term's heading
P-control loop that drives ``ang_vel``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.common.command_ui import (
    CommandTermUISpec,
    GroupControl,
    PresetButton,
    SliderControl,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return (angles + math.pi) % (2 * math.pi) - math.pi


# ──────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────


@dataclass
class CommandTermCfg(ABC):
    """Configuration for a CommandTerm.

    Each subclass defines its own fields (ranges, flags, etc.)
    and implements ``build()`` to construct the corresponding CommandTerm.
    """

    resampling_time_range: tuple[float, float] = (5.0, 10.0)

    @abstractmethod
    def build(self, env: World) -> CommandTerm: ...


class CommandTerm(ABC):
    """Abstract base for a group of related commands.

    Subclasses must implement:
        ``command`` (property):      Return the command tensor [num_envs, dim].
        ``_resample_command(ids)``:  Sample new commands for given env ids.

    Optionally override:
        ``_update_command()``:       Per-step post-processing (e.g., heading control).
        ``column_names``:            Tuple of names for each column (for attribute access).
        ``get_ui_spec()``:           Declarative UI for interactive viewers.
    """

    column_names: tuple[str, ...] = ()

    def __init__(self, env: World, cfg: CommandTermCfg):
        self._env = env
        self.cfg = cfg
        self.num_envs = env.num_envs
        self.device = env.device
        self.time_left = torch.zeros(self.num_envs, device=self.device)

        # Column-wise external-control mask: ``[num_envs, command_dim]``.
        # Allocated lazily in ``__init_command_buffer`` once the subclass
        # has set ``self._command`` (we need its width). Subclasses MUST
        # call :meth:`_init_external_control_mask` after allocating
        # ``self._command``.
        self._externally_controlled: torch.Tensor | None = None

    # ── Subclass helpers ───────────────────────────────────────────

    def _init_external_control_mask(self) -> None:
        """Allocate the column-wise lock mask.

        Subclasses must call this once ``self._command`` is allocated
        (i.e. at the end of their ``__init__``). Kept as an explicit
        step rather than wiring it into the base ``__init__`` because
        subclasses choose their own command_dim.
        """
        assert hasattr(
            self, "_command"
        ), "Subclass must assign self._command before calling _init_external_control_mask()"
        cmd_dim = self._command.shape[1]
        self._externally_controlled = torch.zeros(self.num_envs, cmd_dim, dtype=torch.bool, device=self.device)

    def _resolve_columns(self, columns: tuple[str, ...] | list[str] | None) -> torch.Tensor:
        """Resolve a column-name selector to a long-tensor of indices.

        ``None`` resolves to all columns (legacy whole-row semantics).
        """
        if columns is None:
            return torch.arange(self._command.shape[1], device=self.device)
        idx = []
        for name in columns:
            try:
                idx.append(self.column_names.index(name))
            except ValueError as e:
                raise KeyError(
                    f"Column {name!r} is not declared by {type(self).__name__}.column_names "
                    f"(got {self.column_names!r})"
                ) from e
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    # ── Required subclass surface ──────────────────────────────────

    @property
    @abstractmethod
    def command(self) -> torch.Tensor:
        """Command tensor of shape [num_envs, command_dim]."""
        ...

    @abstractmethod
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample new commands for the given environment indices."""
        ...

    def _update_command(self) -> None:
        """Per-step post-processing. Override if needed.

        Subclasses that overwrite specific columns (e.g. heading
        P-control writing ``ang_vel``) must consult
        ``self._externally_controlled`` for those columns and skip
        envs whose target column is locked. See
        :class:`VelocityCommandTerm._update_command` for the canonical
        pattern.
        """

    def get_ui_spec(self) -> CommandTermUISpec | None:
        """Declarative UI for an interactive viewer.

        Returning ``None`` (the default) means the term exposes no
        interactive knobs. Subclasses with tunable parameters should
        override this and return a :class:`CommandTermUISpec`. Slider
        column names must match entries in ``column_names``.
        """
        return None

    # ── Driver loop ────────────────────────────────────────────────

    def compute(self, dt: float) -> None:
        """Advance timer, resample if expired, then post-process.

        Locked entries (``_externally_controlled``) are protected
        across both the resample and the post-processing step via a
        clone-and-restore around them. The clone is skipped when no
        env has any locked column, so the training-time fast path is
        bit-identical to the pre-refactor code.
        """
        self.time_left -= dt
        resample_ids = (self.time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self.time_left[resample_ids] = torch.empty(len(resample_ids), device=self.device).uniform_(
                *self.cfg.resampling_time_range
            )
            self._resample_command(resample_ids)

        if self._externally_controlled is not None and self._externally_controlled.any():
            # Capture the user-driven values; restore after the
            # subclass's post-processing so e.g. heading P-control
            # never overwrites a locked column.
            locked_snapshot = self._command[self._externally_controlled]
            self._update_command()
            self._command[self._externally_controlled] = locked_snapshot
        else:
            self._update_command()

    # ── External control ──────────────────────────────────────────

    def set_command(
        self,
        env_ids: torch.Tensor,
        values: torch.Tensor,
        columns: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Override (env, column) entries of the command tensor.

        Writes ``values`` into ``self._command`` at the cross-product of
        ``env_ids`` and ``columns``, and marks those entries as
        externally controlled. Locked entries are preserved across
        future :meth:`compute` calls (both automatic resampling and
        :meth:`_update_command` post-processing) until released.

        Args:
            env_ids: Environment indices to override.
            values: Either ``(len(env_ids), len(columns))`` for a
                column-selective override, or ``(len(env_ids), command_dim)``
                when ``columns is None`` (whole-row override; legacy
                shape).
            columns: Column names (from ``column_names``) to override.
                ``None`` (default) means override every column — same as
                the pre-refactor behavior.
        """
        if self._externally_controlled is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support external command override "
                "(no `_command` buffer registered via `_init_external_control_mask()`)."
            )
        col_idx = self._resolve_columns(columns)
        # Index-assign: _command[env_ids[:, None], col_idx[None, :]] = values.
        self._command[env_ids[:, None], col_idx[None, :]] = values
        self._externally_controlled[env_ids[:, None], col_idx[None, :]] = True

    def release_command(
        self,
        env_ids: torch.Tensor,
        columns: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Release external control for the given (env, column) entries.

        Affected entries become eligible for automatic resampling and
        post-processing again. The term's resampling timer is NOT reset
        here; the next natural resample (or :meth:`reset`) refreshes
        them.

        Args:
            env_ids: Environment indices to release.
            columns: Column names to release. ``None`` releases every
                column.
        """
        if self._externally_controlled is None:
            # Term does not support external control — nothing to release.
            return
        col_idx = self._resolve_columns(columns)
        self._externally_controlled[env_ids[:, None], col_idx[None, :]] = False

    def reset(self, env_ids: torch.Tensor) -> None:
        """Force resample for the given environments.

        Also clears external-control state for ALL columns of those
        envs so that they resume normal auto-resampling after an
        episode reset. Terms that do not support external control
        (no ``_externally_controlled`` mask allocated) skip the clear.
        """
        if self._externally_controlled is not None:
            self._externally_controlled[env_ids] = False
        self.time_left[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.resampling_time_range
        )
        self._resample_command(env_ids)


# ──────────────────────────────────────────────
# VelocityCommandTerm
# ──────────────────────────────────────────────


@dataclass
class VelocityCommandTermCfg(CommandTermCfg):
    """Configuration for uniform velocity command sampling.

    Samples (lin_vel_x, lin_vel_y, ang_vel) uniformly from configured ranges.
    Optionally applies heading P-control and standing-env zeroing.
    """

    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-1.0, 1.0)
    ang_vel_range: tuple[float, float] = (-1.0, 1.0)

    rel_standing_envs: float = 0.0
    heading_command: bool = False
    heading_control_stiffness: float = 0.5
    heading_range: tuple[float, float] = (-3.14, 3.14)
    rel_heading_envs: float = 1.0

    def build(self, env: World) -> VelocityCommandTerm:
        return VelocityCommandTerm(env, self)


class VelocityCommandTerm(CommandTerm):
    """3-dim velocity command: [lin_vel_x, lin_vel_y, ang_vel]."""

    column_names = ("lin_vel_x", "lin_vel_y", "ang_vel")

    cfg: VelocityCommandTermCfg

    def __init__(self, env: World, cfg: VelocityCommandTermCfg):
        super().__init__(env, cfg)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self._init_external_control_mask()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def lin_vel_x(self) -> torch.Tensor:
        return self._command[:, 0]

    @property
    def lin_vel_y(self) -> torch.Tensor:
        return self._command[:, 1]

    @property
    def ang_vel(self) -> torch.Tensor:
        return self._command[:, 2]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self._command[env_ids, 0] = torch.empty(n, device=self.device).uniform_(*self.cfg.lin_vel_x_range)
        self._command[env_ids, 1] = torch.empty(n, device=self.device).uniform_(*self.cfg.lin_vel_y_range)
        self._command[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*self.cfg.ang_vel_range)
        if self.cfg.rel_standing_envs > 0.0:
            r = torch.rand(n, device=self.device)
            self.is_standing_env[env_ids] = r < self.cfg.rel_standing_envs

        if self.cfg.heading_command:
            r = torch.rand(n, device=self.device)
            self.is_heading_env[env_ids] = r < self.cfg.rel_heading_envs
            self.heading_target[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.heading_range)

    def _update_command(self) -> None:
        # Training fast path: when nothing is externally controlled,
        # bypass per-column gating to keep behavior bit-identical to
        # the pre-refactor code (and avoid a 3× write in standing
        # zeroing). The base ``compute()`` already short-circuits the
        # clone-and-restore in the same case.
        any_locked = self._externally_controlled.any()

        # Heading P-control overwrites column 2 (ang_vel). Skip envs
        # where the user has locked ang_vel via the viewer.
        if self.cfg.heading_command:
            if any_locked:
                ang_free = ~self._externally_controlled[:, 2]
                heading_ids = (self.is_heading_env & ang_free).nonzero(as_tuple=False).flatten()
            else:
                heading_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            if len(heading_ids) > 0:
                heading_w = self._env.heading_w
                heading_error = _wrap_to_pi(self.heading_target - heading_w)
                self._command[heading_ids, 2] = torch.clamp(
                    self.cfg.heading_control_stiffness * heading_error[heading_ids],
                    self.cfg.ang_vel_range[0],
                    self.cfg.ang_vel_range[1],
                )

        # Standing-env zeroing writes all three columns. In the
        # training fast path zero the whole row in one shot; only when
        # something is locked do we zero per-column so a partial lock
        # (e.g. lin_vel_x pinned, others free) keeps the user-driven
        # column intact.
        if self.cfg.rel_standing_envs > 0.0:
            if any_locked:
                for col in (0, 1, 2):
                    col_free = ~self._externally_controlled[:, col]
                    ids = (self.is_standing_env & col_free).nonzero(as_tuple=False).flatten()
                    if len(ids) > 0:
                        self._command[ids, col] = 0.0
            else:
                standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
                if len(standing_ids) > 0:
                    self._command[standing_ids] = 0.0

    def get_ui_spec(self) -> CommandTermUISpec:
        return CommandTermUISpec(
            section_label="Velocity Command",
            controls=(
                SliderControl(
                    column="lin_vel_x",
                    label="lin_vel_x",
                    low=self.cfg.lin_vel_x_range[0],
                    high=self.cfg.lin_vel_x_range[1],
                    step=0.05,
                    unit="m/s",
                ),
                SliderControl(
                    column="lin_vel_y",
                    label="lin_vel_y",
                    low=self.cfg.lin_vel_y_range[0],
                    high=self.cfg.lin_vel_y_range[1],
                    step=0.05,
                    unit="m/s",
                ),
                SliderControl(
                    column="ang_vel",
                    label="ang_vel_z",
                    low=self.cfg.ang_vel_range[0],
                    high=self.cfg.ang_vel_range[1],
                    step=0.05,
                    unit="rad/s",
                ),
            ),
        )


# ──────────────────────────────────────────────
# GaitCommandTerm
# ──────────────────────────────────────────────

# Column indices for the gait command tensor.
_GAIT_FREQ = 0
_GAIT_PHASE = 1
_GAIT_OFFSET = 2
_GAIT_BOUND = 3
_GAIT_DURATION = 4
_FOOTSWING_HEIGHT = 5
_BODY_HEIGHT = 6
_BODY_PITCH = 7
_BODY_ROLL = 8
_STANCE_WIDTH = 9
_STANCE_LENGTH = 10
_GAIT_DIM = 11


@dataclass
class GaitCommandTermCfg(CommandTermCfg):
    """Configuration for gait behavior command sampling.

    Produces an 11-dim behavior command matching Walk-These-Ways:
        [gait_freq, gait_phase, gait_offset, gait_bound, gait_duration,
         footswing_height, body_height, body_pitch, body_roll,
         stance_width, stance_length]

    Gait phase/offset/bound are sampled uniformly then post-processed
    according to the selected ``gait_category_mode``.
    """

    # Sampling ranges for each parameter.
    freq_range: tuple[float, float] = (2.0, 4.0)
    phase_range: tuple[float, float] = (0.0, 1.0)
    offset_range: tuple[float, float] = (0.0, 1.0)
    bound_range: tuple[float, float] = (0.0, 1.0)
    duration_range: tuple[float, float] = (0.5, 0.5)
    footswing_height_range: tuple[float, float] = (0.03, 0.35)
    body_height_range: tuple[float, float] = (-0.25, 0.15)
    body_pitch_range: tuple[float, float] = (-0.4, 0.4)
    body_roll_range: tuple[float, float] = (0.0, 0.0)
    stance_width_range: tuple[float, float] = (0.10, 0.45)
    stance_length_range: tuple[float, float] = (0.35, 0.45)

    # Gait category post-processing mode.
    #   "gaitwise":    Sample category (pronk/trot/pace/bound), constrain
    #                  phase/offset/bound to that category. Matches WTW
    #                  ``gaitwise_curricula`` mode.
    #   "exclusive":   Randomly zero out two of three phase offsets.
    #                  Matches WTW ``exclusive_phase_offset`` mode.
    #   "balanced":    25% each for pronk/trot/pace/bound.
    #                  Matches WTW ``balance_gait_distribution`` mode.
    #   "none":        No post-processing; phase/offset/bound are independent.
    gait_category_mode: str = "gaitwise"

    # Category names and their equal probability.
    # Used by "gaitwise" mode.
    categories: tuple[str, ...] = ("pronk", "trot", "pace", "bound")

    # Quantize phases to {0, 0.5} after category post-processing.
    # Matches WTW ``binary_phases``.
    binary_phases: bool = True

    def build(self, env: World) -> GaitCommandTerm:
        return GaitCommandTerm(env, self)


class GaitCommandTerm(CommandTerm):
    """11-dim gait behavior command.

    Sampling logic follows Walk-These-Ways (Margolis & Agrawal, CoRL 2022).
    Each resample:
        1. Sample all 11 dims uniformly from configured ranges.
        2. Apply gait category post-processing on phase/offset/bound.
        3. Optionally quantize phases to binary {0, 0.5}.
    """

    column_names = (
        "gait_freq",
        "gait_phase",
        "gait_offset",
        "gait_bound",
        "gait_duration",
        "footswing_height",
        "body_height",
        "body_pitch",
        "body_roll",
        "stance_width",
        "stance_length",
    )

    cfg: GaitCommandTermCfg

    def __init__(self, env: World, cfg: GaitCommandTermCfg):
        super().__init__(env, cfg)
        self._command = torch.zeros(self.num_envs, _GAIT_DIM, device=self.device)
        self._init_external_control_mask()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        cfg = self.cfg

        # 1. Uniform sampling for all parameters.
        self._command[env_ids, _GAIT_FREQ] = torch.empty(n, device=self.device).uniform_(*cfg.freq_range)
        self._command[env_ids, _GAIT_PHASE] = torch.empty(n, device=self.device).uniform_(*cfg.phase_range)
        self._command[env_ids, _GAIT_OFFSET] = torch.empty(n, device=self.device).uniform_(*cfg.offset_range)
        self._command[env_ids, _GAIT_BOUND] = torch.empty(n, device=self.device).uniform_(*cfg.bound_range)
        self._command[env_ids, _GAIT_DURATION] = torch.empty(n, device=self.device).uniform_(*cfg.duration_range)
        self._command[env_ids, _FOOTSWING_HEIGHT] = torch.empty(n, device=self.device).uniform_(
            *cfg.footswing_height_range
        )
        self._command[env_ids, _BODY_HEIGHT] = torch.empty(n, device=self.device).uniform_(*cfg.body_height_range)
        self._command[env_ids, _BODY_PITCH] = torch.empty(n, device=self.device).uniform_(*cfg.body_pitch_range)
        self._command[env_ids, _BODY_ROLL] = torch.empty(n, device=self.device).uniform_(*cfg.body_roll_range)
        self._command[env_ids, _STANCE_WIDTH] = torch.empty(n, device=self.device).uniform_(*cfg.stance_width_range)
        self._command[env_ids, _STANCE_LENGTH] = torch.empty(n, device=self.device).uniform_(*cfg.stance_length_range)

        # 2. Gait category post-processing on phase/offset/bound.
        self._apply_gait_categories(env_ids)

        # 3. Binary phase quantization.
        if cfg.binary_phases:
            for col in (_GAIT_PHASE, _GAIT_OFFSET, _GAIT_BOUND):
                raw = self._command[env_ids, col]
                self._command[env_ids, col] = (torch.round(2 * raw) / 2.0) % 1.0

    def _apply_gait_categories(self, env_ids: torch.Tensor) -> None:
        """Post-process phase/offset/bound based on gait category mode.

        Exactly replicates the Walk-These-Ways ``_resample_commands``
        gait category logic.
        """
        mode = self.cfg.gait_category_mode
        if mode == "none":
            return

        n = len(env_ids)
        rand = torch.rand(n, device=self.device)
        cmd = self._command

        if mode == "gaitwise":
            # Equal probability per category.
            cats = self.cfg.categories
            num_cats = len(cats)
            prob = 1.0 / num_cats

            for i, cat in enumerate(cats):
                mask = (prob * i <= rand) & (rand < prob * (i + 1))
                ids = env_ids[mask]
                if len(ids) == 0:
                    continue

                if cat == "pronk":
                    cmd[ids, _GAIT_PHASE] = (cmd[ids, _GAIT_PHASE] / 2 - 0.25) % 1
                    cmd[ids, _GAIT_OFFSET] = (cmd[ids, _GAIT_OFFSET] / 2 - 0.25) % 1
                    cmd[ids, _GAIT_BOUND] = (cmd[ids, _GAIT_BOUND] / 2 - 0.25) % 1
                elif cat == "trot":
                    cmd[ids, _GAIT_PHASE] = cmd[ids, _GAIT_PHASE] / 2 + 0.25
                    cmd[ids, _GAIT_OFFSET] = 0
                    cmd[ids, _GAIT_BOUND] = 0
                elif cat == "pace":
                    cmd[ids, _GAIT_PHASE] = 0
                    cmd[ids, _GAIT_OFFSET] = cmd[ids, _GAIT_OFFSET] / 2 + 0.25
                    cmd[ids, _GAIT_BOUND] = 0
                elif cat == "bound":
                    cmd[ids, _GAIT_PHASE] = 0
                    cmd[ids, _GAIT_OFFSET] = 0
                    cmd[ids, _GAIT_BOUND] = cmd[ids, _GAIT_BOUND] / 2 + 0.25

        elif mode == "exclusive":
            # Randomly zero out two of three offsets.
            trot = env_ids[rand < 0.34]
            pace = env_ids[(0.34 <= rand) & (rand < 0.67)]
            bound = env_ids[rand >= 0.67]
            cmd[pace, _GAIT_PHASE] = 0
            cmd[bound, _GAIT_PHASE] = 0
            cmd[trot, _GAIT_OFFSET] = 0
            cmd[bound, _GAIT_OFFSET] = 0
            cmd[trot, _GAIT_BOUND] = 0
            cmd[pace, _GAIT_BOUND] = 0

        elif mode == "balanced":
            # 25% each for pronk/trot/pace/bound.
            pronk = env_ids[rand <= 0.25]
            trot = env_ids[(0.25 < rand) & (rand <= 0.50)]
            pace = env_ids[(0.50 < rand) & (rand <= 0.75)]
            bound = env_ids[rand > 0.75]
            # Pronk: all ~0
            cmd[pronk, _GAIT_PHASE] = (cmd[pronk, _GAIT_PHASE] / 2 - 0.25) % 1
            cmd[pronk, _GAIT_OFFSET] = (cmd[pronk, _GAIT_OFFSET] / 2 - 0.25) % 1
            cmd[pronk, _GAIT_BOUND] = (cmd[pronk, _GAIT_BOUND] / 2 - 0.25) % 1
            # Trot: phase~0.5, offset=0, bound=0
            cmd[trot, _GAIT_OFFSET] = 0
            cmd[trot, _GAIT_BOUND] = 0
            cmd[trot, _GAIT_PHASE] = cmd[trot, _GAIT_PHASE] / 2 + 0.25
            # Pace: phase=0, offset~0.5, bound=0
            cmd[pace, _GAIT_PHASE] = 0
            cmd[pace, _GAIT_BOUND] = 0
            cmd[pace, _GAIT_OFFSET] = cmd[pace, _GAIT_OFFSET] / 2 + 0.25
            # Bound: phase=0, offset=0, bound~0.5
            cmd[bound, _GAIT_PHASE] = 0
            cmd[bound, _GAIT_OFFSET] = 0
            cmd[bound, _GAIT_BOUND] = cmd[bound, _GAIT_BOUND] / 2 + 0.25

    def get_ui_spec(self) -> CommandTermUISpec:
        cfg = self.cfg

        # Categories that match the WTW post-processing exactly.
        # Trot: phase=0.5, others 0. Pace: offset=0.5, others 0. Bound:
        # bound=0.5, others 0. Pronk: all 0. The viewer snaps these
        # values into the locked command columns; the user is then
        # free to drag individual sliders to deviate.
        preset_buttons = (
            PresetButton("Pronk", {"gait_phase": 0.0, "gait_offset": 0.0, "gait_bound": 0.0}),
            PresetButton("Trot", {"gait_phase": 0.5, "gait_offset": 0.0, "gait_bound": 0.0}),
            PresetButton("Pace", {"gait_phase": 0.0, "gait_offset": 0.5, "gait_bound": 0.0}),
            PresetButton("Bound", {"gait_phase": 0.0, "gait_offset": 0.0, "gait_bound": 0.5}),
        )

        # Step for phase columns: 0.5 when binary_phases enforces
        # {0, 0.5} during training so the slider snaps to legal values.
        phase_step = 0.5 if cfg.binary_phases else 0.05

        return CommandTermUISpec(
            section_label="Gait Command",
            controls=(
                GroupControl(
                    label="Timing",
                    children=(
                        SliderControl("gait_freq", "freq", *cfg.freq_range, step=0.1, unit="Hz"),
                        SliderControl("gait_phase", "phase", 0.0, 1.0, step=phase_step),
                        SliderControl("gait_offset", "offset", 0.0, 1.0, step=phase_step),
                        SliderControl("gait_bound", "bound", 0.0, 1.0, step=phase_step),
                        SliderControl("gait_duration", "duration", *cfg.duration_range, step=0.05),
                    ),
                ),
                GroupControl(
                    label="Body",
                    children=(
                        SliderControl("footswing_height", "swing h.", *cfg.footswing_height_range, step=0.01, unit="m"),
                        SliderControl("body_height", "h.", *cfg.body_height_range, step=0.01, unit="m"),
                        SliderControl("body_pitch", "pitch", *cfg.body_pitch_range, step=0.02, unit="rad"),
                        SliderControl("body_roll", "roll", *cfg.body_roll_range, step=0.02, unit="rad"),
                    ),
                ),
                GroupControl(
                    label="Stance",
                    children=(
                        SliderControl("stance_width", "width", *cfg.stance_width_range, step=0.01, unit="m"),
                        SliderControl("stance_length", "length", *cfg.stance_length_range, step=0.01, unit="m"),
                    ),
                ),
                GroupControl(label="Presets", children=preset_buttons),
            ),
        )
