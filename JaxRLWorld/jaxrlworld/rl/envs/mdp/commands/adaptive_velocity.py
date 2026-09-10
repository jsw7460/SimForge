"""Velocity command term with binned adaptive sampling and a
performance-gated active range.

Extends :class:`VelocityCommandTerm` on the ``lin_vel_x`` axis only:

* The full ``lin_vel_x_final_range`` is divided into ``num_vx_bins``
  equal bins. Each bin keeps an EMA of "tour" success — a tour is one
  command hold (resample-to-resample, or episode end), successful when
  the mean |vx_cmd - vx_measured| over the tour stays below
  ``success_error_threshold``.
* Sampling is a mixture: with probability ``adaptive_fraction`` a bin
  is drawn with weight ``max(1 - success_ema, min_bin_weight)`` (hard /
  frontier bins get sampled more), otherwise vx is uniform over the
  current ACTIVE range. Only bins inside the active range are eligible.
* The active range starts at ``lin_vel_x_range`` and is widened one bin
  at a time by :meth:`maybe_expand` (driven by the
  ``command_velocity_range`` curriculum term) whenever the frontier bin
  is being tracked reliably.

``lin_vel_y`` / ``ang_vel`` sampling, standing-env zeroing and heading
control are inherited unchanged. Prior art: the velocity curriculum of
Margolis et al., "Rapid Locomotion via Reinforcement Learning" (frontier
-weighted command sampling), and legged_gym's performance-gated
``update_command_curriculum``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.envs.managers.common.command_term import (
    VelocityCommandTerm,
    VelocityCommandTermCfg,
)

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.world import World


@dataclass
class AdaptiveVelocityCommandTermCfg(VelocityCommandTermCfg):
    """Configuration for :class:`AdaptiveVelocityCommandTerm`.

    ``lin_vel_x_range`` (inherited) is the INITIAL active range; the
    curriculum widens it toward ``lin_vel_x_final_range``.
    """

    lin_vel_x_final_range: tuple[float, float] = (-1.0, 2.0)
    num_vx_bins: int = 12
    # Probability of a frontier-weighted bin draw (vs uniform over the
    # active range). Uniform mass keeps mastered speeds rehearsed.
    adaptive_fraction: float = 0.5
    # Mean |vx_cmd - vx_meas| [m/s] below which a tour counts as success.
    success_error_threshold: float = 0.3
    # EMA rate for per-bin success updates (one update per batch of
    # finished tours landing in the bin).
    success_ema: float = 0.05
    # Tours shorter than this many control steps carry no signal
    # (command barely held) and are dropped from the stats.
    min_tour_steps: int = 25
    # Floor on bin sampling weight so mastered bins never starve.
    min_bin_weight: float = 0.1

    def build(self, env: World) -> AdaptiveVelocityCommandTerm:
        return AdaptiveVelocityCommandTerm(env, self)


class AdaptiveVelocityCommandTerm(VelocityCommandTerm):
    """Velocity command with adaptive vx sampling + expandable range."""

    cfg: AdaptiveVelocityCommandTermCfg

    def __init__(self, env: World, cfg: AdaptiveVelocityCommandTermCfg):
        super().__init__(env, cfg)
        final_lo, final_hi = cfg.lin_vel_x_final_range
        init_lo, init_hi = cfg.lin_vel_x_range
        if not (final_lo <= init_lo < init_hi <= final_hi):
            raise ValueError(
                f"lin_vel_x_range {cfg.lin_vel_x_range} must be inside "
                f"lin_vel_x_final_range {cfg.lin_vel_x_final_range}"
            )
        nb = cfg.num_vx_bins
        self.bin_width = (final_hi - final_lo) / nb
        self.bin_edges = torch.linspace(final_lo, final_hi, nb + 1, device=self.device)
        self._bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        if not bool(((self._bin_centers >= init_lo) & (self._bin_centers <= init_hi)).any()):
            raise ValueError("initial lin_vel_x_range does not cover any vx bin center")
        # Per-bin tour statistics.
        self.bin_success = torch.zeros(nb, device=self.device)
        self.bin_tours = torch.zeros(nb, device=self.device)
        # Active sampling range (0-dim tensors so the curriculum can
        # expand them without host syncs).
        self.active_min = torch.tensor(float(init_lo), device=self.device)
        self.active_max = torch.tensor(float(init_hi), device=self.device)
        self._final_lo = torch.tensor(float(final_lo), device=self.device)
        self._final_hi = torch.tensor(float(final_hi), device=self.device)
        self._bin_arange = torch.arange(nb, device=self.device)
        # Frontier success snapshots (updated by maybe_expand; logged by
        # the curriculum term).
        self.frontier_hi_success = torch.zeros((), device=self.device)
        self.frontier_lo_success = torch.zeros((), device=self.device)
        # Per-env tour accumulators.
        self._err_sum = torch.zeros(self.num_envs, device=self.device)
        self._tour_steps = torch.zeros(self.num_envs, device=self.device)

    # ── Tour tracking ──────────────────────────────────────────────

    def _update_command(self) -> None:
        super()._update_command()
        vx_meas = self._env.robot_data.root_link_lin_vel_b[:, 0]
        self._err_sum += (self._command[:, 0] - vx_meas).abs()
        self._tour_steps += 1.0

    def _finish_tours(self, env_ids: torch.Tensor) -> None:
        """Fold the ending tours of ``env_ids`` into the bin stats."""
        steps = self._tour_steps[env_ids]
        weight = (steps >= float(self.cfg.min_tour_steps)).float()
        if self.cfg.rel_standing_envs > 0.0:
            # Standing envs hold a zeroed command — no tracking signal.
            weight = weight * (~self.is_standing_env[env_ids]).float()
        err = self._err_sum[env_ids] / steps.clamp(min=1.0)
        success = (err < self.cfg.success_error_threshold).float()
        bins = ((self._command[env_ids, 0] - self.bin_edges[0]) / self.bin_width).long()
        bins = bins.clamp(0, self.cfg.num_vx_bins - 1)
        cnt = torch.zeros_like(self.bin_tours).index_add_(0, bins, weight)
        ssum = torch.zeros_like(self.bin_tours).index_add_(0, bins, success * weight)
        mean = ssum / cnt.clamp(min=1.0)
        touched = cnt > 0
        a = self.cfg.success_ema
        self.bin_success = torch.where(touched, (1.0 - a) * self.bin_success + a * mean, self.bin_success)
        self.bin_tours = self.bin_tours + cnt
        self._err_sum[env_ids] = 0.0
        self._tour_steps[env_ids] = 0.0

    # ── Sampling ───────────────────────────────────────────────────

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._finish_tours(env_ids)
        # Base fills all three columns (+ standing / heading draws);
        # column 0 is overwritten with the adaptive draw below.
        super()._resample_command(env_ids)
        n = len(env_ids)
        span = self.active_max - self.active_min
        uniform_v = torch.rand(n, device=self.device) * span + self.active_min
        in_active = (self._bin_centers >= self.active_min) & (self._bin_centers <= self.active_max)
        weights = (1.0 - self.bin_success).clamp(min=self.cfg.min_bin_weight) * in_active.float()
        chosen = torch.multinomial(weights, n, replacement=True)
        adaptive_v = self.bin_edges[chosen] + torch.rand(n, device=self.device) * self.bin_width
        adaptive_v = torch.clamp(adaptive_v, self.active_min, self.active_max)
        pick = torch.rand(n, device=self.device) < self.cfg.adaptive_fraction
        self._command[env_ids, 0] = torch.where(pick, adaptive_v, uniform_v)

    # ── Curriculum surface ─────────────────────────────────────────

    def maybe_expand(self, promote_threshold: float, min_tours: float) -> None:
        """Widen the active range by one bin per side whose frontier bin
        is tracked reliably (success EMA above ``promote_threshold``
        with at least ``min_tours`` tours). Fully on-device.
        """
        in_active = (self._bin_centers >= self.active_min) & (self._bin_centers <= self.active_max)
        nb = self.cfg.num_vx_bins
        hi_idx = torch.where(in_active, self._bin_arange, torch.full_like(self._bin_arange, -1)).max()
        lo_idx = torch.where(in_active, self._bin_arange, torch.full_like(self._bin_arange, nb)).min()
        self.frontier_hi_success = self.bin_success[hi_idx]
        self.frontier_lo_success = self.bin_success[lo_idx]
        ok_hi = (self.frontier_hi_success > promote_threshold) & (self.bin_tours[hi_idx] >= min_tours)
        ok_lo = (self.frontier_lo_success > promote_threshold) & (self.bin_tours[lo_idx] >= min_tours)
        self.active_max = torch.where(
            ok_hi, torch.minimum(self.active_max + self.bin_width, self._final_hi), self.active_max
        )
        self.active_min = torch.where(
            ok_lo, torch.maximum(self.active_min - self.bin_width, self._final_lo), self.active_min
        )
