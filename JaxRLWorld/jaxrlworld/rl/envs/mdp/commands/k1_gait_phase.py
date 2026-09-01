"""Per-foot gait-phase clock for the K1 joystick task.

Mirrors the upstream mujoco_playground K1 phase state exactly:

- init ``[0, pi]`` (left, right) on every episode reset
- each control step: ``phi += 2*pi*dt*freq``, wrapped to ``[-pi, pi)``
- when the velocity-command norm is <= ``freeze_threshold`` BOTH feet
  are overwritten with ``pi`` — the left/right offset is intentionally
  lost until the next reset (upstream behaviour, not a bug here)
- ``freq ~ U(freq_range)`` per episode, resampled only on reset

Step-order alignment with upstream: there, obs and rewards of step *k*
both read the phase BEFORE the end-of-step advance. In this framework
rewards run before ``command_manager.compute`` and observations after,
so the term keeps two views: reward terms read :attr:`command` (the
live phase, pre-advance at reward time) and observation terms read
:attr:`phase_obs` (snapshotted just before the advance) — both equal
the upstream step-*k* phase. Just-reset envs (``episode_length_buf ==
0``) skip the advance so the first observation after a reset is the
init phase, as upstream's ``reset()``.

External control (viewer sliders) is not supported for this clock:
:meth:`compute` is overridden without the locked-column machinery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from jaxrlworld.rl.envs.managers.common.command_term import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import World

_TWO_PI = 2.0 * math.pi


@dataclass
class K1GaitPhaseCommandTermCfg(CommandTermCfg):
    """Config for the K1 per-foot gait-phase clock."""

    freq_range: tuple[float, float] = (1.25, 1.75)
    init_phase: tuple[float, float] = (0.0, math.pi)
    freeze_phase: float = math.pi
    velocity_term: str = "velocity"
    freeze_threshold: float = 0.01

    def build(self, env: World) -> K1GaitPhaseCommandTerm:
        return K1GaitPhaseCommandTerm(env, self)


class K1GaitPhaseCommandTerm(CommandTerm):
    """Stateful per-foot phase in ``[-pi, pi)`` (see module docstring)."""

    cfg: K1GaitPhaseCommandTermCfg
    column_names = ("phase_left", "phase_right")

    def __init__(self, env: World, cfg: K1GaitPhaseCommandTermCfg):
        super().__init__(env, cfg)
        init = torch.tensor(cfg.init_phase, device=self.device)
        self._command = init.unsqueeze(0).repeat(self.num_envs, 1)
        # Pre-advance snapshot read by observation terms.
        self.phase_obs = self._command.clone()
        self.freq = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.freq[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.freq_range)
        init = torch.tensor(self.cfg.init_phase, device=self.device)
        self._command[env_ids] = init
        self.phase_obs[env_ids] = init

    def compute(self, dt: float) -> None:
        if dt <= 0.0:
            return
        self.phase_obs = self._command.clone()

        advanced = self._command + _TWO_PI * dt * self.freq.unsqueeze(1)
        # Inputs are >= 0 after the +pi shift, so remainder == upstream fmod.
        advanced = torch.remainder(advanced + math.pi, _TWO_PI) - math.pi

        vel_cmd = self._env.command_manager.get_term(self.cfg.velocity_term).command
        keep_running = vel_cmd.norm(dim=1) > self.cfg.freeze_threshold
        frozen = torch.full_like(advanced, self.cfg.freeze_phase)
        new_phase = torch.where(keep_running.unsqueeze(1), advanced, frozen)

        # Just-reset envs keep the init phase for their first observation.
        active = (self._env.episode_length_buf > 0).unsqueeze(1)
        self._command = torch.where(active, new_phase, self._command)
