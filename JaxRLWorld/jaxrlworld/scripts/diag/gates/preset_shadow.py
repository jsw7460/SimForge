"""Shadow evaluation of the compiled programs for ``check_all_presets``.

The env runs eager; at every call the compiled program of a component
is evaluated on the SAME inputs, compared, and discarded (state rewound
in between). This is the numerical verdict on the compiled paths —
exact and per component — where a trajectory comparison cannot
separate a fused kernel's last-bit rounding from a wrong computation
once physics has amplified it.

Kept apart from ``check_all_presets`` because that script also runs as
the child of a ``--mode baseline`` sweep, under a copy of the package
from before these programs existed: nothing here may be imported there.
"""

from __future__ import annotations

import torch
import torch._dynamo

from jaxrlworld.rl.actuators.actuator_pd import IdealPDActuator
from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager
from jaxrlworld.rl.envs.managers.common.reward_view import RecordingEnvView, RewardEnvView, RewardReadRecord
from jaxrlworld.scripts.diag.gates.check_reward_compile_parity import TermState


def _abs_scale(eager, compiled) -> tuple[float, float]:
    """(largest absolute difference, scale of the eager value)."""
    if eager is None:
        if compiled is not None:
            raise RuntimeError("compiled chain returned a value where eager returned None")
        return 0.0, 1.0
    if eager.dtype == torch.bool:
        return float((eager != compiled).any()), 1.0
    return float((eager - compiled).abs().max()), float(eager.abs().max().clamp_min(1e-6))


def _rel(eager, compiled) -> float:
    diff, scale = _abs_scale(eager, compiled)
    return diff / scale


class _ActuatorShadow:
    """Stands in for ``actuator._chain``: eager torque chain returned,
    the compiled one evaluated on the same joint state and compared."""

    def __init__(self, actuator: IdealPDActuator):
        self._actuator = actuator
        self._compiled = torch.compile(actuator._torque_chain, fullgraph=True, dynamic=False)
        self.worst = 0.0
        # ``DCMotor`` overrides ``compute`` without the chain: never called.
        self.calls = 0
        actuator._chain = self

    def __call__(self, target_pos, joint_pos, joint_vel):
        self.calls += 1
        eager = self._actuator._torque_chain(target_pos, joint_pos, joint_vel)
        compiled = self._compiled(target_pos, joint_pos, joint_vel)
        for e, c in zip(eager, compiled):
            self.worst = max(self.worst, _rel(e, c))
        return eager


class _RewardShadow:
    """Stands in for ``reward_manager.set_rewards``: the compiled chain
    (recorded on the first call, as the manager itself does) runs first
    on the current state, the stateful terms' buffers are put back, then
    the eager path runs and its values are kept. A preset whose terms the
    chain cannot trace is reported, not failed: nothing configures
    ``compile_terms`` there."""

    def __init__(self, mgr):
        self._mgr = mgr
        self._original = mgr.set_rewards
        self._view = None
        self._compiled = None
        self.worst = 0.0
        self.compared = 0
        # Per term: worst relative difference and, at that step, the
        # absolute difference and the eager value's scale — a term that
        # cancels two O(1) quantities into an O(1e-4) result shows a
        # last-bit absolute error under a large relative one.
        self.worst_terms: dict[str, tuple[float, float, float]] = {}
        self.error: str | None = None
        mgr.set_rewards = self

    def __call__(self, reward_buffer, reward_buffer_per_type):
        mgr = self._mgr
        if self.error is None and mgr.reward_terms:
            state = TermState(mgr)
            try:
                if self._view is None:
                    record = RewardReadRecord()
                    mgr._compute_stacked(RecordingEnvView(mgr.env, record))
                    state.restore()
                    self._view = RewardEnvView(mgr.env, record)
                    self._compiled = torch.compile(mgr._chain, fullgraph=True, dynamic=False)
                self._view.refresh()
                total = reward_buffer.clone()
                stacked = self._compiled(self._view, mgr._weights(), mgr._active(), total)
            except Exception as exc:  # noqa: BLE001 — reported per cell, see the class docstring
                self.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                stacked = None
            state.restore()
        else:
            stacked = None
        self._original(reward_buffer=reward_buffer, reward_buffer_per_type=reward_buffer_per_type)
        if stacked is not None:
            self.compared += 1
            for i, name in enumerate(mgr.reward_terms):
                diff, scale = _abs_scale(reward_buffer_per_type[name], stacked[i])
                rel = diff / scale
                if rel > self.worst_terms.get(name, (0.0, 0.0, 1.0))[0]:
                    self.worst_terms[name] = (rel, diff, scale)
                self.worst = max(self.worst, rel)
            self.worst = max(self.worst, _rel(reward_buffer, total))


class _ContactShadow:
    """Stands in for ``GenesisContactBatch._substep``: eager substep,
    timing buffers rewound, compiled substep on the same contact list,
    rings / forces / timing compared, eager results kept."""

    def __init__(self, batch):
        self._batch = batch
        self._fields = BaseContactManager._TIMING_FIELDS
        self._compiled = torch.compile(batch._substep_impl, fullgraph=True, dynamic=False)
        self.worst = 0.0
        batch._substep = self

    def __call__(self, link_a, link_b, force, n_live, quats, found_hists, force_hists, timing, dt):
        before = {f: getattr(timing, f).clone() for f in self._fields}
        eager = self._batch._substep_impl(link_a, link_b, force, n_live, quats, found_hists, force_hists, timing, dt)
        after = {f: getattr(timing, f).clone() for f in self._fields}
        for f in self._fields:
            getattr(timing, f).copy_(before[f])
        for t in (link_a, link_b, force):
            torch._dynamo.mark_dynamic(t, 1)
        compiled = self._compiled(link_a, link_b, force, n_live, quats, found_hists, force_hists, timing, dt)
        for f in self._fields:
            self.worst = max(self.worst, _rel(after[f], getattr(timing, f)))
            getattr(timing, f).copy_(after[f])
        for e_list, c_list in zip(eager, compiled):
            for e, c in zip(e_list, c_list):
                self.worst = max(self.worst, _rel(e, c))
        return eager


def install_shadows(env, sim: str, no_reward_shadow: str | None = None) -> dict:
    """``no_reward_shadow``: why this preset's reward stack cannot be
    evaluated twice in one step (reported as unsupported, not compared)."""
    shadows = {
        "actuator": [
            _ActuatorShadow(act) for act, _ids in env.act_manager.actuators if isinstance(act, IdealPDActuator)
        ],
        "reward": no_reward_shadow if no_reward_shadow else _RewardShadow(env.reward_manager),
    }
    if sim == "genesis" and env.contact_manager._sensors:
        shadows["contact"] = _ContactShadow(env.contact_manager._get_batch())
    return shadows


def shadow_report(shadows: dict) -> dict:
    """Per component: the worst relative difference, ``None`` when the
    preset has no such component, or the compile error text."""
    reward = shadows["reward"]
    if isinstance(reward, str):
        reward_value, reward_terms = reward, {}
    else:
        reward_value = reward.error if reward.error is not None else (reward.worst if reward.compared else None)
        reward_terms = dict(sorted(reward.worst_terms.items(), key=lambda kv: -kv[1][0])[:3])
    return {
        "actuator": max((s.worst for s in shadows["actuator"] if s.calls), default=None),
        "reward": reward_value,
        "reward_terms": reward_terms,
        "contact": shadows["contact"].worst if "contact" in shadows else None,
    }
