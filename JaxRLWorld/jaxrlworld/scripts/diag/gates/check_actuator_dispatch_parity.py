"""Is the trimmed per-substep actuator dispatch bit-identical to the old one?

Two launch-count trims on the explicit-actuator path, both claiming to
change nothing but the number of kernels per substep:

- ``ActionManagerBase._compute_actuator_torques`` hands the full-width
  target / joint state straight to the actuator when a single actuator
  covers every actuated joint in order, instead of gathering three
  subsets by an identity index, computing, and scattering the result
  into a zeroed buffer.
- ``DelayedPDActuator`` looks the ring-buffer read slot up in a table
  rebuilt at reset instead of recomputing ``(head - 1 - delay) % n``
  and an ``arange`` on every substep.

This runs the old formulation next to the new one on the CPU, through
resets, and demands ``torch.equal`` on every torque and on the internal
state.

    python -m jaxrlworld.scripts.diag.gates.check_actuator_dispatch_parity
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from jaxrlworld.rl.actuators.actuator_cfg import DelayedPDActuatorCfg
from jaxrlworld.rl.actuators.actuator_pd import DelayedPDActuator, IdealPDActuator
from jaxrlworld.rl.envs.managers.common.action import ActionManagerBase

_JOINTS = [f"j{i}" for i in range(22)]


def _make_actuator(min_delay: int, max_delay: int, num_envs: int) -> DelayedPDActuator:
    cfg = DelayedPDActuatorCfg(
        target_names_expr=(".*",),
        stiffness={r"j\d": 40.0, r"j1\d": 25.0, r"j2\d": 60.0},
        damping={r"j\d": 1.5, r"j1\d": 0.8, r"j2\d": 2.0},
        effort_limit={r"j\d": 30.0, r"j1\d": 12.0, r"j2\d": 90.0},
        velocity_limit=20.0,
        knee_point_velocity=5.0,
        tau_lpf_time_constant=0.0,
        dyn_gain=1.0,
        dyn_gain_velocity=0.5,
        min_delay=min_delay,
        max_delay=max_delay,
        # This gate is about dispatch, not the torque chain: the eager
        # chain is what the inline old formulation below reproduces
        # exactly (check_actuator_compile_parity covers the compiled one).
        compile_kernel=False,
    )
    return DelayedPDActuator(cfg, num_envs=num_envs, num_joints=len(_JOINTS), device="cpu", joint_names=_JOINTS)


def _old_delayed_compute(act: DelayedPDActuator, target, pos, vel) -> torch.Tensor:
    """``DelayedPDActuator.compute`` as it was before the lookup table."""
    act._buffer[act._head] = target
    act._head = (act._head + 1) % act._max_delay
    read_idx = (act._head - 1 - act._delay) % act._max_delay
    env_idx = torch.arange(act._num_envs, device=act._device)
    delayed = act._buffer[read_idx, env_idx]
    return IdealPDActuator.compute(act, delayed, pos, vel)


def _old_manager_compute(actuators, target, pos, vel) -> torch.Tensor:
    """``_compute_actuator_torques`` as it was: gather / compute / scatter."""
    full = torch.zeros_like(target)
    for actuator, joint_idx in actuators:
        full[:, joint_idx] = actuator.compute(target[:, joint_idx], pos[:, joint_idx], vel[:, joint_idx])
    return full


def _random_inputs(g: torch.Generator, num_envs: int):
    n = len(_JOINTS)
    target = (torch.rand((num_envs, n), generator=g) * 2.0 - 1.0) * 1.5
    pos = (torch.rand((num_envs, n), generator=g) * 2.0 - 1.0) * 1.5
    vel = (torch.rand((num_envs, n), generator=g) * 2.0 - 1.0) * 25.0
    return target, pos, vel


def check_delayed_pd(seeds: int = 8, num_envs: int = 1024, substeps: int = 200, reset_every: int = 7) -> None:
    for seed in range(seeds):
        min_delay, max_delay = (6, 12) if seed % 2 == 0 else (0, 3)
        torch.manual_seed(seed)
        new = _make_actuator(min_delay, max_delay, num_envs)
        old = copy.deepcopy(new)
        g = torch.Generator().manual_seed(seed)
        for k in range(substeps):
            if k % reset_every == 0:
                n_reset = int(torch.randint(0, 200, (), generator=g))
                env_ids = torch.randperm(num_envs, generator=g)[:n_reset]
                # Both draw their new delays from the global stream; seed it so they draw the same.
                torch.manual_seed(seed * 1000 + k)
                new.reset(env_ids)
                torch.manual_seed(seed * 1000 + k)
                old.reset(env_ids)
            target, pos, vel = _random_inputs(g, num_envs)
            tau_new = new.compute(target, pos, vel)
            tau_old = _old_delayed_compute(old, target, pos, vel)
            assert torch.equal(tau_old, tau_new), f"seed {seed} substep {k}: delayed PD torque differs"
            assert torch.equal(old._delay, new._delay), f"seed {seed} substep {k}: delays diverged"
            assert torch.equal(old._buffer, new._buffer), f"seed {seed} substep {k}: ring buffers diverged"
            assert old._head == new._head
    print(f"  delayed PD: {seeds} seeds x {substeps} substeps with resets bit-identical")


def check_manager_fast_path(seeds: int = 8, num_envs: int = 1024, substeps: int = 100) -> None:
    n = len(_JOINTS)
    for seed in range(seeds):
        torch.manual_seed(seed)
        act = _make_actuator(6, 12, num_envs)
        act_old = copy.deepcopy(act)
        joint_idx = torch.arange(n)
        state = SimpleNamespace(pos=None, vel=None)
        stub = SimpleNamespace(
            _actuators=[(act, joint_idx)],
            _actuated_joint_names=list(_JOINTS),
            _single_full_width_actuator=True,
            _get_joint_pos=lambda name, s=state: s.pos,
            _get_joint_vel=lambda name, s=state: s.vel,
        )
        g = torch.Generator().manual_seed(seed)
        for k in range(substeps):
            target, state.pos, state.vel = _random_inputs(g, num_envs)
            tau_new = ActionManagerBase._compute_actuator_torques(stub, target, "robot")
            tau_old = _old_manager_compute([(act_old, joint_idx)], target, state.pos, state.vel)
            assert torch.equal(tau_old, tau_new), f"seed {seed} substep {k}: fast path differs from gather/scatter"
    print(f"  manager fast path: {seeds} seeds x {substeps} substeps bit-identical")


def main() -> None:
    torch.set_default_dtype(torch.float32)
    print("actuator dispatch parity (CPU, torch.equal)")
    check_delayed_pd()
    check_manager_fast_path()
    print("PASS")


if __name__ == "__main__":
    main()
