"""Does the compiled actuator torque chain match the eager one?

``IdealPDActuator`` runs PD -> tanh saturation -> first-order lag ->
velocity-gated efficiency -> effort clip (box or torque-speed curve)
through ``torch.compile`` when ``IdealPDActuatorCfg.compile_kernel`` is
set. Fused multiply-adds round differently from the eager op sequence,
so the two are NOT expected to be bit-identical; this drives every
feature combination with random joint state through both, across resets
(the lag state), and demands agreement to 1e-5 relative of the torque
scale. Run it on the machine whose backend the training uses.

    jaxpy -m jaxrlworld.scripts.diag.gates.check_actuator_compile_parity
"""

from __future__ import annotations

import torch

from jaxrlworld.rl.actuators.actuator_cfg import DelayedPDActuatorCfg, IdealPDActuatorCfg
from jaxrlworld.rl.actuators.actuator_pd import DelayedPDActuator, IdealPDActuator

_JOINTS = [f"j{i}" for i in range(22)]
_FEATURES = {
    "plain": {},
    # j1x left out of the map: those joints are not saturated.
    "tau_scale": {"tau_scale": {r"j\d": 20.0, r"j2\d": 50.0}},
    "lpf": {"tau_lpf_time_constant": {r"j\d": 0.02, r"j[12]\d": 0.0}, "physics_dt": 0.005},
    "dyn_gain": {"dyn_gain": {r"j\d": 0.7, r"j[12]\d": 1.0}, "dyn_gain_velocity": 0.5},
    "torque_speed": {"velocity_limit": 20.0, "knee_point_velocity": 5.0},
    "all": {
        "tau_scale": 30.0,
        "tau_lpf_time_constant": 0.01,
        "physics_dt": 0.005,
        "dyn_gain": 0.8,
        "dyn_gain_velocity": 0.5,
        "velocity_limit": 20.0,
        "knee_point_velocity": 5.0,
    },
}
_BASE = {
    "target_names_expr": (".*",),
    "stiffness": {r"j\d": 40.0, r"j1\d": 25.0, r"j2\d": 60.0},
    "damping": {r"j\d": 1.5, r"j1\d": 0.8, r"j2\d": 2.0},
    "effort_limit": {r"j\d": 30.0, r"j1\d": 12.0, r"j2\d": 90.0},
}


def _pair(feature: str, delayed: bool, num_envs: int, device: str):
    kwargs = {**_BASE, **_FEATURES[feature]}
    if delayed:
        kwargs.update(min_delay=2, max_delay=6)
    out = []
    for compiled in (False, True):
        torch.manual_seed(0)
        cfg = (DelayedPDActuatorCfg if delayed else IdealPDActuatorCfg)(compile_kernel=compiled, **kwargs)
        cls = DelayedPDActuator if delayed else IdealPDActuator
        out.append(cls(cfg, num_envs=num_envs, num_joints=len(_JOINTS), device=device, joint_names=_JOINTS))
    return out


def check(feature: str, delayed: bool, device: str, num_envs: int = 2048, substeps: int = 120, tol: float = 1e-5):
    eager, compiled = _pair(feature, delayed, num_envs, device)
    g = torch.Generator(device="cpu").manual_seed(1)
    n = len(_JOINTS)
    worst = 0.0
    for k in range(substeps):
        if k % 17 == 5:
            env_ids = torch.randperm(num_envs, generator=g)[:97].to(device)
            torch.manual_seed(k)
            eager.reset(env_ids)
            torch.manual_seed(k)
            compiled.reset(env_ids)
        target = ((torch.rand((num_envs, n), generator=g) * 2 - 1) * 1.5).to(device)
        pos = ((torch.rand((num_envs, n), generator=g) * 2 - 1) * 1.5).to(device)
        vel = ((torch.rand((num_envs, n), generator=g) * 2 - 1) * 25.0).to(device)
        a = eager.compute(target, pos, vel)
        b = compiled.compute(target, pos, vel)
        assert bool(torch.isfinite(a).all() and torch.isfinite(b).all()), f"{feature}: non-finite torque"
        for x, y in ((a, b), (eager.computed_effort, compiled.computed_effort)):
            scale = x.abs().max().clamp_min(1e-6)
            rel = ((x - y).abs().max() / scale).item()
            worst = max(worst, rel)
            assert rel <= tol, f"{feature}{' delayed' if delayed else ''}: rel diff {rel:.2e} at substep {k}"
    print(f"  {feature:<14}{'delayed' if delayed else 'ideal':<9} max rel diff {worst:.2e}")


def check_partial_tau_scale(device: str, num_envs: int = 64) -> None:
    """A joint the kappa map leaves out gets the unsaturated torque; the
    others ``kappa * tanh(tau_PD / kappa)``, before the same effort clip."""
    kappa = {r"j\d": 20.0, r"j2\d": 50.0}
    sat = IdealPDActuator(
        IdealPDActuatorCfg(**_BASE, tau_scale=kappa, compile_kernel=False),
        num_envs=num_envs,
        num_joints=len(_JOINTS),
        device=device,
        joint_names=_JOINTS,
    )
    raw = IdealPDActuator(
        IdealPDActuatorCfg(**_BASE, compile_kernel=False),
        num_envs=num_envs,
        num_joints=len(_JOINTS),
        device=device,
        joint_names=_JOINTS,
    )
    g = torch.Generator(device="cpu").manual_seed(2)
    n = len(_JOINTS)
    target = ((torch.rand((num_envs, n), generator=g) * 2 - 1) * 1.5).to(device)
    pos = ((torch.rand((num_envs, n), generator=g) * 2 - 1) * 1.5).to(device)
    vel = ((torch.rand((num_envs, n), generator=g) * 2 - 1) * 25.0).to(device)
    got = sat.compute(target, pos, vel)
    unsat = raw.compute(target, pos, vel)
    left_out = [i for i, j in enumerate(_JOINTS) if j.startswith("j1") and len(j) == 3]
    mapped = [i for i in range(n) if i not in left_out]
    assert torch.equal(got[:, left_out], unsat[:, left_out]), "left-out joints are not the unsaturated torque"
    tau_pd = raw.computed_effort[:, mapped]
    k = sat.tau_scale[:, mapped]
    limit = raw.effort_limit[:, mapped]
    expected = (k * torch.tanh(tau_pd / k)).clamp(-limit, limit)
    assert torch.allclose(got[:, mapped], expected, rtol=1e-6, atol=1e-6), "mapped joints are not kappa * tanh"
    print(f"  partial kappa  {len(left_out)} left-out joints unsaturated, {len(mapped)} saturated")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"compiled vs eager actuator chain ({device}, tol 1e-5 relative)")
    for feature in _FEATURES:
        check(feature, delayed=False, device=device)
    check("all", delayed=True, device=device)
    check_partial_tau_scale(device)
    print("PASS")


if __name__ == "__main__":
    main()
