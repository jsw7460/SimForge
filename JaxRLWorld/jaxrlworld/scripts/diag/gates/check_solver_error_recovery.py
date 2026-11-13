"""Gate: an env the solver corrupted ends the step as ``solver_error``, alone.

Two identical envs step in lockstep under zero actions. One joint velocity
of one env of the first is then overwritten with NaN, and the next step
must:

1. terminate that env under the built-in ``solver_error`` term, without a
   bootstrap, with a zero reward in the total and in every per-term
   readout, and with a finite terminal observation;
2. leave every other env exactly as the clean twin has it (bitwise on
   Genesis; mujoco-warp's parallel reductions make the mjlab and Newton
   twins agree only to a tolerance);
3. hand back a finite observation for every env, and step on without
   raising, the corrupted env included.

The configuration-error path is checked apart: on Genesis a capacity flag
raised by hand must still raise Genesis' own message; on Newton and mjlab
a one-slot ``nconmax`` must raise the contact-overflow guard's.

Usage:
    jaxpy -m jaxrlworld.scripts.diag.gates.check_solver_error_recovery --sim genesis
    jaxpy -m jaxrlworld.scripts.diag.gates.check_solver_error_recovery --sim newton
    jaxpy -m jaxrlworld.scripts.diag.gates.check_solver_error_recovery --sim mujoco
"""

from __future__ import annotations

import argparse
import importlib
import math

import torch

from jaxrlworld.rl.envs.managers.common.termination import SOLVER_ERROR_TERM
from jaxrlworld.rl.evals.sim_initializers import get_initializer

_SIM_NAMES = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_DEFAULT_CONFIG = "jaxrlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig"

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


def build_env(config: str, sim: str, num_envs: int, nconmax: int | None = None):
    mod_path, cls_name = config.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs, seed=0).build()
    if nconmax is not None:
        cfgs.scene.nconmax = nconmax
    env = get_initializer(_SIM_NAMES[sim]).init_environment(cfgs)
    env.reset()
    return env


def _rows_equal(a: torch.Tensor, b: torch.Tensor, keep: torch.Tensor, exact: bool) -> bool:
    a, b = a[keep], b[keep]
    if a.dtype == torch.bool or not a.is_floating_point():
        return torch.equal(a, b)
    if exact:
        return torch.equal(a, b)
    return torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def run_recovery(sim: str, config: str, num_envs: int, warmup: int, corrupt_env: int) -> None:
    print(f"--- recovery ({sim}, {num_envs} envs, corrupt env {corrupt_env})")
    env_a = build_env(config, sim, num_envs)
    env_b = build_env(config, sim, num_envs)
    zeros = torch.zeros(num_envs, env_a.num_actions, device=env_a.device)
    for _ in range(warmup):
        env_a.step(zeros)
        env_b.step(zeros)

    ids = torch.tensor([corrupt_env], device=env_a.device)
    robot = env_a.get_robot_data(env_a.robot_entity_name)
    pos = robot.joint_pos[ids].clone()
    vel = robot.joint_vel[ids].clone()
    vel[:, 0] = math.nan
    env_a.get_robot_state_writer(env_a.robot_entity_name).set_dof_state(pos, vel, ids)

    obs_a, rew_a, term_a, trunc_a, extras_a = env_a.step(zeros)
    obs_b, rew_b, term_b, trunc_b, extras_b = env_b.step(zeros)
    k = corrupt_env
    exact = sim == "genesis"

    fires = env_a.termination_manager.term_dones[SOLVER_ERROR_TERM]
    check("solver_error fired on the corrupt env", bool(fires[k]))
    check("solver_error fired on no other env", int(fires.sum()) == 1, f"{int(fires.sum())} envs")
    check("corrupt env terminated", bool(term_a[k]))
    check("corrupt env not bootstrapped", not bool(extras_a["bootstrap_mask"][k]))
    check("corrupt env reward is 0", float(rew_a[k]) == 0.0, f"{float(rew_a[k])}")
    per_type = extras_a["rewards_per_type"]
    bad_terms = [name for name, value in per_type.items() if float(value[k]) != 0.0]
    check("every per-term reward is 0 on the corrupt env", not bad_terms, ", ".join(bad_terms))
    final = extras_a["final_observation"]
    check("terminal observation finite", all(torch.isfinite(v).all() for v in final.values()))
    check("returned observation finite", all(torch.isfinite(v).all() for v in obs_a.values()))
    stats = env_a.termination_manager.consume_episode_stats()
    key = f"Episode_Termination/{SOLVER_ERROR_TERM}"
    check("episode stats count the solver error", key in stats and stats[key] > 0.0, f"{stats.get(key)}")

    keep = torch.ones(num_envs, dtype=torch.bool, device=env_a.device)
    keep[k] = False
    check("other envs: terminated matches twin", _rows_equal(term_a, term_b, keep, exact))
    check("other envs: truncated matches twin", _rows_equal(trunc_a, trunc_b, keep, exact))
    check("other envs: reward matches twin", _rows_equal(rew_a, rew_b, keep, exact))
    for name in obs_a:
        check(f"other envs: obs[{name}] matches twin", _rows_equal(obs_a[name], obs_b[name], keep, exact))

    obs_a, rew_a, term_a, trunc_a, _ = env_a.step(zeros)
    check("step after the recovery raises nothing and is finite", all(torch.isfinite(v).all() for v in obs_a.values()))
    check("corrupt env's reward finite after recovery", bool(torch.isfinite(rew_a[k])))
    check("no solver_error on the step after", int(env_a.termination_manager.term_dones[SOLVER_ERROR_TERM].sum()) == 0)


def run_misconfiguration(sim: str, config: str, num_envs: int) -> None:
    print(f"--- configuration error ({sim})")
    if sim == "genesis":
        import genesis as gs
        from genesis.utils.array_class import ErrorCode

        env = build_env(config, sim, num_envs)
        zeros = torch.zeros(num_envs, env.num_actions, device=env.device)
        env.step(zeros)
        env.scene.rigid_solver.flag_errno(int(ErrorCode.OVERFLOW_CONTACTS), 0)
        raised = None
        try:
            for _ in range(3):
                env.step(zeros)
        except gs.GenesisException as e:
            raised = str(e)
        check(
            "capacity flag raises Genesis' own message", raised is not None and "max_contacts" in raised, raised or ""
        )
        return

    env = build_env(config, sim, num_envs, nconmax=1)
    zeros = torch.zeros(num_envs, env.num_actions, device=env.device)
    raised = None
    try:
        for _ in range(3):
            env.step(zeros)
    except RuntimeError as e:
        raised = str(e)
    check("one-slot nconmax raises the overflow guard", raised is not None and "nconmax" in raised, raised or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=tuple(_SIM_NAMES), required=True)
    ap.add_argument("--config", default=_DEFAULT_CONFIG)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--corrupt-env", type=int, default=3)
    args = ap.parse_args()

    run_recovery(args.sim, args.config, args.num_envs, args.warmup, args.corrupt_env)
    run_misconfiguration(args.sim, args.config, args.num_envs)
    print(f"\n{'ALL PASS' if not _failures else 'FAILED: ' + ', '.join(_failures)}")
    return 0 if not _failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
