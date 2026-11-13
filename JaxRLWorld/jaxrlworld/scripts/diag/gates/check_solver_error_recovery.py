"""Gate: an env the solver corrupted ends the step as ``solver_error``, alone.

The env steps under zero actions, then one joint velocity of one env is
overwritten with NaN. The next step must:

1. terminate that env, and only that env, under the built-in
   ``solver_error`` term, without a bootstrap, with a zero reward in the
   total and in every per-term readout, and with a finite terminal
   observation;
2. hand back a finite observation for every env, and step on without
   raising, the corrupted env included.

The configuration-error path is checked apart: on Genesis a capacity flag
raised by hand must still raise Genesis' own message; on Newton and mjlab
a one-slot ``nconmax`` under violent random actions must raise the
mujoco-warp overflow guard's.

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
        # The per-world contact budget lives on the solver config for Newton
        # and on the scene config for mjlab.
        if sim == "newton":
            cfgs.scene.solver_cfg.nconmax = nconmax
        elif sim == "mujoco":
            cfgs.scene.nconmax = nconmax
        else:
            raise ValueError(f"{sim} has no nconmax")
    env = get_initializer(_SIM_NAMES[sim]).init_environment(cfgs)
    env.reset()
    return env


def _all_finite(groups: dict[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(v).all()) for v in groups.values())


def run_recovery(sim: str, config: str, num_envs: int, warmup: int, corrupt_env: int) -> None:
    print(f"--- recovery ({sim}, {num_envs} envs, corrupt env {corrupt_env})")
    env = build_env(config, sim, num_envs)
    zeros = torch.zeros(num_envs, env.num_actions, device=env.device)
    for _ in range(warmup):
        env.step(zeros)

    k = corrupt_env
    ids = torch.tensor([k], device=env.device)
    robot = env.get_robot_data(env.robot_entity_name)
    pos = robot.joint_pos[ids].clone()
    vel = robot.joint_vel[ids].clone()
    vel[:, 0] = math.nan
    env.get_robot_state_writer(env.robot_entity_name).set_dof_state(pos, vel, ids)

    obs, rew, terminated, truncated, extras = env.step(zeros)
    fires = env.termination_manager.term_dones[SOLVER_ERROR_TERM]
    check("solver_error fired on the corrupt env", bool(fires[k]))
    check("solver_error fired on no other env", int(fires.sum()) == 1, f"{int(fires.sum())} envs")
    check("corrupt env terminated", bool(terminated[k]))
    check("corrupt env not bootstrapped", not bool(extras["bootstrap_mask"][k]))
    check("corrupt env reward is 0", float(rew[k]) == 0.0, f"{float(rew[k])}")
    bad_terms = [name for name, value in extras["rewards_per_type"].items() if float(value[k]) != 0.0]
    check("every per-term reward is 0 on the corrupt env", not bad_terms, ", ".join(bad_terms))
    check("terminal observation finite", _all_finite(extras["final_observation"]))
    check("returned observation finite", _all_finite(obs))
    check("other envs' rewards finite", bool(torch.isfinite(rew).all()))
    stats = env.termination_manager.consume_episode_stats()
    key = f"Episode_Termination/{SOLVER_ERROR_TERM}"
    check("episode stats count the solver error", key in stats and stats[key] > 0.0, f"{stats.get(key)}")

    obs, rew, terminated, truncated, _ = env.step(zeros)
    check("step after the recovery raises nothing and is finite", _all_finite(obs))
    check("corrupt env's reward finite after recovery", bool(torch.isfinite(rew[k])))
    fires = env.termination_manager.term_dones[SOLVER_ERROR_TERM]
    check("no solver_error on the step after", int(fires.sum()) == 0, f"{int(fires.sum())} envs")


def run_misconfiguration(sim: str, config: str, num_envs: int, thrash_steps: int) -> None:
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

    # One slot per world. mujoco-warp lifts it to the initial pose's contact
    # count, so the robots have to be thrown about before the pool overflows.
    env = build_env(config, sim, num_envs, nconmax=1)
    raised = None
    step = -1
    try:
        for step in range(thrash_steps):
            actions = torch.empty(num_envs, env.num_actions, device=env.device).uniform_(-5.0, 5.0)
            env.step(actions)
    except RuntimeError as e:
        raised = str(e)
    check(
        "one-slot nconmax raises the overflow guard",
        raised is not None and "nconmax" in raised,
        raised if raised is not None else f"no raise in {step + 1} steps",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=tuple(_SIM_NAMES), required=True)
    ap.add_argument("--config", default=_DEFAULT_CONFIG)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--corrupt-env", type=int, default=3)
    ap.add_argument("--thrash-steps", type=int, default=100)
    args = ap.parse_args()

    run_recovery(args.sim, args.config, args.num_envs, args.warmup, args.corrupt_env)
    run_misconfiguration(args.sim, args.config, args.num_envs, args.thrash_steps)
    print(f"\n{'ALL PASS' if not _failures else 'FAILED: ' + ', '.join(_failures)}")
    return 0 if not _failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
