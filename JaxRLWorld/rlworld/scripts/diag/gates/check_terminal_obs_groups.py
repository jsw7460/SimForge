"""Bitwise check: group-restricted terminal-observation pass.

``World.step`` captures the terminal observation with
``process_observations(groups=env.terminal_obs_groups)`` — the on-policy
runner narrows it to the critic's groups so terminal steps skip computing
what nobody reads. This diag proves the narrowed pass produces a
bit-identical critic group and leaves the obs pipeline's state (history,
delay, RNG) exactly where the full pass would have left it.

Method, per step, on one live env:

1. Save the torch RNG state (CPU + CUDA).
2. FULL pass: ``process_observations(update_history=True, groups=None)``,
   clone the critic group, ``rollback_last_history_append(None)``.
3. Restore the RNG state.
4. SUBSET pass: same with ``groups=("critic",)``, rollback the subset.
5. Restore the RNG state, compare the two critic captures bitwise, then
   let the env step normally.

The RNG restore removes ordering effects of *earlier groups'* noise
draws; the critic group itself draws the same stream in both passes.
Running the dual pass every step also stress-tests the subset rollback:
a wrong rewind corrupts history-based terms within a few steps and shows
up as a bitwise mismatch.

Usage:
    jaxpy -m rlworld.scripts.diag.gates.check_terminal_obs_groups --sim mujoco
    jaxpy -m rlworld.scripts.diag.gates.check_terminal_obs_groups --sim newton --steps 50
"""

from __future__ import annotations

import argparse
import importlib

import torch

_PER_SIM_PRESETS: dict[str, dict[str, tuple[str, str]]] = {
    "go2_gait": {
        "genesis": ("rlworld.rl.configs.presets.go2.genesis.gait_conditioned", "Go2GaitConditionedGenesisConfig"),
        "newton": ("rlworld.rl.configs.presets.go2.newton.gait_conditioned", "Go2GaitConditionedNewtonConfig"),
        "mujoco": ("rlworld.rl.configs.presets.go2.mujoco.gait_conditioned", "Go2GaitConditionedMujocoConfig"),
    },
}


def _build_env(preset: str, sim: str, num_envs: int):
    from rlworld.rl.runners import BaseRunner

    if ":" in preset:
        mod_path, cls_name = preset.split(":", 1)
    else:
        mod_path, cls_name = _PER_SIM_PRESETS[preset][sim]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _rng_snapshot():
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return torch.get_rng_state(), cuda


def _rng_restore(snap) -> None:
    cpu, cuda = snap
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _clone_group(value):
    if isinstance(value, dict):
        return {k: v.clone() for k, v in value.items()}
    return value.clone()


def _compare_group(a, b) -> list[str]:
    """Return the names of mismatching (sub-)tensors; empty means bitwise equal."""
    if isinstance(a, dict):
        return [k for k in a if not torch.equal(a[k], b[k])]
    return [] if torch.equal(a, b) else ["<tensor>"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="go2_gait")
    ap.add_argument("--sim", default="mujoco", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = _build_env(args.preset, args.sim, args.num_envs)
    # The runner narrowed this at construction; the diag drives both modes
    # by hand, so the step path itself must stay on the full pass.
    env.terminal_obs_groups = None
    obs_man = env.obs_manager

    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    failures = 0
    for step in range(args.steps):
        env.step(0.1 * torch.randn_like(action))

        snap = _rng_snapshot()
        obs_man.process_observations(update_history=True, groups=None)
        crit_full = _clone_group(obs_man.obs_dict["critic"])
        obs_man.rollback_last_history_append(groups=None)

        _rng_restore(snap)
        obs_man.process_observations(update_history=True, groups=("critic",))
        crit_sub = _clone_group(obs_man.obs_dict["critic"])
        obs_man.rollback_last_history_append(groups=("critic",))
        _rng_restore(snap)

        bad = _compare_group(crit_full, crit_sub)
        if bad:
            failures += 1
            print(f"[step {step:3d}] MISMATCH in critic parts: {bad}")
        elif step % 10 == 0:
            print(f"[step {step:3d}] critic bitwise identical")

    if failures:
        print(f"\nFAIL — {failures}/{args.steps} steps mismatched")
        return 1
    print(f"\nPASS — critic group bit-identical between full and subset pass, {args.steps} steps ({args.sim})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
