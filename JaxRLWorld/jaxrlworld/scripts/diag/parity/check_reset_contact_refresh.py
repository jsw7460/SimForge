"""Does a partial reset leave the other envs' contact history untouched?

After ``_reset_idx(env_ids)`` the contact manager refreshes its sensors for
the freshly written poses. A refresh that pushes a history frame for EVERY
env hands the non-reset envs a duplicate of this substep's frame and drops
their oldest one, and a refresh that clears every env's narrowphase warm
start changes what the non-reset envs' next substep detects: a reset in env
A then moves what env B's rewards read. This diag steps a preset, snapshots
every env's contact force history and contact state, resets half of the envs
the way ``World.step`` does, and checks:

1. non-reset envs: force history bit-identical to the snapshot (hard),
   ``is_contact`` unchanged (reported);
2. reset envs: at most one non-zero history slot, the fresh frame.

Usage (any backend):

    jaxpy -m jaxrlworld.scripts.diag.parity.check_reset_contact_refresh --preset go2_flat --sim newton
    jaxpy -m jaxrlworld.scripts.diag.parity.check_reset_contact_refresh --preset go2_flat --sim genesis
    jaxpy -m jaxrlworld.scripts.diag.parity.check_reset_contact_refresh --preset g1_flat --sim mujoco
"""

from __future__ import annotations

import argparse

import torch

from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.scripts.diag.gates.check_all_presets import _PUBLIC, _load


def _snapshot(env) -> dict[str, tuple[torch.Tensor | None, torch.Tensor]]:
    out = {}
    for name in env.contact_manager.group_names():
        history = env.contact_manager.contact_force_history(name)
        out[name] = (None if history is None else history.clone(), env.contact_manager.is_contact(name).clone())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", required=True, help="label from check_all_presets' table")
    ap.add_argument("--sim", required=True, choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--settle-steps", type=int, default=20, help="random-action steps before the reset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    loader = next(loader for label, loader, sims in _PUBLIC if label == args.preset and args.sim in sims)
    cfgs = _load(loader, args.sim, args.num_envs)
    env = BaseRunner.create_with_env(cfgs, use_wandb=False).env
    env.reset()
    gen = torch.Generator().manual_seed(args.seed)
    for _ in range(args.settle_steps):
        actions = (torch.randn((env.num_envs, env.num_actions), generator=gen) * 0.5).to(env.device)
        env.step(actions)

    before = _snapshot(env)
    reset_ids = torch.arange(0, env.num_envs, 2, device=env.device)
    keep_ids = torch.arange(1, env.num_envs, 2, device=env.device)
    # World.step's reset sequence: _reset_idx, cache bump, the step's forward.
    env._reset_idx(reset_ids)
    env._invalidate_cache()
    env._post_reset_forward()
    env._invalidate_cache()
    after = _snapshot(env)

    print(f"{args.preset}:{args.sim}  num_envs={env.num_envs}  reset={len(reset_ids)}  kept={len(keep_ids)}")
    failed = False
    for name in before:
        hist_b, contact_b = before[name]
        hist_a, contact_a = after[name]
        flips = int((contact_a[keep_ids] != contact_b[keep_ids]).sum())
        print(f"  [{name}]")
        print(f"    kept envs: is_contact flips = {flips}")
        if hist_b is None:
            print("    no force history on this backend for this group")
            continue
        kept_same = torch.equal(hist_a[keep_ids], hist_b[keep_ids])
        slots = (hist_a[reset_ids].abs().sum(dim=-1) > 0).sum(dim=-1)  # (n_reset, N) non-zero slots
        max_slots = int(slots.max()) if slots.numel() else 0
        print(f"    kept envs: history bit-identical = {kept_same}")
        print(f"    reset envs: non-zero history slots per body max = {max_slots} (want <= 1)")
        if not kept_same:
            diff = (hist_a[keep_ids] - hist_b[keep_ids]).abs()
            print(f"      max |delta| = {float(diff.max()):.3e}, slots differing = {int((diff.sum(-1) > 0).sum())}")
            failed = True
        if max_slots > 1:
            failed = True

    # The env must keep stepping after a refresh that wrote only part of the rings.
    actions = (torch.randn((env.num_envs, env.num_actions), generator=gen) * 0.5).to(env.device)
    env.step(actions)
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
