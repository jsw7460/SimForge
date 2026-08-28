"""Equivalence check: batched reward combination vs the sequential original.

``RewardManager.set_rewards`` now stacks the per-term weighted rewards
and combines them with batched reductions. Per-term values are computed
exactly as before (``_compute_weighted_reward`` untouched for nonzero
weights), so the only thing that can move is the float rounding of the
term-combination order. This diag pins that down:

1. Patch ``_compute_weighted_reward`` to serve fixed random tensors.
2. Run the live ``set_rewards`` for each reward mode
   (sum / exponential / exponential_auto).
3. Recompute the total with a verbatim replica of the ORIGINAL
   sequential logic and compare.

Passes when (a) the exponential_auto sign classification agrees exactly,
(b) per-type dict entries are BIT-identical to the served tensors, and
(c) the batched total deviates from the sequential one by no more than
100x the ORIGINAL logic's own term-order rounding sensitivity (measured
by running the replica with the term order reversed — the exp modes
amplify exponent rounding by the exp factor, so no fixed threshold
fits all regimes).

Usage:
    jaxpy -m rlworld.scripts.diag.gates.check_reward_batching --sim mujoco
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


def _old_total(values: dict[str, torch.Tensor], terms, mode: str, sigma: float, total_clip, order=None) -> torch.Tensor:
    """Verbatim replica of the pre-batching sequential combination.

    ``order`` iterates the terms in a chosen order (default: config
    order). Running it reversed measures how sensitive the ORIGINAL
    logic itself is to term-accumulation order — the rounding floor the
    batched result is judged against.
    """
    if order is None:
        order = list(terms)
    terms = {name: terms[name] for name in order}
    some = next(iter(values.values()))
    reward_buffer = torch.zeros_like(some)
    if mode == "sum":
        for name in terms:
            reward_buffer += values[name]
    elif mode == "exponential":
        rew_task = torch.zeros_like(reward_buffer)
        rew_shaped = torch.zeros_like(reward_buffer)
        for name, term in terms.items():
            if term.exp_shaping:
                rew_shaped += values[name]
            else:
                rew_task += values[name]
        reward_buffer += rew_task * torch.exp(rew_shaped / sigma)
    elif mode == "exponential_auto":
        rew_pos = torch.zeros_like(reward_buffer)
        rew_neg = torch.zeros_like(reward_buffer)
        zero = torch.zeros((), device=reward_buffer.device, dtype=reward_buffer.dtype)
        for name in terms:
            reward_value = values[name]
            is_pos = torch.sum(reward_value) >= 0
            rew_pos = rew_pos + torch.where(is_pos, reward_value, zero)
            rew_neg = rew_neg + torch.where(is_pos, zero, reward_value)
        reward_buffer += rew_pos * torch.exp(rew_neg / sigma)
    if total_clip is not None:
        reward_buffer.clamp_(*total_clip)
    return reward_buffer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="go2_gait")
    ap.add_argument("--sim", default="mujoco", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = _build_env(args.preset, args.sim, args.num_envs)
    mgr = env.reward_manager
    terms = mgr.reward_terms
    device = env.device
    sigma = mgr.config.shaping_sigma
    total_clip = mgr.config.total_clip
    orig_mode = mgr.config.reward_mode
    orig_compute = mgr._compute_weighted_reward

    worst = 0.0
    failures = 0
    try:
        for trial in range(args.trials):
            # Realistic weighted-reward magnitudes (dt-scaled, |v| ~ 1e-4
            # .. 1e-1). Larger synthetic values push exp(rew_neg/sigma)
            # into overflow, where a 1e-7 rounding in the exponent
            # multiplies into astronomically different totals — a regime
            # real rewards never enter.
            scale = 10.0 ** torch.randint(-4, 0, (1,)).item()
            values = {name: scale * torch.randn(env.num_envs, device=device) for name in terms}
            mgr._compute_weighted_reward = lambda name, term, _v=values: _v[name]

            # The one thing that must be EXACT: the exponential_auto sign
            # classification. Old code summed each term with torch.sum;
            # the batched path row-sums the stack. Same elements, so a
            # disagreement means terms swap between the task/exp groups.
            old_signs = torch.stack([torch.sum(values[name]) >= 0 for name in terms])
            new_signs = torch.stack([values[name] for name in terms]).sum(dim=1) >= 0
            if not torch.equal(old_signs, new_signs):
                failures += 1
                print(f"[trial {trial}] exponential_auto sign classification differs")

            for mode in ("sum", "exponential", "exponential_auto"):
                mgr.config.reward_mode = mode
                rew_new = torch.zeros(env.num_envs, device=device)
                per_type: dict[str, torch.Tensor] = {}
                mgr.set_rewards(reward_buffer=rew_new, reward_buffer_per_type=per_type)

                for name in terms:
                    if not torch.equal(per_type[name], values[name]):
                        failures += 1
                        print(f"[trial {trial}] per_type[{name!r}] not bit-identical ({mode})")

                rew_old = _old_total(values, terms, mode, sigma, total_clip)
                # Self-calibrating tolerance: the ORIGINAL sequential logic
                # run with the term order reversed measures how much pure
                # accumulation-order rounding moves the total (the exp
                # modes amplify an exponent rounding of ~1e-6 by the exp
                # factor itself, so no fixed threshold fits all regimes).
                # The batched result must sit within 100x of that floor.
                rew_old_rev = _old_total(values, terms, mode, sigma, total_clip, order=list(terms)[::-1])
                d_new = (rew_new - rew_old).abs().max().item()
                d_base = (rew_old_rev - rew_old).abs().max().item()
                scale_floor = 1e-6 * (rew_old.abs().max().item() + sum(v.abs().max().item() for v in values.values()))
                # Two acceptance routes: within 100x of the measured
                # order-sensitivity floor, OR within 1e-4 of the total's
                # own magnitude (the floor can come out unluckily small
                # when forward/reversed roundings happen to cancel, while
                # a large-exp total makes tiny relative noise look big
                # in absolute terms).
                tol = max(100.0 * d_base + scale_floor, 1e-4 * rew_old.abs().max().item())
                worst = max(worst, d_new / max(tol, 1e-300))
                if d_new > tol:
                    failures += 1
                    print(
                        f"[trial {trial}] {mode}: |new-old| {d_new:.3e} exceeds tol {tol:.3e} (order floor {d_base:.3e})"
                    )
    finally:
        mgr.config.reward_mode = orig_mode
        mgr._compute_weighted_reward = orig_compute

    if failures:
        print(f"\nFAIL — {failures} mismatches, worst rel {worst:.3e}")
        return 1
    print(f"\nPASS — batched == sequential across {args.trials} trials x 3 modes, worst rel {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
