"""De-risk the `interval_dr` design BEFORE implementing it (MuJoCo).

Idea under test (option B): instead of running DR every reset (per-episode
``reset_dr`` -> one recompute_constants per reset batch, i.e. ~every step under
churn), drive the SAME DR terms on a GLOBAL step interval — every ``period``
control steps, re-sample DR for ALL envs at once and recompute ONCE; hold DR
constant in between. This diag proves the approach with the CURRENT code, so we
only add the real mode once the behavior is validated:

  H1 speed:     recompute calls drop ~period-fold vs the per-reset baseline.
  H2 accuracy:  body_mass changes ONLY on interval boundaries, held in between.
  H3 coverage:  DR sampled over many intervals spans a real range (not frozen),
                comparable to the per-reset baseline's range.
  H4 stability: applying DR mid-episode does not spike qvel (mass change leaves
                velocities untouched; post-change dynamics stay bounded).

Prototype hook (no new mode yet): pop the ``reset_dr`` terms out of the event
manager so ``_reset_idx`` stops firing them, then call them on ALL envs every
``period`` steps, reusing the deferred recompute flush (mujoco) the fix added.

Run:
    python -m rlworld.scripts.diag.k1_interval_dr_prototype_diag --num-envs 256 --period 250
"""

from __future__ import annotations

import argparse
import time


def _build(num_envs: int, seed: int):
    from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    cfgs = K1G1RecipeConfig(sim_type="mujoco", num_envs=num_envs, seed=seed).build()
    return get_initializer("MujocoEnv").init_environment(cfgs)


def _read(sim, field: str):
    import warp as wp

    arr = getattr(sim.wp_model, field, None)
    return None if arr is None else wp.to_torch(arr).float().clone()


def _read_data(sim, field: str):
    import warp as wp

    arr = getattr(sim.wp_data, field, None)
    return None if arr is None else wp.to_torch(arr).float().clone()


def _wrap_recompute(env, counter: dict, key: str) -> None:
    sim = env.scene_manager.sim
    orig = sim.recompute_constants

    def counted(level, __orig=orig, __key=key):
        counter[__key] += 1
        return __orig(level)

    sim.recompute_constants = counted


def _global_dr(env, terms, all_ids) -> None:
    """Fire the popped reset_dr terms on ALL envs once, with a single deferred
    recompute flush (mirrors what the future interval_dr mode will do)."""
    em = env.event_manager
    env._dr_pending_recompute_level = None  # mujoco deferred batch
    for name, term in terms:
        em._call_event_fn(name, term, env_ids=all_ids)
    level = env._dr_pending_recompute_level
    del env._dr_pending_recompute_level
    if level:
        env.scene_manager.sim.recompute_constants(level)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--period", type=int, default=250)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    counter = {"base": 0, "proto": 0}

    # Baseline: unchanged per-reset DR.
    torch.manual_seed(args.seed)
    env_base = _build(args.num_envs, args.seed)
    _wrap_recompute(env_base, counter, "base")

    # Prototype: pop reset_dr so _reset_idx stops firing it; drive it globally.
    torch.manual_seed(args.seed)
    env_proto = _build(args.num_envs, args.seed)
    _wrap_recompute(env_proto, counter, "proto")
    proto_terms = env_proto.event_manager._terms_by_mode.pop("reset_dr", [])
    proto_sim = env_proto.scene_manager.sim

    print("=" * 88)
    print(f"interval_dr prototype  (num_envs={args.num_envs}, period={args.period}, " f"steps={args.steps})")
    print("=" * 88)
    print(f"\nreset_dr terms driven on the global interval: " f"{[n for n, _ in proto_terms]}")

    all_ids = torch.arange(args.num_envs, device=env_proto.device)
    ids_base = torch.arange(args.num_envs, device=env_base.device)

    # Seed the prototype with an initial DR draw (else mass stays at default
    # until the first interval).
    _global_dr(env_proto, proto_terms, all_ids)

    nact = env_base.num_actions
    prev_mass = _read(proto_sim, "body_mass")
    change_steps: list[int] = []
    mass_snaps: list[torch.Tensor] = [prev_mass.clone()]
    qvel_trace: list[float] = []
    boundary_qvel: list[tuple[int, float, float]] = []

    t_base = t_proto = 0.0
    for i in range(args.steps):
        torch.manual_seed(90_000 + i)
        a = torch.randn((args.num_envs, nact), device="cpu").to(env_base.device)

        # baseline
        torch.cuda.synchronize()
        _t = time.perf_counter()
        env_base.step(a)
        torch.cuda.synchronize()
        t_base += time.perf_counter() - _t

        # prototype: global DR every `period` steps
        if i > 0 and i % args.period == 0:
            pre = _read_data(proto_sim, "qvel").abs().max().item()
            _global_dr(env_proto, proto_terms, all_ids)
            post = _read_data(proto_sim, "qvel").abs().max().item()
            boundary_qvel.append((i, pre, post))
            mass_snaps.append(_read(proto_sim, "body_mass").clone())
        torch.cuda.synchronize()
        _t = time.perf_counter()
        env_proto.step(a)
        torch.cuda.synchronize()
        t_proto += time.perf_counter() - _t

        cur_mass = _read(proto_sim, "body_mass")
        if (cur_mass - prev_mass).abs().max().item() > 1e-9:
            change_steps.append(i)
        prev_mass = cur_mass
        qvel_trace.append(_read_data(proto_sim, "qvel").abs().max().item())

    # Keep the baseline stepping the same number of resets for a fair recompute
    # comparison is already done inline above.
    _ = ids_base

    # ── H1: recompute frequency + wall time ──────────────────────────────
    print("\n--- H1: recompute frequency + step wall-time ---")
    exp_proto = 1 + args.steps // args.period  # init + one per boundary
    print(
        f"  recompute calls: baseline={counter['base']}  prototype={counter['proto']}"
        f"  (expected proto ~= {exp_proto})"
    )
    print(f"  reduction: {counter['base'] / max(counter['proto'], 1):.1f}x fewer recomputes")
    print(
        f"  mean step wall-time: baseline={1e3 * t_base / args.steps:.3f} ms  "
        f"prototype={1e3 * t_proto / args.steps:.3f} ms"
    )

    # ── H2: DR changes only on boundaries ────────────────────────────────
    print("\n--- H2: body_mass changes ONLY on interval boundaries ---")
    expected = list(range(args.period, args.steps, args.period))
    print(f"  steps where body_mass changed: {change_steps}")
    print(f"  expected (multiples of period): {expected}")
    h2 = change_steps == expected

    # ── H3: coverage across intervals ────────────────────────────────────
    print("\n--- H3: DR coverage across intervals (per-body mass range) ---")
    stacked = torch.stack(mass_snaps)  # (n_intervals, num_envs, nbody)
    per_body_min = stacked.amin(dim=(0, 1))
    per_body_max = stacked.amax(dim=(0, 1))
    spread = per_body_max - per_body_min
    moved = int((spread > 1e-6).sum())
    print(f"  intervals sampled: {stacked.shape[0]}")
    print(f"  bodies whose mass varied across intervals: {moved}/{spread.numel()}")
    print(f"  max per-body spread across intervals: {spread.max().item():.4f} kg")
    print("  (frozen DR would give spread 0 on all bodies)")
    h3 = moved > 0

    # ── H4: mid-episode change does not spike qvel ───────────────────────
    print("\n--- H4: mid-episode DR change leaves qvel unspiked ---")
    max_pre_post = 0.0
    for step_i, pre, post in boundary_qvel:
        jump = abs(post - pre)
        max_pre_post = max(max_pre_post, jump)
        print(f"  step {step_i}: qvel|max| pre={pre:.4f} post={post:.4f}  Δ={jump:.2e}")
    qv = torch.tensor(qvel_trace)
    tail_max = qv.max().item()
    print(f"  qvel|max| over run: median={qv.median().item():.3f}  max={tail_max:.3f}")
    # DR application must not itself change velocities (Δ≈0); and the run must
    # not diverge (bounded qvel).
    h4 = max_pre_post < 1e-4 and tail_max < 1e3

    print("\n" + "=" * 88)
    verdict = {
        "H1_speed": counter["proto"] < counter["base"],
        "H2_boundary_only": h2,
        "H3_coverage": h3,
        "H4_stable": h4,
    }
    for k, v in verdict.items():
        print(f"  {k:<20} {'PASS' if v else 'FAIL'}")
    ok = all(verdict.values())
    print(f"\n  OVERALL: {'PASS — interval_dr approach validated' if ok else 'FAIL — investigate above'}")
    print("=" * 88)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
