"""Decompose the Genesis reset_path — which reset event term costs the 8x?

k1_step_speed_diag showed Genesis K1 is ~8x slower than mujoco, and the gap is
ENTIRELY the reset_path (genesis 161 ms/step vs mujoco 17), not the physics step
(genesis 20 vs mujoco 15). The reset_path is DR events (mode reset_dr) + reset
writers (mode reset). This diag times each of those event terms individually
(cuda-synced) so the 161 ms is attributed term-by-term.

Why Genesis specifically: the event manager batches Newton's per-term
``notify_model_changed`` into a single combined notify (each notify is a ~5 ms
GPU model refresh), but Genesis/MuJoCo do NOT batch — so on Genesis every DR
term that writes mass/armature/dofs_info/friction may trigger its own internal
recompute, and ~8 DR terms fire per reset. This diag proves whether that is the
cost and which terms dominate.

Synchronizing inside each term serializes async GPU work, so the summed
per-term time overstates wall time; the ATTRIBUTION (which term, relative cost)
is the takeaway, not the absolute total.

Run (server, from SimForge root):
    python -m jaxrlworld.scripts.diag.k1.k1_genesis_reset_decompose_diag --num_envs 4096
"""

from __future__ import annotations

import argparse


def run(num_envs: int, steps: int, warmup: int, seed: int) -> int:
    import torch

    from jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    torch.manual_seed(seed)
    print(f"[stage] building genesis K1 g1_recipe env (num_envs={num_envs})", flush=True)
    preset = K1G1RecipeConfig(sim_type="genesis", num_envs=num_envs, seed=seed)
    env = get_initializer("Genesis").init_environment(preset.build())
    env.reset()
    dev = env.device

    em = env.event_manager
    # Per-term (name -> mode) so the report groups reset_dr (DR) vs reset (writers).
    mode_of = {}
    for mode in ("reset", "reset_dr", "startup"):
        for name, _term in em._terms_by_mode.get(mode, []):
            mode_of[name] = mode

    acc: dict[str, float] = {}
    calls: dict[str, int] = {}

    def _wrap(name: str, orig):
        def timed(*a, **k):
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            r = orig(*a, **k)
            t1.record()
            torch.cuda.synchronize()
            acc[name] = acc.get(name, 0.0) + t0.elapsed_time(t1)  # ms
            calls[name] = calls.get(name, 0) + 1
            return r

        return timed

    # Patch the manager's resolved-fn table so every reset/reset_dr term is timed
    # exactly where the manager calls it (_call_event_fn reads _resolved_fns).
    for name in list(em._resolved_fns.keys()):
        if mode_of.get(name) in ("reset", "reset_dr"):
            em._resolved_fns[name] = _wrap(name, em._resolved_fns[name])

    n_act = env.num_actions
    resets = 0

    print(f"[stage] warmup {warmup} steps", flush=True)
    for _ in range(warmup):
        env.step(torch.rand(num_envs, n_act, device=dev) * 2.0 - 1.0)
    acc.clear()
    calls.clear()

    print(f"[stage] measuring {steps} random-action steps (drives resets)", flush=True)
    for _ in range(steps):
        obs, _rew, term, trunc, _extras = env.step(torch.rand(num_envs, n_act, device=dev) * 2.0 - 1.0)
        resets += int((term | trunc).sum())

    # ── report ────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"GENESIS reset_path decomposition — K1 g1_recipe | {num_envs} envs | {steps} steps")
    print("=" * 84)
    print(f"total resets over {steps} steps: {resets}  ({resets/steps:.1f} envs reset/step)")

    def _subtotal(mode):
        return sum(v for n, v in acc.items() if mode_of.get(n) == mode)

    dr_total = _subtotal("reset_dr")
    reset_total = _subtotal("reset")
    grand = dr_total + reset_total

    print(f"\n  {'term':26} {'mode':9} {'ms/step':>9} {'ms/reset':>9} {'calls':>6}  share")
    print("  " + "-" * 74)
    for name, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        ms_step = v / steps
        ms_reset = v / max(calls.get(name, 1), 1)
        share = 100.0 * v / max(grand, 1e-9)
        bar = "#" * int(share / 2)
        print(
            f"  {name:26} {mode_of.get(name, '?'):9} {ms_step:9.3f} {ms_reset:9.3f} "
            f"{calls.get(name, 0):6d}  {share:4.0f}% {bar}"
        )
    print("  " + "-" * 74)
    print(f"  {'SUBTOTAL reset_dr (DR)':26} {'':9} {dr_total/steps:9.3f}")
    print(f"  {'SUBTOTAL reset (writers)':26} {'':9} {reset_total/steps:9.3f}")
    print(f"  {'GRAND (timed event terms)':26} {'':9} {grand/steps:9.3f}")

    print("\n[interpretation]")
    top = max(acc.items(), key=lambda kv: kv[1]) if acc else ("-", 0.0)
    print(f"  Biggest single term: {top[0]} ({top[1]/steps:.2f} ms/step).")
    if dr_total > 2.0 * reset_total:
        print("  DR events dominate the reset. On Genesis each mass/armature/dofs_info/")
        print("  friction write can trigger an internal recompute (no combined-notify")
        print("  batching like Newton). Candidate fixes: move build-params (mass, armature)")
        print("  to mode='startup' (the _build_dr_terms docstring's stated intent), and/or")
        print("  batch the Genesis model refresh across DR terms.")
    else:
        print("  Reset WRITERS (reset_root / reset_joints) are a large share — the Genesis")
        print("  state-write path itself, not DR, is the cost. Optimize the writers.")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return run(args.num_envs, args.steps, args.warmup, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
