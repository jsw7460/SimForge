"""bench_step_speed — layered wall-clock comparison of physics-step
speed across Genesis / Newton / MuJoCo on a chosen preset.

Builds each env via ``presets.<preset>.mlp.get_config(sim=...)`` →
``apply_overrides({"env": {"num_envs": N}})`` → standard env class
(``GenesisEnv`` / ``NewtonEnv`` / ``MjlabEnv``), runs ``warmup`` steps,
then times four layers per ``(sim, num_envs)`` with
``torch.cuda.synchronize()`` at every boundary so each measurement
reflects real GPU work.

Layers (per sim, per ``num_envs``):

* **L1** — bare ``scene_manager.step()`` only.  Pure solver substep
  cost.  Skips action write / contact-manager advance / reward / obs.
* **L2** — ``env._step_physics()`` — decimation ×
  (apply_actions + scene.step + contact_manager.advance + sim-specific
  bookkeeping).  This is what ``World.step``'s ``phys:`` sections cover.
* **L3** — L2 + ``reward_manager.set_rewards(...)``.
* **L4** — full ``env.step(random_action)`` — obs / termination / reset
  / extras included.

The four layers run *in order* on the same env without resets in
between; reads at later layers reflect the cumulative simulation state
from earlier layers' steps.  This is a benchmark, not a parity check —
we just need representative physics work happening.

Output is written to
``<SimForge>/bench_outputs/bench_step_speed_<preset>_<timestamp>.txt``
unless overridden via ``--output``.

NOTE: do *not* set ``JAXRLWORLD_PROFILE_STEP=1`` while benching — it
adds extra ``torch.cuda.synchronize()`` boundaries inside
``World.step`` and biases the L3 / L4 numbers upward.

The ``--genesis_loose`` flag overrides ``scene.rigid_options.tolerance``
(``1e-8 → 1e-5``) and ``contact_pruning_tolerance`` (``None → 0.02``)
*for the bench only* — it never touches the preset file on disk.  Used
to isolate how much of the Genesis cost comes from the preset asking
for higher solver accuracy than Genesis's own published benchmarks.
The override is benchmark-only and policy-impacting: a checkpoint
trained against the strict preset will see slightly different
constraint solutions if this flag is left on for evaluation.
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

# ── timing helpers ──────────────────────────────────────────────────


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_loop(fn: Callable[[], None], n_iters: int) -> list[float]:
    """Run ``fn`` ``n_iters`` times with CUDA-sync boundaries.

    Returns per-iter wall-clock in milliseconds.
    """
    times: list[float] = []
    for _ in range(n_iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def _stats(samples: list[float]) -> tuple[float, float, float, float]:
    """Return ``(mean, p50, p95, max)`` in ms."""
    n = len(samples)
    if n == 0:
        return (float("nan"),) * 4
    sorted_samples = sorted(samples)
    mean = statistics.fmean(sorted_samples)
    p50 = sorted_samples[n // 2]
    p95 = sorted_samples[min(n - 1, int(n * 0.95))]
    mx = sorted_samples[-1]
    return mean, p50, p95, mx


# ── env build ───────────────────────────────────────────────────────


_GENESIS_LOOSE_OVERRIDES: dict = {
    # Genesis default ``tolerance`` is ``1e-5`` (auto for single precision);
    # the G1 preset asks for ``1e-8``, which is 1000× tighter and forces the
    # Newton constraint solver into many more inner iterations per substep.
    "tolerance": 1e-5,
    # Genesis default ``contact_pruning_tolerance`` is ``0.02``; the G1 preset
    # sets it to ``None`` which disables culling, so every detected contact
    # (including near-zero-force ones) enters the constraint system.
    "contact_pruning_tolerance": 0.02,
}

# Mirror of ``Genesis/tests/test_rigid_benchmarks.py::make_g1_fall`` — the
# upstream ``g1_fall`` benchmark that the parametrized row at
# ``test_rigid_benchmarks.py:1100`` runs at ``n_envs=4096``.  Use this to
# put our Genesis path on the same operating point as Genesis's own
# published G1 numbers.
_GENESIS_G1_BENCH_OVERRIDES: dict = {
    "iterations": 10,  # G1 bench: 10 (vs preset 50)
    "ls_iterations": 20,  # G1 bench: 20 (vs preset 50)
    "tolerance": 1e-5,  # G1 bench: 1e-5 (vs preset 1e-8)
    "contact_pruning_tolerance": 0.02,  # Genesis default (vs preset None)
    "max_collision_pairs": 150,  # Genesis default (vs preset 100)
    # ``constraint_solver`` stays at Newton — the G1 bench parametrize row
    # explicitly passes ``constraint_solver=Newton``, matching our preset.
    # ``enable_self_collision`` stays True (Genesis default) — both setups
    # keep this on for G1.
}


def _apply_rigid_overrides(cfgs, overrides: dict) -> dict:
    """Override ``scene.rigid_options`` fields and return the
    ``{field: (was, now)}`` diff actually applied.

    The preset file on disk is *not* touched — this lives only in this
    process's ``cfgs.scene.rigid_options`` instance.  Uses pydantic-v2
    ``model_copy(update=...)`` since ``RigidOptions`` is a Genesis
    pydantic model that may reject in-place attribute writes once frozen.
    """
    opts = cfgs.scene.rigid_options
    diff: dict = {}
    for field, new_value in overrides.items():
        old_value = getattr(opts, field, "<missing>")
        diff[field] = (old_value, new_value)
    cfgs.scene.rigid_options = opts.model_copy(update=overrides)
    return diff


def build_env(
    preset_module_name: str,
    sim: str,
    num_envs: int,
    genesis_loose: bool = False,
    match_genesis_g1_bench: bool = False,
):
    """Build an env for ``sim`` using ``presets.<preset>.mlp.get_config``
    with ``num_envs`` and viewer-off overrides applied.

    Genesis-only override modes (mutually exclusive, applied only when
    ``sim == "genesis"``):

    * ``genesis_loose`` — restore Genesis defaults for the two fields
      our preset tightens beyond their documented defaults
      (``tolerance``, ``contact_pruning_tolerance``).
    * ``match_genesis_g1_bench`` — match Genesis's own ``make_g1_fall``
      benchmark options (``iterations=10``, ``ls_iterations=20``,
      ``tolerance=1e-5``, ``contact_pruning_tolerance=0.02``,
      ``max_collision_pairs=150``).  Use this to put our path on the
      same operating point as Genesis's published G1 numbers.

    The applied diff (if any) is attached to the returned env as
    ``_bench_loose_diff`` so the report can record what changed.
    """
    preset_mod = importlib.import_module(f"rlworld.rl.configs.presets.{preset_module_name}.mlp")
    cfgs = preset_mod.get_config(sim=sim)
    cfgs.apply_overrides(env={"num_envs": num_envs})
    # Force viewer off — the bench cares about physics + obs cost, not
    # rendering.  Set a safe long episode so resets don't bias timings.
    cfgs.apply_overrides(env={"episode_length_s": 10e9})
    cfgs.visualization.show_viewer = False
    cfgs.visualization.record_video = False
    # ``viewer_type = "none"`` skips ViserVisualizationManager / GL
    # viewer construction inside each env's ``_build_scene``.
    if hasattr(cfgs.visualization, "viewer_type"):
        cfgs.visualization.viewer_type = "none"

    loose_diff: dict | None = None
    override_mode: str | None = None
    if sim == "genesis":
        if genesis_loose and match_genesis_g1_bench:
            raise ValueError("--genesis_loose and --match_genesis_g1_bench are mutually exclusive.")
        if match_genesis_g1_bench:
            loose_diff = _apply_rigid_overrides(cfgs, _GENESIS_G1_BENCH_OVERRIDES)
            override_mode = "match_genesis_g1_bench"
        elif genesis_loose:
            loose_diff = _apply_rigid_overrides(cfgs, _GENESIS_LOOSE_OVERRIDES)
            override_mode = "genesis_loose"

    from rlworld.rl import envs

    env_class_name = cfgs.env.env_name
    env_class = getattr(envs, env_class_name)
    kwargs = dict(
        num_envs=cfgs.env.num_envs,
        env_cfg=cfgs.env,
        scene_cfg=cfgs.scene,
        visualization_cfg=cfgs.visualization,
        obs_cfg=cfgs.observation,
        act_cfg=cfgs.action,
        reward_cfg=cfgs.reward,
        command_cfg=cfgs.command,
        event_cfg=cfgs.event,
        curriculum_cfg=cfgs.curriculum,
    )
    gait_cfg = getattr(cfgs, "gait", None)
    if gait_cfg is not None:
        kwargs["gait_cfg"] = gait_cfg
    env = env_class(**kwargs)
    # Attach the override diff + mode label (if any) so the report can record it.
    env._bench_loose_diff = loose_diff
    env._bench_override_mode = override_mode
    return env


# ── benchmark ───────────────────────────────────────────────────────


def bench_env(env, num_steps: int, warmup: int, action_seed: int = 0) -> dict:
    """Run L1..L4 measurements on ``env``.  Returns ``{layer: stats}``."""
    num_envs = env.num_envs
    num_actions = env.act_manager.num_actions
    device = env.device

    # CPU-side generator so action sampling is deterministic and does
    # not consume GPU work that would skew bench numbers.
    gen = torch.Generator(device="cpu").manual_seed(action_seed)

    def random_action() -> torch.Tensor:
        return torch.empty((num_envs, num_actions), device="cpu").uniform_(-0.5, 0.5, generator=gen).to(device)

    env.reset()
    for _ in range(warmup):
        env.step(random_action())

    # ── L4: full env.step ────────────────────────────────────────────
    l4 = _time_loop(lambda: env.step(random_action()), num_steps)

    # Pre-set processed actions once; L1/L2/L3 reuse them.
    env.act_manager.process_actions(random_action())

    # ── L3: physics + reward ─────────────────────────────────────────
    def step_l3() -> None:
        env._step_physics()
        env.reward_manager.set_rewards(
            reward_buffer=env.rew_buf,
            episode_sums=env.episode_sums,
            reward_buffer_per_type=env.rew_buf_per_type,
        )

    l3 = _time_loop(step_l3, num_steps)

    # ── L2: physics only ─────────────────────────────────────────────
    l2 = _time_loop(env._step_physics, num_steps)

    # ── L1: bare solver substep ──────────────────────────────────────
    # Skips ``apply_actions`` / ``contact_manager.advance`` / etc.; whatever
    # was last written to ``control.joint_*`` is reused for every call.
    # This is the closest fair "solver only" measurement across the three
    # sims — they all expose ``scene_manager.step`` as their canonical
    # one-substep solver entry point.
    l1 = _time_loop(env.scene_manager.step, num_steps)

    return {"L1": _stats(l1), "L2": _stats(l2), "L3": _stats(l3), "L4": _stats(l4)}


# ── report ──────────────────────────────────────────────────────────


def _fmt_row(label: str, stats: tuple[float, float, float, float]) -> str:
    mean, p50, p95, mx = stats
    return f"  {label:<24s} mean={mean:8.3f}  p50={p50:8.3f}  " f"p95={p95:8.3f}  max={mx:8.3f}  ms"


def _build_report(
    preset: str,
    sims: list[str],
    num_envs_list: list[int],
    num_steps: int,
    warmup: int,
    meta: dict,
    results: dict,
    loose_diff: dict | None = None,
    override_mode: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f" bench_step_speed — preset={preset}")
    lines.append(f" generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f" num_steps={num_steps}  warmup={warmup}  " f"cuda={torch.cuda.is_available()}")
    if loose_diff:
        lines.append(f" Genesis rigid_options override mode: {override_mode}")
        for field, (was, now) in loose_diff.items():
            lines.append(f"   scene.rigid_options.{field}: {was!r} -> {now!r}")
    else:
        lines.append(" Genesis rigid_options override mode: <none> (preset as-is)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Per-sim timing config:")
    for sim in sims:
        m = meta.get(sim)
        if m is None:
            lines.append(f"  {sim:<8s} <not built>")
            continue
        lines.append(
            f"  {sim:<8s} decimation={m['decimation']}  "
            f"physics_dt={m['physics_dt']:.4f}s  "
            f"control_dt={m['control_dt']:.4f}s  "
            f"num_actions={m['num_actions']}"
        )
    lines.append("")

    for sim in sims:
        lines.append("─" * 78)
        lines.append(f"== {sim} ==")
        for num_envs in num_envs_list:
            r = results.get(sim, {}).get(num_envs)
            lines.append("")
            lines.append(f"  num_envs = {num_envs}")
            if r is None:
                lines.append("    <not measured>")
                continue
            if "error" in r:
                lines.append(f"    BUILD/BENCH FAILED: {r['error']}")
                continue
            lines.append(_fmt_row("L1 bare scene.step", r["L1"]))
            lines.append(_fmt_row("L2 _step_physics", r["L2"]))
            lines.append(_fmt_row("L3 + reward", r["L3"]))
            lines.append(_fmt_row("L4 full env.step", r["L4"]))
        lines.append("")

    # Realtime ratio at num_envs=1, when measured.
    if 1 in num_envs_list:
        lines.append("─" * 78)
        lines.append("Realtime ratio at num_envs=1 (control_dt / L4.mean):")
        for sim in sims:
            r = results.get(sim, {}).get(1)
            m = meta.get(sim)
            if not r or "error" in r or not m:
                continue
            l4_mean_ms = r["L4"][0]
            cdt_ms = m["control_dt"] * 1e3
            ratio = cdt_ms / max(l4_mean_ms, 1e-9)
            verdict = "realtime OK" if ratio >= 1.0 else "below realtime"
            lines.append(
                f"  {sim:<8s} L4={l4_mean_ms:7.3f} ms  "
                f"control_dt={cdt_ms:6.3f} ms  ratio={ratio:5.2f}x  ({verdict})"
            )
        lines.append("")

    # Genesis-style metric on L1 (bare ``scene.step``): aggregated env-steps
    # per second of wall clock × physics_dt = realtime factor.  Matches the
    # formula in Genesis's ``run_benchmark``
    # (``Genesis/tests/test_rigid_benchmarks.py::run_benchmark``):
    # ``runtime_fps = num_steps × n_envs / elapsed`` and
    # ``realtime_factor = runtime_fps × step_dt``.  Reporting on L1 keeps the
    # comparison apples-to-apples — Genesis's bench step_fn only calls
    # ``scene.step()``, not a full RL loop.
    lines.append("─" * 78)
    lines.append("Genesis-style metric on L1 (bare scene.step):")
    lines.append("  runtime_fps     = num_envs × 1000 / L1.mean  (aggregated env-steps/sec)")
    lines.append("  realtime_factor = runtime_fps × physics_dt    (×realtime)")
    lines.append("")
    header = f"  {'sim':<8s} {'n_envs':>8s} {'L1.mean':>10s} {'runtime_fps':>14s} {'realtime':>12s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for sim in sims:
        m = meta.get(sim)
        if not m:
            continue
        phys_dt = m["physics_dt"]
        for num_envs in num_envs_list:
            r = results.get(sim, {}).get(num_envs)
            if not r or "error" in r:
                continue
            l1_mean_ms = r["L1"][0]
            runtime_fps = num_envs * 1e3 / max(l1_mean_ms, 1e-9)
            realtime_factor = runtime_fps * phys_dt
            lines.append(
                f"  {sim:<8s} {num_envs:>8d} {l1_mean_ms:>9.3f}ms " f"{runtime_fps:>13.0f} {realtime_factor:>11.1f}x"
            )
    lines.append("")

    # Per-env throughput at the largest swept num_envs, if any > 1.
    biggest = max(num_envs_list) if num_envs_list else 0
    if biggest > 1:
        lines.append("─" * 78)
        lines.append(
            f"Per-env throughput at num_envs={biggest} "
            "(1000 / L4.mean, env-steps/sec — one env.step advances every env once):"
        )
        for sim in sims:
            r = results.get(sim, {}).get(biggest)
            if not r or "error" in r:
                continue
            l4_mean_ms = r["L4"][0]
            tput = 1e3 / max(l4_mean_ms, 1e-9)
            lines.append(f"  {sim:<8s} L4={l4_mean_ms:7.3f} ms  " f"throughput={tput:8.1f} env-steps/sec/env")
        lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────


def _safe_close(env) -> None:
    """Best-effort env shutdown.  Sim-specific cleanup varies; ignore
    any teardown error so a single failing close doesn't tank the bench.
    """
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layered physics step speed bench across simulators.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="g1_29dof",
        help="Preset package under ``rlworld.rl.configs.presets`` (default: g1_29dof).",
    )
    parser.add_argument(
        "--sim",
        type=str,
        nargs="+",
        choices=["genesis", "newton", "mujoco"],
        default=["genesis", "newton", "mujoco"],
        help="Simulators to benchmark (default: all three).",
    )
    parser.add_argument(
        "--num_envs_list",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="num_envs values to sweep (default: 1 4 16 64).",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Timed steps per layer (default: 200).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup steps before timing (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=("Output .txt path.  Default: " "``<SimForge>/bench_outputs/bench_step_speed_<preset>_<timestamp>.txt``."),
    )
    genesis_mode = parser.add_mutually_exclusive_group()
    genesis_mode.add_argument(
        "--genesis_loose",
        action="store_true",
        help=(
            "Relax ``scene.rigid_options`` for Genesis only (``tolerance`` "
            "1e-8 → 1e-5, ``contact_pruning_tolerance`` None → 0.02) — "
            "matches Genesis's own published-benchmark defaults for the "
            "two fields the G1 preset over-tightens."
        ),
    )
    genesis_mode.add_argument(
        "--match_genesis_g1_bench",
        action="store_true",
        help=(
            "Override ``scene.rigid_options`` for Genesis only to match "
            "Genesis's own ``make_g1_fall`` benchmark options exactly "
            "(``iterations=10``, ``ls_iterations=20``, ``tolerance=1e-5``, "
            "``contact_pruning_tolerance=0.02``, ``max_collision_pairs=150``). "
            "Use together with ``--num_envs_list 4096`` to put our Genesis "
            "path on the same operating point as Genesis's published G1 "
            "numbers in ``Genesis/tests/test_rigid_benchmarks.py``."
        ),
    )
    args = parser.parse_args()

    # ── resolve output path ──────────────────────────────────────────
    if args.output is None:
        # ``__file__`` is ``<SimForge>/JaxRLWorld/rlworld/scripts/diag/bench_step_speed.py``.
        # parents[4] is ``<SimForge>``.
        repo_root = Path(__file__).resolve().parents[4]
        out_dir = repo_root / "bench_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"bench_step_speed_{args.preset}_{ts}.txt"
    else:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── run benches ──────────────────────────────────────────────────
    results: dict = {}
    meta: dict = {}

    applied_loose_diff: dict | None = None
    applied_override_mode: str | None = None

    for sim in args.sim:
        results[sim] = {}
        for num_envs in args.num_envs_list:
            print(
                f"[bench] sim={sim} num_envs={num_envs}: building env ...",
                flush=True,
            )
            env = None
            try:
                env = build_env(
                    args.preset,
                    sim,
                    num_envs,
                    genesis_loose=args.genesis_loose,
                    match_genesis_g1_bench=args.match_genesis_g1_bench,
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(
                    f"[bench] sim={sim} num_envs={num_envs}: BUILD FAILED — {err}",
                    flush=True,
                )
                traceback.print_exc()
                results[sim][num_envs] = {"error": err}
                continue
            # Record per-sim timing config once (first successful build).
            if sim not in meta:
                meta[sim] = {
                    "decimation": env.decimation,
                    "physics_dt": float(env.physics_dt),
                    "control_dt": float(env.control_dt),
                    "num_actions": int(env.act_manager.num_actions),
                }
            # Remember the override diff + mode once for the report.
            if applied_loose_diff is None and getattr(env, "_bench_loose_diff", None) is not None:
                applied_loose_diff = env._bench_loose_diff
                applied_override_mode = getattr(env, "_bench_override_mode", None)
            print(
                f"[bench] sim={sim} num_envs={num_envs}: timing "
                f"({args.num_steps} steps × 4 layers + {args.warmup} warmup) ...",
                flush=True,
            )
            try:
                results[sim][num_envs] = bench_env(env, num_steps=args.num_steps, warmup=args.warmup)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(
                    f"[bench] sim={sim} num_envs={num_envs}: BENCH FAILED — {err}",
                    flush=True,
                )
                traceback.print_exc()
                results[sim][num_envs] = {"error": err}
            finally:
                _safe_close(env)
                del env

    # ── write report ─────────────────────────────────────────────────
    report = _build_report(
        preset=args.preset,
        sims=args.sim,
        num_envs_list=args.num_envs_list,
        num_steps=args.num_steps,
        warmup=args.warmup,
        meta=meta,
        results=results,
        loose_diff=applied_loose_diff,
        override_mode=applied_override_mode,
    )
    print()
    print(report)
    out_path.write_text(report)
    print(f"\n[bench] report written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
