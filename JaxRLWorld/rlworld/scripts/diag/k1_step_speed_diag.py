"""K1 per-component step-speed comparison across the three backends.

Symptom: K1 training on Newton runs ~2x slower than on mjlab even
though both wrap the SAME mjwarp solver with the SAME solver options,
so the difference must live in the per-step wrapping around the mjwarp
step (state/control conversion, un-graphed sensor extraction, reset
path), not in the physics.

Method: monkeypatch the per-step components of each backend with
cuda-synchronized timers and run two phases at training scale:

- zero-action steps (robot stands; almost no resets) — pure step cost
- random-action steps (mass terminations; training-like) — adds the
  reset path (events + managers + post-reset contact refresh)

Synchronizing inside the loop serializes async GPU work, so the parts
add up to more than the unsynced wall time; the UNSYNCED total is the
real throughput number, the synced breakdown is the attribution.

Run (server, from SimForge root):
    python -m rlworld.scripts.diag.k1_step_speed_diag --num-envs 4096
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "rlworld.scripts.diag.k1_step_speed_diag"
_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_WARMUP = 30
_STEPS = 200


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


class _Timers:
    """Accumulates cuda-synced durations per named component."""

    def __init__(self) -> None:
        self.acc: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.enabled = False

    def wrap(self, obj, attr: str, name: str) -> None:
        import torch

        orig = getattr(obj, attr)

        def timed(*args, __orig=orig, __name=name, **kwargs):
            if not self.enabled:
                return __orig(*args, **kwargs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = __orig(*args, **kwargs)
            torch.cuda.synchronize()
            self.acc[__name] = self.acc.get(__name, 0.0) + (time.perf_counter() - t0)
            self.calls[__name] = self.calls.get(__name, 0) + 1
            return out

        setattr(obj, attr, timed)

    def wrap_by_mode(self, obj, attr: str, name_prefix: str) -> None:
        """Like ``wrap`` but splits accumulation by the call's ``mode`` arg
        (e.g. event_manager.apply(mode="reset") vs mode="reset_dr" vs the
        step-loop's mode="interval"). One timer per distinct mode."""
        import torch

        orig = getattr(obj, attr)

        def timed(*args, __orig=orig, __prefix=name_prefix, **kwargs):
            if not self.enabled:
                return __orig(*args, **kwargs)
            mode = kwargs.get("mode")
            if mode is None and args:
                mode = args[0]
            name = f"{__prefix}_{mode}"
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = __orig(*args, **kwargs)
            torch.cuda.synchronize()
            self.acc[name] = self.acc.get(name, 0.0) + (time.perf_counter() - t0)
            self.calls[name] = self.calls.get(name, 0) + 1
            return out

        setattr(obj, attr, timed)

    def reset(self) -> None:
        self.acc.clear()
        self.calls.clear()

    def report(self, n_steps: int) -> dict:
        return {
            name: {"ms_per_step": round(1e3 * total / n_steps, 3), "calls": self.calls[name]}
            for name, total in sorted(self.acc.items(), key=lambda kv: -kv[1])
        }


def _host_int(x) -> int:
    """Scalar host read of a torch tensor / warp array / tensor proxy."""
    import torch
    import warp as wp

    if isinstance(x, torch.Tensor):
        return int(x.max().item())
    if isinstance(x, wp.array):
        return int(x.numpy().max())
    t = x.detach()
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"unsupported array type: {type(x)}")
    return int(t.max().item())


def _measure(
    env, timers: _Timers, num_envs: int, steps: int, random_actions: bool, seed: int, counters_fn=None
) -> dict:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    zero = torch.zeros((num_envs, env.num_actions), device=env.device)

    peaks: dict[str, int] = {}
    timers.reset()
    timers.enabled = True
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    resets = 0
    for _ in range(steps):
        if random_actions:
            a = torch.randn((num_envs, env.num_actions), generator=gen, device="cpu").to(env.device)
        else:
            a = zero
        _o, _r, term_b, trunc_b, _e = env.step(a)
        resets += int((term_b | trunc_b).sum())
        if counters_fn is not None:
            for k, v in counters_fn().items():
                peaks[k] = max(peaks.get(k, 0), v)
    torch.cuda.synchronize()
    synced_total = time.perf_counter() - t0
    timers.enabled = False

    # Unsynced pass for the true throughput (component syncs disabled).
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        if random_actions:
            a = torch.randn((num_envs, env.num_actions), generator=gen, device="cpu").to(env.device)
        else:
            a = zero
        env.step(a)
    torch.cuda.synchronize()
    unsynced_total = time.perf_counter() - t0

    ms = 1e3 * unsynced_total / steps
    return {
        "ms_per_step_unsynced": round(ms, 3),
        "fps_env_steps": int(num_envs * steps / unsynced_total),
        "ms_per_step_synced_total": round(1e3 * synced_total / steps, 3),
        "resets_in_timed_window": resets,
        "peak_counters": peaks,
        "components_ms_per_step": timers.report(steps),
    }


# Solver knobs: each maps a single JaxRLWorld K1 value -> its mjlab-velocity
# value, so we can flip ONE at a time from the asis baseline and isolate which
# knob dominates the mujoco physics-step cost.
#   asis  (JaxRLWorld): 100/50 iters, elliptic, impratio 100, 2 substeps
#   mjlab (velocity)  : 10/20 iters,  pyramidal, impratio 1,   1 substep
_SOLVER_KNOBS = ("iters", "ls", "impratio", "cone", "substeps")


def _apply_solver_knob(scene_cfg, knob: str) -> None:
    """Flip ONE solver knob (or all, for ``mjlab``) from asis -> mjlab value.

    Only the named knob changes; num_envs, managers, obs, reward, decimation
    stay fixed, so the ``mj_scene_step`` delta vs the asis run is that single
    knob's contribution to the physics-step cost.
    """
    mj = scene_cfg.mjlab_sim_cfg.mujoco
    if knob in ("iters", "mjlab"):
        mj.iterations = 10
    if knob in ("ls", "mjlab"):
        mj.ls_iterations = 20
    if knob in ("impratio", "mjlab"):
        mj.impratio = 1.0
    if knob in ("cone", "mjlab"):
        mj.cone = "pyramidal"
    if knob in ("substeps", "mjlab"):
        scene_cfg.substeps = 1  # 8 -> 4 mjwarp steps / control step (decimation 4)
        mj.timestep = scene_cfg.physics_dt  # substeps=1 -> full physics_dt per step
    if knob not in (*_SOLVER_KNOBS, "mjlab"):
        raise ValueError(f"unknown solver knob: {knob}")


def run_cell(
    sim: str,
    num_envs: int,
    seed: int,
    mujoco_solver: str = "asis",
    reset_decompose: bool = False,
    dr_interval_period: float | None = None,
) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(
        f"cell start: {sim} num_envs={num_envs} mujoco_solver={mujoco_solver} "
        f"dr_interval_period={dr_interval_period}"
    )

    from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    kwargs = dict(sim_type=sim, num_envs=num_envs, seed=seed)
    if dr_interval_period is not None:
        # <=0 forces per-reset reset_dr (baseline); >0 sets the interval seconds.
        kwargs["dr_interval_period_s"] = None if dr_interval_period <= 0 else dr_interval_period
    cfgs = K1G1RecipeConfig(**kwargs).build()
    if mujoco_solver != "asis":
        if sim != "mujoco":
            raise ValueError("--mujoco-solver only applies to the mujoco cell")
        _apply_solver_knob(cfgs.scene, mujoco_solver)
        _stage(f"applied solver knob '{mujoco_solver}' (asis -> mjlab value)")
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    _stage("env built")

    timers = _Timers()
    sm = env.scene_manager
    # Shared components (present on every backend).
    timers.wrap(env.act_manager, "apply_actions", "act_apply")
    timers.wrap(env.contact_manager, "advance", "contact_advance")
    timers.wrap(env, "_reset_idx", "reset_path")
    timers.wrap(env.obs_manager, "process_observations", "obs_compute")
    # Backend-specific step internals.
    if sim == "newton":
        # scene.step = CUDA-graph replay + _update_sensors; the sensor
        # patch fires inside it, so graph time = nt_scene_step minus
        # nt_sensor_extract.
        timers.wrap(sm, "step", "nt_scene_step")
        timers.wrap(sm, "_update_sensors", "nt_sensor_extract")
        timers.wrap(env.contact_manager, "refresh_after_reset", "nt_reset_refresh")
    elif sim == "mujoco":
        timers.wrap(sm, "write_data_to_sim", "mj_write_data")
        timers.wrap(sm, "step", "mj_scene_step")
        timers.wrap(sm, "update", "mj_scene_update")
    else:
        timers.wrap(sm, "step", "gs_scene_step")
        timers.wrap(env.contact_manager, "refresh_after_reset", "gs_reset_refresh")

    # reset_path decomposition: split _reset_idx into its sub-calls (world.py).
    # ``rd_`` prefix; event_manager.apply is split by mode (rd_ev_reset,
    # rd_ev_reset_dr, and the step-loop's rd_ev_interval). These are timed
    # INSIDE reset_path, so sum(rd_* reset-side) ~= reset_path.
    if reset_decompose:
        timers.wrap(env.curriculum_manager, "compute", "rd_curr_compute")
        timers.wrap(env.event_manager, "reset", "rd_ev_statereset")
        timers.wrap_by_mode(env.event_manager, "apply", "rd_ev")
        for _mname, _short in (
            ("termination_manager", "term"),
            ("command_manager", "cmd"),
            ("act_manager", "act"),
            ("obs_manager", "obs"),
            ("reward_manager", "rew"),
            ("curriculum_manager", "curr"),
        ):
            timers.wrap(getattr(env, _mname), "reset", f"rd_{_short}_reset")
        timers.wrap(env.contact_manager, "reset", "rd_contact_reset")
        # newton/genesis already time refresh_after_reset (nt_/gs_reset_refresh);
        # only mujoco needs it added here to avoid double-wrapping.
        if sim == "mujoco":
            timers.wrap(env.contact_manager, "refresh_after_reset", "rd_contact_refresh")

    def _host_geom_stats(mj_model) -> dict[str, int]:
        import numpy as np

        collidable = (np.asarray(mj_model.geom_contype) != 0) | (np.asarray(mj_model.geom_conaffinity) != 0)
        return {
            "ngeom": int(mj_model.ngeom),
            "n_collidable_geoms": int(collidable.sum()),
            "npair": int(mj_model.npair),
            "nexclude": int(mj_model.nexclude),
        }

    counters_fn = None
    capacities: dict[str, int] = {}
    if sim == "newton":
        _d = sm.solver.mjw_data
        capacities = {"naconmax": int(_d.naconmax), "njmax": int(_d.njmax), "nworld": int(_d.nworld)}
        capacities.update(_host_geom_stats(sm.solver.mj_model))
        counters_fn = lambda d=_d: {"nacon_peak": _host_int(d.nacon), "nefc_peak_per_world": _host_int(d.nefc)}  # noqa: E731
    elif sim == "mujoco":
        _d = sm.data
        capacities = {"naconmax": int(_d.naconmax), "njmax": int(_d.njmax), "nworld": int(_d.nworld)}
        capacities.update(_host_geom_stats(sm.mj_model))
        counters_fn = lambda d=_d: {"nacon_peak": _host_int(d.nacon), "nefc_peak_per_world": _host_int(d.nefc)}  # noqa: E731

    _stage(f"capacities: {capacities}")

    zero = torch.zeros((num_envs, env.num_actions), device=env.device)
    for _ in range(_WARMUP):
        env.step(zero)
    _stage("warmup done")

    out: dict = {"sim": sim, "num_envs": num_envs, "mjwarp_capacities": capacities}
    out["zero_actions"] = _measure(
        env, timers, num_envs, _STEPS, random_actions=False, seed=seed + 1, counters_fn=counters_fn
    )
    _stage("zero-action phase done")
    out["random_actions"] = _measure(
        env, timers, num_envs, _STEPS, random_actions=True, seed=seed + 2, counters_fn=counters_fn
    )
    _stage("random-action phase done")
    return out


# ── Parent orchestration ─────────────────────────────────────────────


def _run_mujoco_variants(args, variants: tuple[str, ...]) -> int:
    """Run the mujoco cell once per solver variant (each is asis with exactly
    one knob flipped to its mjlab value; ``asis`` = none, ``mjlab`` = all).
    Everything else is held fixed, so each variant's ``mj_scene_step`` drop vs
    asis isolates that single knob's share of the physics-step cost."""
    logdir = Path.cwd() / "k1_step_speed_diag_logs"
    logdir.mkdir(exist_ok=True)
    results: dict = {}
    for v in variants:
        print(f"[sweep] mujoco solver={v} num_envs={args.num_envs} ...", flush=True)
        t0 = time.time()
        log_path = logdir / f"mujoco_{v}.log"
        with open(log_path, "w") as fh:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    "--cell",
                    "mujoco",
                    "--num-envs",
                    str(args.num_envs),
                    "--seed",
                    str(args.seed),
                    "--mujoco-solver",
                    v,
                ],
                stdout=fh,
                stderr=subprocess.STDOUT,
                cwd=str(Path.cwd()),
            )
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"[sweep]   -> CRASH (see {log_path}) ({dt:.0f}s)")
            continue
        for line in log_path.read_text().splitlines():
            if line.startswith("RESULT_JSON:"):
                results[v] = json.loads(line[len("RESULT_JSON:") :])
        print(f"[sweep]   -> ok ({dt:.0f}s)")

    print("\n" + "=" * 100)
    print(f"K1 mujoco solver-knob sweep (num_envs={args.num_envs}; ms per control step)")
    print("  asis = JaxRLWorld baseline: 100/50 iters, elliptic, impratio 100, 2 substeps")
    print("  each knob flips ONE value to mjlab-velocity: iters->10 ls->20 impratio->1 " "cone->pyramidal substeps->1")
    print("  mjlab = all knobs at once. 'physics saved' = asis mj_scene_step - variant (bigger = more critical)")
    print("=" * 100)
    if "asis" not in results:
        print("[sweep] MISSING asis baseline (crashed) — cannot compute deltas.")
        return 1
    for phase in ("zero_actions", "random_actions"):
        base = results["asis"][phase]["components_ms_per_step"]["mj_scene_step"]["ms_per_step"]
        print(f"\n[{phase}]  asis mj_scene_step baseline = {base} ms")
        print(f"    {'variant':<12}{'mj_scene_step':>16}{'physics saved':>16}{'env total':>14}")
        # Rank knobs by how much physics they save (most critical first).
        order = (
            ["asis"]
            + sorted(
                (v for v in variants if v not in ("asis", "mjlab")),
                key=lambda v: -(base - _phys(results, v, phase)) if v in results else 0,
            )
            + (["mjlab"] if "mjlab" in variants else [])
        )
        for v in order:
            if v not in results:
                print(f"    {v:<12}{'MISSING (crashed)':>16}")
                continue
            phys = _phys(results, v, phase)
            tot = results[v][phase]["ms_per_step_unsynced"]
            saved = "—" if v == "asis" else f"{base - phys:+.3f}"
            print(f"    {v:<12}{_fmt(phys):>16}{saved:>16}{_fmt(tot):>14}")
    print(
        "\nInterpretation: the knob with the largest 'physics saved' is the single"
        "\nmost critical driver of the mujoco physics-step cost. impratio+cone interact"
        "\n(high impratio + elliptic keeps the Newton solver from converging early, so"
        "\nthe 100 iters are actually spent); substeps halves the mjwarp step COUNT."
    )
    return 0


def _phys(results: dict, v: str, phase: str) -> float:
    return results[v][phase]["components_ms_per_step"]["mj_scene_step"]["ms_per_step"]


def _fmt(x) -> str:
    return "-" if x is None else str(x)


def _ratio(a, b) -> str:
    if a is None or b is None or not b:
        return "-"
    return f"{a / b:.2f}x"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=_SIMS)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mujoco-solver",
        choices=("asis", *_SOLVER_KNOBS, "mjlab"),
        default="asis",
        help="Solver variant for the mujoco cell: asis (JaxRLWorld baseline), a "
        "single knob flipped to its mjlab value, or mjlab (all knobs).",
    )
    ap.add_argument(
        "--mujoco-knob-sweep",
        action="store_true",
        help="Run the mujoco cell once per solver variant (asis, each single knob, "
        "mjlab) and print which knob is most critical to the physics-step cost.",
    )
    ap.add_argument(
        "--mujoco-ab",
        action="store_true",
        help="Run ONLY asis vs mjlab (both extremes) for the mujoco cell.",
    )
    ap.add_argument(
        "--reset-decompose",
        action="store_true",
        help="Split reset_path (_reset_idx) into sub-components (rd_* rows: "
        "DR reset_dr, per-manager resets, contact refresh) to find what dominates.",
    )
    ap.add_argument(
        "--dr-interval-period",
        type=float,
        default=None,
        help="Override dr_interval_period_s: >0 = interval_dr seconds, 0 = per-reset "
        "reset_dr (baseline), omit = preset default (g1_recipe = 10s).",
    )
    args = ap.parse_args()

    if args.cell:
        result = run_cell(
            args.cell,
            args.num_envs,
            args.seed,
            args.mujoco_solver,
            args.reset_decompose,
            args.dr_interval_period,
        )
        print("RESULT_JSON:" + json.dumps(result))
        return 0

    if args.mujoco_knob_sweep:
        return _run_mujoco_variants(args, ("asis", *_SOLVER_KNOBS, "mjlab"))
    if args.mujoco_ab:
        return _run_mujoco_variants(args, ("asis", "mjlab"))

    logdir = Path.cwd() / "k1_step_speed_diag_logs"
    logdir.mkdir(exist_ok=True)
    results: dict = {}
    for sim in _SIMS:
        print(f"[diag] running {sim} ...", flush=True)
        t0 = time.time()
        log_path = logdir / f"{sim}.log"
        with open(log_path, "w") as fh:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    "--cell",
                    sim,
                    "--num-envs",
                    str(args.num_envs),
                    "--seed",
                    str(args.seed),
                ]
                + (["--reset-decompose"] if args.reset_decompose else [])
                + (
                    ["--dr-interval-period", str(args.dr_interval_period)]
                    if args.dr_interval_period is not None
                    else []
                ),
                stdout=fh,
                stderr=subprocess.STDOUT,
                cwd=str(Path.cwd()),
            )
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"[diag]   -> CRASH (see {log_path}) ({dt:.0f}s)")
            continue
        for line in log_path.read_text().splitlines():
            if line.startswith("RESULT_JSON:"):
                results[sim] = json.loads(line[len("RESULT_JSON:") :])
        print(f"[diag]   -> ok ({dt:.0f}s)")

    print("\n" + "=" * 100)
    print(f"K1 g1recipe step-speed breakdown ({_STEPS} steps per phase; ms per control step)")
    print("=" * 100)
    for sim in _SIMS:
        if sim not in results:
            print(f"\n[{sim}] MISSING (crashed)")
            continue
        r = results[sim]
        if r.get("mjwarp_capacities"):
            print(f"\n[{sim}] mjwarp capacities: {r['mjwarp_capacities']}")
        for phase in ("zero_actions", "random_actions"):
            p = r[phase]
            print(
                f"\n[{sim}][{phase}] total={p['ms_per_step_unsynced']} ms/step  "
                f"({p['fps_env_steps']} env-steps/s)  synced_total={p['ms_per_step_synced_total']} ms  "
                f"resets={p['resets_in_timed_window']}"
            )
            if p.get("peak_counters"):
                print(f"    peaks: {p['peak_counters']}")
            for name, d in p["components_ms_per_step"].items():
                print(f"    {name:<20} {d['ms_per_step']:>8} ms/step  ({d['calls']} calls)")
    print(
        "\nInterpretation: newton vs mujoco share the same mjwarp solver+options, so any wall-clock"
        "\ngap must show up in the component rows (sensor extraction, state conversion, reset path)"
        "\nor in the solver-step row itself (graph granularity / collision-pipeline flags)."
    )
    return 0 if len(results) == len(_SIMS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
