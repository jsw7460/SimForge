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


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs}")

    from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    cfgs = K1G1RecipeConfig(sim_type=sim, num_envs=num_envs, seed=seed).build()
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=_SIMS)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cell:
        result = run_cell(args.cell, args.num_envs, args.seed)
        print("RESULT_JSON:" + json.dumps(result))
        return 0

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
                ],
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
