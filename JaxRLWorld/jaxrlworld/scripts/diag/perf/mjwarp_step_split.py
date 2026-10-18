"""Where does a backend's physics substep go: the mjwarp step itself, or
the wrapper around it?

The mujoco (mjlab) and newton backends both integrate with MuJoCo Warp,
yet their ``scene_manager.step`` costs differ (K1 8192: 3.7 against
6.1 ms per substep on the 10-17 baseline). A substep is two things:

- ``mujoco_warp.step(model, data)`` — the engine proper, whose cost is
  set by the model it was handed (bodies, geoms, mesh vertices, pairs,
  solver options, contact / constraint budgets);
- the wrapper — mjlab launches the step graph and its sensor graph;
  Newton's ``SolverMuJoCo.step`` also pushes the Newton state and
  controls into the mjwarp buffers, runs the step, and pulls body /
  joint state and observables back out, every substep.

This times both on the LIVE model and data of a built env, from the same
warmed-up state, and prints the model facts side by side so a difference
in the raw step can be read against the model it ran on. Run it once per
backend; the two outputs are the comparison.

Usage (GPU box, from the SimForge root):
    jaxpy -m jaxrlworld.scripts.diag.perf.mjwarp_step_split --preset k1_velocity --sim newton
    jaxpy -m jaxrlworld.scripts.diag.perf.mjwarp_step_split --preset k1_velocity --sim mujoco
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import time

import mujoco
import mujoco_warp
import numpy as np
import torch
import warp as wp

from jaxrlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
from jaxrlworld.rl.configs.presets.go2.base import Go2FlatConfig
from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig
from jaxrlworld.rl.runners import BaseRunner

_PRESETS = {"go2": Go2FlatConfig, "g1_29dof": G1FlatConfig, "k1_velocity": K1VelocityConfig}


def _enum_names(enum, prefix: str) -> dict[int, str]:
    """pybind11 enums are not iterable; ``__members__`` is the roster."""
    return {int(v): n.removeprefix(prefix).lower() for n, v in enum.__members__.items() if n.startswith(prefix)}


_GEOM_NAMES = _enum_names(mujoco.mjtGeom, "mjGEOM_")
_CONE_NAMES = _enum_names(mujoco.mjtCone, "mjCONE_")
_DISABLE_BITS = _enum_names(mujoco.mjtDisableBit, "mjDSBL_")
_ENABLE_BITS = _enum_names(mujoco.mjtEnableBit, "mjENBL_")


def _bit_names(flags: int, bits: dict[int, str]) -> str:
    names = [name for bit, name in bits.items() if flags & bit]
    return "+".join(names) if names else "none"


def _scalar(value):
    """A per-world option array prints as its distinct values."""
    if isinstance(value, wp.array):
        arr = value.numpy()
        if arr.size == 1:
            return arr.item()
        distinct = np.unique(arr)
        return distinct.tolist() if distinct.size <= 4 else f"{distinct.size} distinct in [{arr.min()}, {arr.max()}]"
    return value


def _handles(env, sim: str):
    """(mjwarp Model, mjwarp Data) of the built env."""
    sm = env.scene_manager
    if sim == "mujoco":
        # ``sm.model`` / ``sm.data`` are mjlab's torch-view bridges; the
        # raw warp objects the step kernels take live on the Simulation.
        return sm.sim.wp_model, sm.sim.wp_data
    return sm.solver.mjw_model, sm.solver.mjw_data


def _time_graph(fn, device, launches: int, repeats: int) -> float:
    """Median ms per launch of ``fn`` captured once into a CUDA graph."""
    with wp.ScopedDevice(device):
        fn()  # compile every kernel outside the capture
        wp.synchronize_device(device)
        with wp.ScopedCapture() as capture:
            fn()
        graph = capture.graph
        samples = []
        for _ in range(repeats):
            wp.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(launches):
                wp.capture_launch(graph)
            wp.synchronize_device(device)
            samples.append((time.perf_counter() - t0) * 1e3 / launches)
    return statistics.median(samples)


def _time_calls(fn, launches: int, repeats: int) -> float:
    """Median ms per call of ``fn`` (the backend's own step, already graphed)."""
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(launches):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3 / launches)
    return statistics.median(samples)


def _print_model_facts(m, d) -> None:
    geom_types = m.geom_type.numpy()
    counts = np.bincount(geom_types, minlength=len(_GEOM_NAMES))
    hist = ", ".join(f"{_GEOM_NAMES[i]} {n}" for i, n in enumerate(counts) if n)
    nefc = d.nefc.numpy()
    nacon = int(d.nacon.numpy()[0])
    worldid = d.contact.worldid.numpy()[:nacon]
    per_world = np.bincount(worldid, minlength=d.nworld) if nacon else np.zeros(d.nworld, dtype=int)
    opt = m.opt
    rows = [
        ("nworld", d.nworld),
        ("nbody / nv / ngeom", f"{m.nbody} / {m.nv} / {m.ngeom}"),
        ("geom types", hist),
        ("nmesh / nmeshvert / nmeshface", f"{m.nmesh} / {m.nmeshvert} / {m.nmeshface}"),
        ("nhfield / nhfielddata", f"{m.nhfield} / {m.nhfielddata}"),
        ("npair / nexclude", f"{m.npair} / {m.nexclude}"),
        ("nxn geom pairs (all / filtered)", f"{m.nxn_geom_pair.shape[0]} / {m.nxn_geom_pair_filtered.shape[0]}"),
        ("geom_pair_type_count", m.geom_pair_type_count),
        ("naconmax / njmax / nvmax", f"{d.naconmax} / {d.njmax} / {d.nvmax}"),
        ("njmax_pad / njmax_nnz / nvmax_pad", f"{d.njmax_pad} / {d.njmax_nnz} / {d.nvmax_pad}"),
        ("contacts now: total / per world mean / max", f"{nacon} / {per_world.mean():.1f} / {per_world.max()}"),
        ("nefc now: per world mean / max", f"{nefc.mean():.1f} / {nefc.max()}"),
        ("opt.timestep", _scalar(opt.timestep)),
        ("opt.iterations / ls_iterations", f"{opt.iterations} / {opt.ls_iterations}"),
        ("opt.tolerance / ls_tolerance", f"{_scalar(opt.tolerance)} / {_scalar(opt.ls_tolerance)}"),
        ("opt.cone", _CONE_NAMES[int(opt.cone)]),
        ("opt.impratio_invsqrt", _scalar(opt.impratio_invsqrt)),
        ("opt.ccd_iterations", _scalar(opt.ccd_iterations)),
        ("opt.disableflags", _bit_names(int(opt.disableflags), _DISABLE_BITS)),
        ("opt.enableflags", _bit_names(int(opt.enableflags), _ENABLE_BITS)),
        ("opt.broadphase / broadphase_filter", f"{opt.broadphase!s} / {opt.broadphase_filter!s}"),
        ("opt.run_collision_detection", opt.run_collision_detection),
        ("opt.warn_overflow", opt.warn_overflow),
    ]
    print("=" * 78)
    print("MJWARP MODEL AND DATA FACTS  (after warmup)")
    print("=" * 78)
    for name, value in rows:
        print(f"  {name:<44}{value}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="k1_velocity", help=f"{', '.join(_PRESETS)} or module.path:ClassName")
    ap.add_argument("--sim", default="newton", choices=("mujoco", "newton"))
    ap.add_argument("--num-envs", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=24, help="env steps with random actions before timing")
    ap.add_argument("--launches", type=int, default=100, help="substeps per timing sample")
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    if ":" in args.preset:
        mod_path, cls_name = args.preset.split(":", 1)
        cfg_cls = importlib.import_module(mod_path).__dict__[cls_name]
    else:
        cfg_cls = _PRESETS[args.preset]
    cfgs = cfg_cls(sim_type=args.sim, num_envs=args.num_envs).build()
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    env = runner.env

    torch.manual_seed(0)
    action = torch.zeros(env.num_envs, env.act_manager.num_actions, device=env.device)
    for _ in range(args.warmup):
        env.step(0.1 * torch.randn_like(action))
    torch.cuda.synchronize()

    m, d = _handles(env, args.sim)
    print()
    print(
        f"  preset {args.preset}  sim {args.sim}  envs {env.num_envs}  substeps per scene step {env.scene_manager.config.substeps}"
    )
    print()
    _print_model_facts(m, d)

    device = d.qpos.device
    # mjwarp's convex narrowphase allocates its multiccd scratch
    # (naccdmax x polygon vertices) on every call outside a graph; hand
    # torch's cached blocks back first so that allocation has room next to
    # the backend's own graph.
    torch.cuda.empty_cache()
    raw = _time_graph(lambda: mujoco_warp.step(m, d), device, args.launches, args.repeats)
    wrapped = _time_calls(env.scene_manager.step, args.launches, args.repeats)
    print()
    print("=" * 78)
    print("ONE PHYSICS SUBSTEP, SPLIT  (median ms)")
    print("=" * 78)
    print(f"  {'mujoco_warp.step on this model/data (own graph)':<52}{raw:8.3f}")
    print(f"  {'scene_manager.step (the backend, as collect_breakdown)':<52}{wrapped:8.3f}")
    print(f"  {'wrapper = scene_manager.step - raw step':<52}{wrapped - raw:8.3f}")
    print("=" * 78)
    if args.sim == "newton":
        solver = env.scene_manager.solver
        print(f"  newton SolverMuJoCo.update_data_interval = {solver.update_data_interval}")
        print("  (the wrapper is state -> mjwarp, controls, step, mjwarp -> state, observables, sensors)")
    else:
        print("  (the wrapper is the step graph launch plus the sensor graph)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
