"""Where does the GPU spend a control step, and how much of the step is it idle?

``env_step_breakdown`` splits the wall time of a step between the
sections of ``World.step``, synchronising around each. That says which
Python section the host spends its time in, not what the GPU does: a
section's time is the GPU work it launched plus the launches themselves
plus every wait for the device. When another framework runs the same
engine on the same robot faster, the question is whether the GPU itself
has more to do here (more or longer kernels), or whether it sits idle
while the host launches and waits.

This records every CUDA kernel of ``--steps`` un-instrumented control
steps with the torch profiler (CUPTI sees the simulator's kernels as
well as torch's) and reports, per control step: the wall time, the GPU
busy time (the union of the kernel intervals), the idle remainder, the
kernel count, and the split of the busy time between the simulator's
kernels and torch's, followed by the kernels that cost the most. A step
whose busy time is far below its wall time is bound by the host (launch
overhead and device synchronisations); one whose busy time fills the
wall is bound by the kernels themselves.

    jaxpy -m jaxrlworld.scripts.diag.perf.step_gpu_timeline --sim genesis --num-envs 4096
    jaxpy -m jaxrlworld.scripts.diag.perf.step_gpu_timeline --sim genesis --num-envs 4096 --no-stagger
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from jaxrlworld.scripts.diag.perf.env_step_breakdown import build_env

_SIM_KERNEL_MARKERS = ("kernel_", "func_", "quadrants", "taichi", "gs_")


def is_simulator_kernel(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _SIM_KERNEL_MARKERS) and "at::" not in lowered


def summarize_kernels(prof, steps: int, wall_ms: float) -> str:
    """Per-step GPU busy/idle split and the costliest kernels from a profiler run."""
    kernels = [ev for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA]
    intervals = sorted((ev.time_range.start, ev.time_range.end) for ev in kernels)
    busy_us = 0.0
    cur_start, cur_end = None, None
    for start, end in intervals:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                busy_us += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        busy_us += cur_end - cur_start

    by_name: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    sim_us = torch_us = 0.0
    for ev in kernels:
        dur = ev.time_range.elapsed_us()
        by_name[ev.name][0] += 1
        by_name[ev.name][1] += dur
        if is_simulator_kernel(ev.name):
            sim_us += dur
        else:
            torch_us += dur

    busy_ms = busy_us / steps / 1e3
    lines = [
        f"{'ms/ctrl-step':>13}  quantity",
        f"{wall_ms:13.3f}  wall (un-instrumented, measured before profiling)",
        f"{busy_ms:13.3f}  GPU busy (union of kernel intervals)",
        f"{wall_ms - busy_ms:13.3f}  GPU idle",
        f"{sim_us / steps / 1e3:13.3f}  kernel time, simulator kernels (sum; overlapping kernels count twice)",
        f"{torch_us / steps / 1e3:13.3f}  kernel time, torch kernels",
        f"{len(kernels) / steps:13.1f}  kernel launches per step",
        "",
        f"{'ms/step':>9}  {'calls/step':>10}  {'us/call':>8}  kernel",
    ]
    for name, (count, total) in sorted(by_name.items(), key=lambda item: -item[1][1])[:30]:
        lines.append(f"{total / steps / 1e3:9.3f}  {count / steps:10.1f}  {total / count:8.1f}  {name[:100]}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="jaxrlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig")
    ap.add_argument("--sim", default="genesis", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--no-stagger", action="store_true", help="start every episode together (no resets in the window)")
    ap.add_argument("--random-actions", action="store_true", help="uniform [-1, 1] actions instead of zeros")
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = ap.parse_args()

    env = build_env(args.config, args.sim, args.num_envs, args.set)
    env.reset()
    if not args.no_stagger:
        env.termination_manager.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

    def act():
        return actions.uniform_(-1.0, 1.0) if args.random_actions else actions

    for _ in range(args.warmup):
        env.step(act())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(act())
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / args.steps * 1e3

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            env.step(act())
        torch.cuda.synchronize()
    print(f"[timeline] {args.config} sim={args.sim} num_envs={args.num_envs} steps={args.steps}")
    print(summarize_kernels(prof, args.steps, wall_ms))


if __name__ == "__main__":
    main()
