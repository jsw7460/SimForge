"""Genesis getter-call profile: how many taichi reads does one env.step pay?

Candidate-3 measurement (see genesis_g1_step_perf memory): on Genesis every
``RobotData`` property is a direct entity/solver getter — one taichi kernel
launch + sync + copy per CALL, so the wrapper cost scales with the NUMBER of
getter calls per step, not with the data volume.  Before batching them we
measure exactly which getters fire, how often, and what they cost.

Method: build the g1 (or any) genesis preset env, wrap every state getter on
the robot entity + rigid solver + native sensors with a counting/timing
shim (device-synced so each call's cost is attributed to it), then run a
random-action rollout and print a per-getter table:

    calls/step  x  ms/call  =  ms/step  (share of the instrumented total)

The wrapping is diagnostic-only (bound-method shims on live instances);
framework code is untouched.  Absolute numbers are inflated by the forced
per-call sync — read the CALL COUNTS and the relative shares.

Usage (GPU box):
    python -m rlworld.scripts.diag.perf.genesis_getter_profile_diag
    python -m rlworld.scripts.diag.perf.genesis_getter_profile_diag --preset go2 --num-envs 4096
"""

from __future__ import annotations

import argparse
import functools
import importlib
import time
from collections import defaultdict
from pathlib import Path

_PRESETS: dict[str, tuple[str, str]] = {
    "go2": ("rlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("rlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
    "t1_tracking": ("rlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
    "t1_getup": ("rlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
}

# Getter names to instrument on the robot RigidEntity.
_ENTITY_GETTERS = (
    "get_pos",
    "get_quat",
    "get_vel",
    "get_ang",
    "get_dofs_position",
    "get_dofs_velocity",
    "get_dofs_force",
    "get_dofs_control_force",
    "get_links_pos",
    "get_links_quat",
    "get_links_vel",
    "get_links_ang",
    "get_links_net_contact_force",
)
# Getter names to instrument on the rigid solver (root/body CoM reads go through it).
_SOLVER_GETTERS = (
    "get_links_pos",
    "get_links_vel",
    "get_links_COM_shift",
)


class _Meter:
    def __init__(self):
        self.calls: dict[str, int] = defaultdict(int)
        self.secs: dict[str, float] = defaultdict(float)
        self.enabled = False

    def wrap(self, obj, name: str, label: str) -> None:
        fn = getattr(obj, name, None)
        if fn is None:
            return

        import torch

        @functools.wraps(fn)
        def shim(*a, **kw):
            if not self.enabled:
                return fn(*a, **kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            self.secs[label] += time.perf_counter() - t0
            self.calls[label] += 1
            return out

        setattr(obj, name, shim)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="g1_29dof", choices=sorted(_PRESETS))
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="genesis_getter_profile.txt")
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    module, cls_name = _PRESETS[args.preset]
    cfg_cls = getattr(importlib.import_module(module), cls_name)
    cfgs = cfg_cls(sim_type="genesis", num_envs=args.num_envs, seed=args.seed).build()

    from rlworld.rl.evals.sim_initializers import get_initializer

    env = get_initializer("Genesis").init_environment(cfgs)
    env.reset()
    print(f"[STAGE] env built ({args.preset}, num_envs={args.num_envs})", flush=True)

    meter = _Meter()
    robot = env.scene_manager.robot
    solver = robot._solver
    for name in _ENTITY_GETTERS:
        meter.wrap(robot, name, f"entity.{name}")
    for name in _SOLVER_GETTERS:
        meter.wrap(solver, name, f"solver.{name}")
    # Contact-list backend: one shared collider read per substep (all groups).
    meter.wrap(env.contact_manager._list_reader, "raw", "contactlist.raw")
    meter.wrap(solver.collider, "get_contacts", "collider.get_contacts")

    actions = torch.zeros((args.num_envs, env.num_actions), device=env.device)

    for _ in range(args.warmup):
        actions.uniform_(-1.0, 1.0)
        env.step(actions)
    torch.cuda.synchronize()

    meter.enabled = True
    t0 = time.perf_counter()
    for _ in range(args.num_steps):
        actions.uniform_(-1.0, 1.0)
        env.step(actions)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    meter.enabled = False

    n = args.num_steps
    total_instrumented = sum(meter.secs.values())
    lines: list[str] = []
    lines.append("=" * 104)
    lines.append(f"Genesis getter-call profile — preset={args.preset} num_envs={args.num_envs} steps={n}")
    lines.append("=" * 104)
    lines.append(f"env.step wall: {wall / n * 1e3:.2f} ms/step (instrumentation-inflated)")
    lines.append(f"instrumented getters total: {total_instrumented / n * 1e3:.2f} ms/step")
    lines.append("")
    lines.append(f"{'getter':<40}{'calls/step':>12}{'ms/call':>10}{'ms/step':>10}{'share':>8}")
    lines.append("-" * 104)
    for label in sorted(meter.secs, key=lambda k: -meter.secs[k]):
        calls = meter.calls[label]
        secs = meter.secs[label]
        lines.append(
            f"{label:<40}{calls / n:>12.1f}{secs / calls * 1e3:>10.3f}"
            f"{secs / n * 1e3:>10.3f}{secs / total_instrumented * 100:>7.1f}%"
        )
    lines.append("-" * 104)
    total_calls = sum(meter.calls.values())
    lines.append(f"{'TOTAL':<40}{total_calls / n:>12.1f}{'':>10}{total_instrumented / n * 1e3:>10.3f}{100.0:>7.1f}%")
    lines.append("")
    lines.append("[Reading] high calls/step with small payloads = batching candidates;")
    lines.append(
        "          theoretical floor per step: get_links_{pos,quat,vel,ang} + get_dofs_{position,velocity} = 6 bulk calls."
    )

    report = "\n".join(lines)
    Path(args.out).write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
