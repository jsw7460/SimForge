"""Genesis per-step read-cache verification.

``GenesisRobotData`` properties are memoized per cache generation
(``_per_step_read``), and ``GenesisEnv._step_physics`` bumps the generation
after every substep so the explicit-actuator PD path still sees fresh joint
state inside the decimation loop.  This diag proves both properties on a
live rollout (contact sensors are ring-served by the contact-list backend
and verified separately by ``genesis_contact_list_wiring_diag``):

    1. FRESHNESS / PARITY: after every control step, every cached quantity is
       compared (torch.equal) against a direct, cache-bypassing read of the
       same underlying getter — a stale or mis-keyed cache fails immediately.
    2. PD SUBSTEP FRESHNESS: ``entity.get_dofs_position`` is counted during
       env.step; with a correct per-substep bump the explicit PD path must
       re-read joint state every substep, so calls/step >= decimation.  (A
       broken cache that serves stale state inside the loop would show
       calls/step < decimation.)  Only applies to presets with explicit
       actuators (``term.has_explicit_actuators``); implicit-PD presets
       (Genesis-internal ``control_dofs_position``) never read joint state
       on the action path, so the check is reported as N/A there.
    3. DEDUP EFFECT: reports calls/step per getter for comparison against the
       pre-change measurement (genesis_getter_profile_diag: get_quat ~11.6,
       get_dofs_position ~8.6).

Usage (GPU box):
    python -m jaxrlworld.scripts.diag.perf.genesis_read_cache_diag
    python -m jaxrlworld.scripts.diag.perf.genesis_read_cache_diag --preset go2 --num-envs 1024
"""

from __future__ import annotations

import argparse
import functools
import importlib
import time
from collections import defaultdict
from pathlib import Path

_PRESETS: dict[str, tuple[str, str]] = {
    "go2": ("jaxrlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "g1_29dof": ("jaxrlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("jaxrlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
    "t1_tracking": ("jaxrlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
    "t1_getup": ("jaxrlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="g1_29dof", choices=sorted(_PRESETS))
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="genesis_read_cache_diag.txt")
    args = ap.parse_args()

    import genesis as gs
    import torch

    torch.manual_seed(args.seed)
    module, cls_name = _PRESETS[args.preset]
    cfg_cls = getattr(importlib.import_module(module), cls_name)
    cfgs = cfg_cls(sim_type="genesis", num_envs=args.num_envs, seed=args.seed).build()

    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    env = get_initializer("Genesis").init_environment(cfgs)
    env.reset()
    print(f"[STAGE] env built ({args.preset}, num_envs={args.num_envs})", flush=True)

    rd = env.get_robot_data()
    entity = rd._entity
    solver = entity._solver
    dof_ids = rd._actuated_dof_ids
    link_ids = rd._global_link_ids

    # ── getter call counting (inside env.step only) ──────────────────
    counting = {"on": False}
    calls: dict[str, int] = defaultdict(int)

    def counted(obj, name: str):
        fn = getattr(obj, name)

        @functools.wraps(fn)
        def shim(*a, **kw):
            if counting["on"]:
                calls[name] += 1
            return fn(*a, **kw)

        setattr(obj, name, shim)
        return fn  # the unwrapped original, for cache-bypassing fresh reads

    fresh_get_pos = counted(entity, "get_pos")
    fresh_get_quat = counted(entity, "get_quat")
    fresh_get_vel = counted(entity, "get_vel")
    fresh_get_ang = counted(entity, "get_ang")
    fresh_dofs_pos = counted(entity, "get_dofs_position")
    fresh_dofs_vel = counted(entity, "get_dofs_velocity")
    fresh_ctrl_force = counted(entity, "get_dofs_control_force")
    fresh_links_pos = counted(entity, "get_links_pos")
    fresh_links_quat = counted(entity, "get_links_quat")
    fresh_links_vel = counted(entity, "get_links_vel")
    fresh_links_ang = counted(entity, "get_links_ang")

    # cached-property -> cache-bypassing fresh read of the same getter
    pairs = {
        "root_link_pos_w": (lambda: rd.root_link_pos_w, fresh_get_pos),
        "root_link_quat_w": (lambda: rd.root_link_quat_w, fresh_get_quat),
        "root_link_lin_vel_w": (lambda: rd.root_link_lin_vel_w, fresh_get_vel),
        "root_link_ang_vel_w": (lambda: rd.root_link_ang_vel_w, fresh_get_ang),
        "joint_pos": (lambda: rd.joint_pos, lambda: fresh_dofs_pos(dof_ids)),
        "joint_vel": (lambda: rd.joint_vel, lambda: fresh_dofs_vel(dof_ids)),
        "applied_torque": (lambda: rd.applied_torque, lambda: fresh_ctrl_force(dofs_idx_local=dof_ids)),
        "body_pos_w_all": (lambda: rd.body_pos_w_all, fresh_links_pos),
        "body_quat_w_all": (lambda: rd.body_quat_w_all, fresh_links_quat),
        "body_lin_vel_w_all": (lambda: rd.body_lin_vel_w_all, fresh_links_vel),
        "body_ang_vel_w_all": (lambda: rd.body_ang_vel_w_all, fresh_links_ang),
        "root_com_pos_w": (
            lambda: rd.root_com_pos_w,
            lambda: solver.get_links_pos(entity.base_link_idx, ref=gs.link_ref_frame.link_COM)[..., 0, :],
        ),
        "root_com_lin_vel_w": (
            lambda: rd.root_com_lin_vel_w,
            lambda: solver.get_links_vel(entity.base_link_idx, ref=gs.link_ref_frame.link_COM)[..., 0, :],
        ),
        "body_com_pos_w_all": (
            lambda: rd.body_com_pos_w_all,
            lambda: solver.get_links_pos(link_ids, ref=gs.link_ref_frame.link_COM),
        ),
        "body_com_lin_vel_w_all": (
            lambda: rd.body_com_lin_vel_w_all,
            lambda: solver.get_links_vel(link_ids, ref=gs.link_ref_frame.link_COM),
        ),
    }

    # Contact sensors are ring-served from the contact-list backend and
    # verified end-to-end by genesis_contact_list_wiring_diag; this diag
    # only covers the RobotData read cache.
    actions = torch.zeros((args.num_envs, env.num_actions), device=env.device)
    mismatches: dict[str, int] = defaultdict(int)
    step_times: list[float] = []

    for k in range(args.warmup + args.num_steps):
        actions.uniform_(-1.0, 1.0)
        counting["on"] = True
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        env.step(actions)
        torch.cuda.synchronize()
        counting["on"] = False
        if k >= args.warmup:
            step_times.append(time.perf_counter() - t0)

        for name, (cached_fn, fresh_fn) in pairs.items():
            if not torch.equal(cached_fn(), fresh_fn()):
                mismatches[name] += 1

    n = args.warmup + args.num_steps
    has_explicit = env.act_manager.has_explicit_actuators
    dofs_pos_per_step = calls["get_dofs_position"] / n
    pd_fresh_ok = (not has_explicit) or dofs_pos_per_step >= env.decimation
    parity_ok = not mismatches
    ms = sum(step_times) / len(step_times) * 1e3

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"Genesis per-step read-cache verification — preset={args.preset} num_envs={args.num_envs}")
    lines.append("=" * 100)
    lines.append(f"steps: {args.num_steps} (+{args.warmup} warmup)   env.step: {ms:.2f} ms (sync-bracketed)")
    lines.append("")
    verdict = "PASS" if parity_ok else "FAIL"
    lines.append(f"[1] cache freshness/parity ({len(pairs)} RobotData quantities): {verdict}")
    if mismatches:
        for name, cnt in sorted(mismatches.items()):
            lines.append(f"      MISMATCH {name}: {cnt} steps")
    if has_explicit:
        lines.append(
            f"[2] PD substep freshness: get_dofs_position {dofs_pos_per_step:.1f} calls/step "
            f"(decimation={env.decimation}) -> {'PASS' if pd_fresh_ok else 'FAIL'}"
        )
    else:
        lines.append(
            f"[2] PD substep freshness: N/A (implicit actuators only — action path reads no joint state; "
            f"get_dofs_position {dofs_pos_per_step:.1f} calls/step)"
        )
    lines.append("[3] getter calls/step inside env.step (dedup effect; pre-change reference in parentheses):")
    ref = {"get_quat": 11.6, "get_dofs_position": 8.6, "get_dofs_velocity": 6.6, "get_links_pos": 4.3}
    for name in sorted(calls, key=lambda x: -calls[x]):
        extra = f"  (was {ref[name]})" if name in ref else ""
        lines.append(f"      {name:<28} {calls[name] / n:>6.1f}{extra}")
    lines.append("")
    overall = parity_ok and pd_fresh_ok
    lines.append(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    report = "\n".join(lines)
    Path(args.out).write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {Path(args.out).resolve()}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
