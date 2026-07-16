"""Genesis batched contact-timing verification — every robot, bit-exact.

The genesis ``ContactManager.advance`` was changed from "read each group's
contact boolean every substep" (one GPU sync per primary link per substep)
to "read the native per-substep history ring ONCE per control step and
replay its frames through the identical arithmetic".  This diag proves,
per robot preset, that the change is behavior-preserving and measures the
step-time gain:

    1. PARITY (bit-exact): every control step, an independent shadow
       implementation re-reads the same sensor history and accumulates
       air/contact timing with the ORIGINAL per-frame arithmetic; all five
       group buffers (current/last air/contact time, prev_is_contact) must
       be exactly equal (torch.equal) to the manager's — on every step,
       for every group, with mid-rollout resets mirrored.
    2. CONSISTENCY: the newest history frame must equal the live
       ``read_found()`` value (ring orientation / indexing check).
    3. SPEED: mean env.step wall time over the rollout (after warmup);
       compare against the pre-change baseline you recorded with
       ``g1_step_benchmark`` to quantify the win.

Covers every public preset with a genesis variant + contact sensors.
Each preset runs in its own subprocess.

Usage (GPU box):
    python -m rlworld.scripts.diag.genesis_contact_batching_diag
    python -m rlworld.scripts.diag.genesis_contact_batching_diag --presets go2,g1_29dof
    python -m rlworld.scripts.diag.genesis_contact_batching_diag --num-envs 2048
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "rlworld.scripts.diag.genesis_contact_batching_diag"

# preset key -> (module, class)
_PRESETS: dict[str, tuple[str, str]] = {
    "go2": ("rlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("rlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
    "t1_tracking": ("rlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
    "t1_getup": ("rlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
}


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child ───────────────────────────────────────────────────────────


class _ShadowGroup:
    """Reference re-implementation of the ORIGINAL per-frame timing
    arithmetic (mirrors BaseContactManager._apply_contact_frame verbatim)."""

    def __init__(self, num_envs: int, num_tracked: int, device):
        import torch

        z = lambda: torch.zeros(num_envs, num_tracked, device=device)
        self.current_air_time = z()
        self.current_contact_time = z()
        self.last_air_time = z()
        self.last_contact_time = z()
        self.prev_is_contact = torch.zeros(num_envs, num_tracked, device=device, dtype=torch.bool)

    def apply_frame(self, is_contact, dt: float) -> None:
        import torch

        is_landing = ~self.prev_is_contact & is_contact
        is_liftoff = self.prev_is_contact & ~is_contact
        self.last_air_time = torch.where(is_landing, self.current_air_time + dt, self.last_air_time)
        self.last_contact_time = torch.where(is_liftoff, self.current_contact_time + dt, self.last_contact_time)
        self.current_contact_time = torch.where(
            is_contact, self.current_contact_time + dt, torch.zeros_like(self.current_contact_time)
        )
        self.current_air_time = torch.where(
            ~is_contact, self.current_air_time + dt, torch.zeros_like(self.current_air_time)
        )
        self.prev_is_contact = is_contact

    def reset(self, env_ids) -> None:
        self.current_air_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self.prev_is_contact[env_ids] = False


def run_cell(preset: str, num_envs: int, num_steps: int, warmup: int, seed: int) -> dict:
    import importlib

    import torch

    torch.manual_seed(seed)
    module, cls_name = _PRESETS[preset]
    _stage(f"cell start: {preset} (num_envs={num_envs}, steps={num_steps})")

    cfg_cls = getattr(importlib.import_module(module), cls_name)
    cfgs = cfg_cls(sim_type="genesis", num_envs=num_envs, seed=seed).build()

    from rlworld.rl.evals.sim_initializers import get_initializer

    env = get_initializer("Genesis").init_environment(cfgs)
    env.reset()
    _stage("env built + reset")

    manager = env.contact_manager
    groups = manager._groups
    sensors = manager._sensors
    if not groups:
        _stage("no contact groups in this preset — nothing to verify")
        return {"preset": preset, "num_envs": num_envs, "groups": [], "parity_ok": True, "steps": 0}

    decimation = env.decimation
    dt = env.physics_dt
    shadows = {name: _ShadowGroup(num_envs, g.num_tracked, env.device) for name, g in groups.items()}

    actions = torch.zeros((num_envs, env.num_actions), device=env.device)
    mismatches: dict[str, int] = {name: 0 for name in groups}
    newest_frame_mismatch = 0
    step_times: list[float] = []

    for k in range(warmup + num_steps):
        actions.uniform_(-1.0, 1.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        env.step(actions)
        torch.cuda.synchronize()
        if k >= warmup:
            step_times.append(time.perf_counter() - t0)

        # ── shadow replay from a fresh read of the same ring ─────────
        reset_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
        keep = torch.ones(num_envs, dtype=torch.bool, device=env.device)
        keep[reset_ids] = False
        for name, g in groups.items():
            hist = sensors[name].read_found_history()  # (n, H, N) oldest-first
            frames = hist[:, -decimation:, :]
            sh = shadows[name]
            for j in range(frames.shape[1]):
                sh.apply_frame(frames[:, j, :], dt)

            # Consistency: newest ring frame == live read.
            if not torch.equal(frames[:, -1, :], sensors[name].read_found()):
                newest_frame_mismatch += 1

            # Parity on non-reset envs (the manager zeroed reset envs
            # AFTER its replay; mirror that on the shadow below).
            pairs = (
                (sh.current_air_time, g.current_air_time),
                (sh.current_contact_time, g.current_contact_time),
                (sh.last_air_time, g.last_air_time),
                (sh.last_contact_time, g.last_contact_time),
                (sh.prev_is_contact, g._prev_is_contact),
            )
            for mine, theirs in pairs:
                if not torch.equal(mine[keep], theirs[keep]):
                    mismatches[name] += 1
                    break
            if len(reset_ids) > 0:
                sh.reset(reset_ids)

    parity_ok = all(v == 0 for v in mismatches.values()) and newest_frame_mismatch == 0
    ms = sum(step_times) / len(step_times) * 1e3
    _stage(
        f"cell done: parity={'OK' if parity_ok else 'MISMATCH'} "
        f"({num_steps} steps x {len(groups)} groups), env.step {ms:.2f} ms"
    )
    return {
        "preset": preset,
        "num_envs": num_envs,
        "groups": list(groups.keys()),
        "steps": num_steps,
        "mismatch_steps": mismatches,
        "newest_frame_mismatch": newest_frame_mismatch,
        "parity_ok": parity_ok,
        "ms_per_step": ms,
    }


# ── Parent ──────────────────────────────────────────────────────────


def run_parent(args) -> int:
    out_path = Path(args.out).resolve()
    log_dir = out_path.parent / (out_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    presets = [p.strip() for p in args.presets.split(",")] if args.presets else list(_PRESETS)
    results: list[dict] = []
    for preset in presets:
        log_path = log_dir / f"{preset}.log"
        result_path = log_dir / f"{preset}.json"
        if result_path.exists():
            result_path.unlink()
        print(f"[diag] running {preset} ...", flush=True)
        t0 = time.perf_counter()
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    "--cell",
                    preset,
                    "--result-json",
                    str(result_path),
                    "--num-envs",
                    str(args.num_envs),
                    "--num-steps",
                    str(args.num_steps),
                    "--warmup",
                    str(args.warmup),
                    "--seed",
                    str(args.seed),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        wall = time.perf_counter() - t0
        if result_path.exists():
            d = json.loads(result_path.read_text())
        else:
            d = {"preset": preset, "parity_ok": False, "error": f"crashed rc={proc.returncode} (see log)"}
        results.append(d)
        verdict = "PASS" if d.get("parity_ok") else "FAIL"
        print(f"[diag]   -> {verdict} ({wall:.0f}s)", flush=True)

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("Genesis batched contact-timing verification")
    lines.append("=" * 100)
    lines.append(f"num_envs: {args.num_envs}   steps: {args.num_steps} (+{args.warmup} warmup)   seed: {args.seed}")
    lines.append(f"cell logs: {log_dir}")
    lines.append("")
    lines.append(f"{'preset':<15}{'verdict':<10}{'groups':<38}{'env.step ms':<14}detail")
    lines.append("-" * 100)
    for d in results:
        verdict = "PASS" if d.get("parity_ok") else "FAIL"
        groups = ",".join(d.get("groups", [])) or "-"
        ms = f"{d['ms_per_step']:.2f}" if "ms_per_step" in d else "—"
        detail = (
            d.get("error", "")
            or f"mismatch_steps={d.get('mismatch_steps', {})} newest_frame_mismatch={d.get('newest_frame_mismatch', '?')}"
        )
        lines.append(f"{d['preset']:<15}{verdict:<10}{groups:<38}{ms:<14}{detail}")
    lines.append("")
    lines.append("[Reading] PASS = manager buffers bit-identical to the reference per-frame arithmetic")
    lines.append("          on every step (resets mirrored) AND ring orientation verified against live reads.")
    lines.append("          For the headline speedup, re-run: g1_step_benchmark --cells genesis:jaxrlworld:4096")
    report = "\n".join(lines)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {out_path}")
    return 0 if all(d.get("parity_ok") for d in results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, help="internal: run one preset")
    ap.add_argument("--result-json", default=None, help="internal: child result path")
    ap.add_argument("--presets", default=None, help="comma-separated subset of presets")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--num-steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="genesis_contact_batching_diag.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(args.cell, args.num_envs, args.num_steps, args.warmup, args.seed)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
