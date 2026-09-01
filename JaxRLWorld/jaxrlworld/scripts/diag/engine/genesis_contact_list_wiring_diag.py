"""Genesis contact-list backend wiring verification — every preset, end to end.

The Genesis contact backend computes every group from one shared
``collider.get_contacts`` read per substep (see
``managers/genesis/contact_sensor.py``). Source-level VALUE parity of
that reconstruction against Genesis's native sensors is proven by
``genesis_contact_list_parity_diag`` (standalone scene, both paths side
by side). This diag covers what remains: the PRODUCTION WIRING inside
the wrapped env, for every genesis preset, on a random-action rollout
with periodic side-lying drops (so body-vs-ground and self-collision
groups all actually fire):

    1. FRAME PARITY: after every ``ContactManager.advance`` (i.e. every
       substep), each group's ``read_found``/``read_force`` is compared
       against an INDEPENDENT recomputation from the collider state
       (this file's own masking/rotation math, the group's primary and
       counterpart link sets re-derived here). Catches generation/cache
       wiring bugs, ring indexing bugs, and stale-row leaks.
    2. TIMING PARITY: the independently recomputed frames are replayed
       through a shadow copy of the air/contact-time arithmetic
       (mirroring ``BaseContactManager._apply_contact_frame``); all five
       timing buffers must match the manager's bit-exactly every control
       step (mid-rollout resets mirrored).
    3. HISTORY PARITY: ``compute_history()``'s newest ``decimation``
       frames must match the frames captured this control step.
    4. COVERAGE: a group with zero contact events over the rollout
       counts as FAIL (unverified), not PASS.

Each preset runs in its own subprocess.

Usage (GPU box):
    python -m jaxrlworld.scripts.diag.engine.genesis_contact_list_wiring_diag
    python -m jaxrlworld.scripts.diag.engine.genesis_contact_list_wiring_diag --presets go2,t1_getup
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "jaxrlworld.scripts.diag.engine.genesis_contact_list_wiring_diag"

_PRESETS: dict[str, tuple[str, str]] = {
    "go2": ("jaxrlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "g1_29dof": ("jaxrlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("jaxrlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
    "t1_tracking": ("jaxrlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
    "t1_getup": ("jaxrlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
}

_ATOL, _RTOL = 1e-3, 1e-4


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child ───────────────────────────────────────────────────────────


class _ShadowGroup:
    """Reference re-implementation of the per-frame timing arithmetic
    (mirrors BaseContactManager._apply_contact_frame verbatim)."""

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
    from genesis.utils.geom import inv_transform_by_quat
    from genesis.utils.misc import qd_to_torch

    torch.manual_seed(seed)
    module, cls_name = _PRESETS[preset]
    _stage(f"cell start: {preset} (num_envs={num_envs}, steps={num_steps})")

    cfg_cls = getattr(importlib.import_module(module), cls_name)
    cfgs = cfg_cls(sim_type="genesis", num_envs=num_envs, seed=seed).build()

    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    env = get_initializer("Genesis").init_environment(cfgs)
    env.reset()
    _stage("env built + reset")

    manager = env.contact_manager
    sensors = manager._sensors
    groups = manager._groups
    if not sensors:
        _stage("no contact sensors in this preset — nothing to verify")
        return {"preset": preset, "groups": {}, "ok": True, "note": "no contact sensors"}

    solver = env.scene_manager.scene.sim.rigid_solver
    dev = env.device
    decimation = env.decimation
    dt = env.physics_dt

    # Independent group definitions, re-derived from the live sensors'
    # resolved link sets (primary/counterpart in global link space).
    meta: dict[str, dict] = {}
    for name, sensor in sensors.items():
        meta[name] = {
            "sensor": sensor,
            "entity": sensor._entity,
            "primary_g": sensor._primary_links.clone(),
            "primary_local": sensor._primary_local.clone(),
            "counterpart": sensor._counterpart_links.clone(),
            "counterpart_is_all": sensor._counterpart_is_all,
        }
        _stage(
            f"group {name}: {int(meta[name]['primary_g'].numel())} primary links, "
            f"{'ALL' if meta[name]['counterpart_is_all'] else int(meta[name]['counterpart'].numel())} counterparts"
        )

    def recompute(m: dict):
        """This file's own found/force math from the collider state."""
        cd = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_a, link_b, force = cd["link_a"], cd["link_b"], cd["force"]
        n_live = qd_to_torch(solver.collider._collider_state.n_contacts, copy=False)
        row_valid = torch.arange(link_a.shape[1], device=link_a.device)[None, :] < n_live[:, None]
        primary, counterpart = m["primary_g"], m["counterpart"]
        on_p_a = link_a.unsqueeze(-1) == primary
        on_p_b = link_b.unsqueeze(-1) == primary
        if m["counterpart_is_all"]:
            a_ok = b_ok = row_valid
        else:
            a_ok = row_valid & (link_a.unsqueeze(-1) == counterpart).any(-1)
            b_ok = row_valid & (link_b.unsqueeze(-1) == counterpart).any(-1)
        pmask_a = on_p_a & b_ok.unsqueeze(-1)
        pmask_b = on_p_b & a_ok.unsqueeze(-1)
        found = (pmask_a | pmask_b).any(1)
        f_world = torch.einsum("ncp,nci->npi", pmask_b.float() - pmask_a.float(), force)
        quats = m["entity"].get_links_quat()[:, m["primary_local"]]
        return found, inv_transform_by_quat(f_world, quats)

    stats = {
        name: {
            "frames": 0,
            "found_mismatch": 0,
            "force_bad": 0,
            "force_max_diff": 0.0,
            "timing_mismatch_steps": 0,
            "history_bad_steps": 0,
            "found_events": 0,
        }
        for name in sensors
    }
    shadows = {name: _ShadowGroup(num_envs, g.num_tracked, dev) for name, g in groups.items()}
    step_frames: dict[str, list] = {name: [] for name in sensors}

    # Diag-only shim: run the production advance, then compare each
    # group's served frame against the independent recomputation and
    # feed the recomputed frame to the shadow timing arithmetic.
    orig_advance = manager.advance
    gate = {"on": False}

    def advance_shim(dt):
        orig_advance(dt=dt)
        if not gate["on"]:
            return
        for name, m in meta.items():
            found_ref, force_ref = recompute(m)
            sensor = m["sensor"]
            s = stats[name]
            s["frames"] += 1
            s["found_events"] += int(found_ref.sum())
            if not torch.equal(sensor.read_found(), found_ref):
                s["found_mismatch"] += int((sensor.read_found() != found_ref).sum())
            diff = (sensor.read_force() - force_ref).abs().max()
            s["force_max_diff"] = max(s["force_max_diff"], float(diff))
            if not torch.allclose(sensor.read_force(), force_ref, atol=_ATOL, rtol=_RTOL):
                s["force_bad"] += 1
            shadows[name].apply_frame(found_ref, dt)
            step_frames[name].append(force_ref)

    manager.advance = advance_shim

    robot = env.scene_manager.robot
    side_quat = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0], device=dev).expand(num_envs, 4).contiguous()
    actions = torch.zeros((num_envs, env.num_actions), device=dev)

    for k in range(warmup + num_steps):
        actions.uniform_(-1.0, 1.0)
        gate["on"] = k >= warmup
        if k == warmup:
            # Seed the shadows from the manager's warmup-accumulated
            # buffers. Presets whose envs rarely reset (t1_getup: lying
            # down IS the task, so only the episode timeout resets)
            # would otherwise carry the warmup offset forever.
            for name, g in groups.items():
                sh = shadows[name]
                sh.current_air_time = g.current_air_time.clone()
                sh.current_contact_time = g.current_contact_time.clone()
                sh.last_air_time = g.last_air_time.clone()
                sh.last_contact_time = g.last_contact_time.clone()
                sh.prev_is_contact = g._prev_is_contact.clone()
        for frames in step_frames.values():
            frames.clear()
        if k % 10 == 0:
            # Side-lying at ground height so body-vs-ground groups fire.
            pos = robot.get_pos().clone()
            pos[:, 2] = 0.15
            robot.set_pos(pos, zero_velocity=True)
            robot.set_quat(side_quat, zero_velocity=True)
        env.step(actions)
        if k < warmup:
            continue

        reset_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
        keep = torch.ones(num_envs, dtype=torch.bool, device=dev)
        keep[reset_ids] = False
        for name, g in groups.items():
            sh, s = shadows[name], stats[name]
            pairs = (
                (sh.current_air_time, g.current_air_time),
                (sh.current_contact_time, g.current_contact_time),
                (sh.last_air_time, g.last_air_time),
                (sh.last_contact_time, g.last_contact_time),
                (sh.prev_is_contact, g._prev_is_contact),
            )
            for mine, theirs in pairs:
                if not torch.equal(mine[keep], theirs[keep]):
                    s["timing_mismatch_steps"] += 1
                    break
            if len(reset_ids) > 0:
                sh.reset(reset_ids)

            # compute_history(): newest-first (B, N, H, 3); its first
            # ``decimation`` frames must be this step's captured frames.
            hist = sensors[name].compute_history()[:, :, :decimation, :]
            ref = torch.stack(list(reversed(step_frames[name])), dim=2)
            if not torch.allclose(hist, ref, atol=_ATOL, rtol=_RTOL):
                s["history_bad_steps"] += 1

    manager.advance = orig_advance

    ok = True
    groups_out: dict[str, dict] = {}
    for name, s in stats.items():
        coverage = s["found_events"] / max(s["frames"], 1)
        group_ok = (
            s["found_mismatch"] == 0
            and s["force_bad"] == 0
            and s["timing_mismatch_steps"] == 0
            and s["history_bad_steps"] == 0
            and coverage > 0
        )
        ok &= group_ok
        groups_out[name] = {**s, "coverage": coverage, "ok": group_ok}
        _stage(
            f"group {name}: {'PASS' if group_ok else 'FAIL'} — found_mm={s['found_mismatch']} "
            f"force_bad={s['force_bad']} max|dF|={s['force_max_diff']:.4g} "
            f"timing_mm={s['timing_mismatch_steps']} hist_bad={s['history_bad_steps']} "
            f"events/frame={coverage:.1f}"
        )
    return {"preset": preset, "groups": groups_out, "ok": ok}


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
            d = {"preset": preset, "ok": False, "error": f"crashed rc={proc.returncode} (see log)"}
        results.append(d)
        print(f"[diag]   -> {'PASS' if d.get('ok') else 'FAIL'} ({wall:.0f}s)", flush=True)

    lines: list[str] = []
    lines.append("=" * 110)
    lines.append("Genesis contact-list backend wiring verification")
    lines.append("=" * 110)
    lines.append(
        f"num_envs: {args.num_envs}   steps: {args.num_steps} (+{args.warmup} warmup, x substeps)   "
        f"seed: {args.seed}   force tol: atol={_ATOL} rtol={_RTOL}"
    )
    lines.append(f"cell logs: {log_dir}")
    lines.append("")
    lines.append(f"{'preset':<15}{'verdict':<10}detail")
    lines.append("-" * 110)
    for d in results:
        verdict = "PASS" if d.get("ok") else "FAIL"
        if "error" in d:
            detail = d["error"]
        elif not d.get("groups"):
            detail = d.get("note", "-")
        else:
            parts = []
            for g, s in d["groups"].items():
                parts.append(
                    f"{g}[{'ok' if s['ok'] else 'BAD'} mm={s['found_mismatch']} t={s['timing_mismatch_steps']} "
                    f"h={s['history_bad_steps']} dF={s['force_max_diff']:.3g} ev={s['coverage']:.1f}]"
                )
            detail = " ".join(parts)
        lines.append(f"{d['preset']:<15}{verdict:<10}{detail}")
    lines.append("")
    lines.append("[Reading] PASS = every group: served frames match an independent collider recomputation")
    lines.append("          (found bit-exact), timing buffers match the shadow arithmetic bit-exactly,")
    lines.append("          force history matches the captured frames, and contact events actually occurred.")
    report = "\n".join(lines)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {out_path}")
    return 0 if all(d.get("ok") for d in results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, help="internal: run one preset")
    ap.add_argument("--result-json", default=None, help="internal: child result path")
    ap.add_argument("--presets", default=None, help="comma-separated subset of presets")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="genesis_contact_list_wiring_diag.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(args.cell, args.num_envs, args.num_steps, args.warmup, args.seed)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
