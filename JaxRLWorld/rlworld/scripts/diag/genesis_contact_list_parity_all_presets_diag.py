"""Contact-list vs native-sensor parity on EVERY genesis preset's real env.

Extends ``genesis_contact_list_parity_diag`` (standalone g1 scene) to the
migration gate the contact-list switch actually needs: for every public
preset with a genesis variant, build the REAL wrapped env — so each
preset's own ``ContactSensorCfg`` groups (primary patterns, excludes,
secondary entity/self/terrain filters) are exercised exactly as
configured — and compare, at EVERY substep of a random-action rollout:

    native ``gs.sensors.Contact``  vs  "unfiltered pair in collider list"
        (bit-exact required)
    native ``gs.sensors.ContactForce`` vs signed pair-force sum rotated
        into the link local frame (tolerance = float32 sum-order noise)

The list side is reconstructed from the LIVE sensor objects' own resolved
state (primary local link ids, global blacklist), so any preset-specific
resolution path (exclude patterns, terrain sentinel, self filter) is
covered by construction.  Per-substep access is obtained by shimming the
bound ``scene_manager.step`` (diag-only; framework untouched).  Native
values are read via ``read_ground_truth()`` directly, bypassing the
per-step read cache, which would otherwise serve pre-substep frames here.

Each preset runs in its own subprocess.  A group with ZERO contact events
over the rollout counts as FAIL (unverified), not PASS.

Usage (GPU box):
    python -m rlworld.scripts.diag.genesis_contact_list_parity_all_presets_diag
    python -m rlworld.scripts.diag.genesis_contact_list_parity_all_presets_diag --presets go2,t1_getup
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "rlworld.scripts.diag.genesis_contact_list_parity_all_presets_diag"

_PRESETS: dict[str, tuple[str, str]] = {
    "go2": ("rlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "g1_tracking": ("rlworld.rl.configs.presets.g1_tracking.base", "G1TrackingConfig"),
    "t1_tracking": ("rlworld.rl.configs.presets.t1_tracking.base", "T1TrackingConfig"),
    "t1_getup": ("rlworld.rl.configs.presets.t1_getup.base", "T1GetupConfig"),
}

_ATOL, _RTOL = 1e-2, 1e-3


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child ───────────────────────────────────────────────────────────


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

    from rlworld.rl.evals.sim_initializers import get_initializer

    env = get_initializer("Genesis").init_environment(cfgs)
    env.reset()
    _stage("env built + reset")

    sensors = env.contact_manager._sensors
    if not sensors:
        _stage("no contact sensors in this preset — nothing to verify")
        return {"preset": preset, "groups": {}, "ok": True, "note": "no contact sensors"}

    solver = env.scene_manager.scene.rigid_solver
    n_links_global = solver.n_links
    dev = env.device

    # Per-group static data, derived from each LIVE sensor's resolved state.
    group_meta: dict[str, dict] = {}
    for name, sensor in sensors.items():
        entity = sensor._entity
        link_start = entity.links[0].idx
        primary_local = torch.tensor(sensor._link_ids_local, device=dev)
        primary_global = primary_local + link_start
        blacklist = set(sensor._filter_link_idx)
        counterpart = torch.tensor(
            sorted(set(range(n_links_global)) - blacklist) if blacklist else list(range(n_links_global)),
            device=dev,
        )
        has_history = sensor.cfg.history_length > 0
        group_meta[name] = {
            "sensor": sensor,
            "entity": entity,
            "primary_local": primary_local,
            "primary_global": primary_global,
            "counterpart": counterpart,
            "has_history": has_history,
        }
        _stage(
            f"group {name}: {len(sensor._link_ids_local)} primary links, "
            f"{counterpart.numel()} counterpart links (blacklist {len(blacklist)})"
        )

    stats = {
        name: {"frames": 0, "found_mismatch": 0, "found_events": 0, "force_max_diff": 0.0, "force_bad": 0}
        for name in sensors
    }

    def native_newest(sensor, has_history: bool):
        found_cols, force_cols = [], []
        for cs, fs in zip(sensor._contact_sensors, sensor._force_sensors):
            c = cs.read_ground_truth()
            f = fs.read_ground_truth()
            if has_history:  # (B, H, D) newest-first
                c, f = c[:, 0, :], f[:, 0, :]
            found_cols.append(c[..., 0] != 0)
            force_cols.append(f)
        return torch.stack(found_cols, dim=1), torch.stack(force_cols, dim=1)

    def compare_substep() -> None:
        cd = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_a, link_b, force = cd["link_a"], cd["link_b"], cd["force"]
        n_live = qd_to_torch(solver.collider._collider_state.n_contacts, copy=False)
        row_valid = torch.arange(link_a.shape[1], device=link_a.device)[None, :] < n_live[:, None]
        for name, meta in group_meta.items():
            primary, counterpart = meta["primary_global"], meta["counterpart"]
            a_is_p = (link_a.unsqueeze(-1) == primary).any(-1)
            b_is_p = (link_b.unsqueeze(-1) == primary).any(-1)
            a_is_c = (link_a.unsqueeze(-1) == counterpart).any(-1)
            b_is_c = (link_b.unsqueeze(-1) == counterpart).any(-1)
            pair = row_valid & ((a_is_p & b_is_c) | (b_is_p & a_is_c))
            pmask_a = (link_a.unsqueeze(-1) == primary) & (pair & b_is_c).unsqueeze(-1)
            pmask_b = (link_b.unsqueeze(-1) == primary) & (pair & a_is_c).unsqueeze(-1)
            li_found = (pmask_a | pmask_b).any(1)
            f_world = torch.einsum("ncp,nci->npi", pmask_b.float() - pmask_a.float(), force)
            quats = meta["entity"].get_links_quat()[:, meta["primary_local"]]
            li_force = inv_transform_by_quat(f_world, quats)

            nat_found, nat_force = native_newest(meta["sensor"], meta["has_history"])
            s = stats[name]
            s["frames"] += 1
            s["found_events"] += int(nat_found.sum())
            if not torch.equal(nat_found, li_found):
                s["found_mismatch"] += int((nat_found != li_found).sum())
            s["force_max_diff"] = max(s["force_max_diff"], float((nat_force - li_force).abs().max()))
            if not torch.allclose(nat_force, li_force, atol=_ATOL, rtol=_RTOL):
                s["force_bad"] += 1

    # Diag-only shim on the bound scene_manager.step: compare after every substep.
    orig_step = env.scene_manager.step
    gate = {"on": False}

    def stepped():
        orig_step()
        if gate["on"]:
            compare_substep()

    env.scene_manager.step = stepped

    # Periodic topple: drop every robot lying on its side so body-vs-ground
    # groups (e.g. go2 body_ground_contact) are guaranteed contact events —
    # random joint actions alone can leave a group untouched for the whole
    # rollout, which this diag counts as FAIL (unverified).
    robot = env.scene_manager.robot
    side_quat = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0], device=dev).expand(num_envs, 4).contiguous()

    actions = torch.zeros((num_envs, env.num_actions), device=dev)
    for k in range(warmup + num_steps):
        actions.uniform_(-1.0, 1.0)
        gate["on"] = k >= warmup
        if k % 10 == 0:
            # Side-lying at (near-)ground height so body-ground contact
            # happens within the very next substeps — a higher drop never
            # reaches the ground before termination resets the env.
            pos = robot.get_pos().clone()
            pos[:, 2] = 0.15
            robot.set_pos(pos, zero_velocity=True)
            robot.set_quat(side_quat, zero_velocity=True)
        env.step(actions)
    env.scene_manager.step = orig_step

    ok = True
    groups_out: dict[str, dict] = {}
    for name, s in stats.items():
        coverage = s["found_events"] / max(s["frames"], 1)
        group_ok = s["found_mismatch"] == 0 and s["force_bad"] == 0 and coverage > 0
        ok &= group_ok
        groups_out[name] = {**{k: v for k, v in s.items()}, "coverage": coverage, "ok": group_ok}
        _stage(
            f"group {name}: {'PASS' if group_ok else 'FAIL'} — found_mismatch={s['found_mismatch']} "
            f"force_bad={s['force_bad']}/{s['frames']} max|dF|={s['force_max_diff']:.4g} "
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
    lines.append("Genesis contact-list parity — all presets, real wrapped envs, per-substep")
    lines.append("=" * 110)
    lines.append(
        f"num_envs: {args.num_envs}   steps: {args.num_steps} (+{args.warmup} warmup, x4 substeps)   "
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
                    f"{g}[{'ok' if s['ok'] else 'BAD'} mm={s['found_mismatch']} "
                    f"dF={s['force_max_diff']:.3g} ev={s['coverage']:.1f}]"
                )
            detail = " ".join(parts)
        lines.append(f"{d['preset']:<15}{verdict:<10}{detail}")
    lines.append("")
    lines.append("[Reading] PASS = every group: found bit-exact, force within float32 sum-order tolerance,")
    lines.append("          and the rollout actually produced contact events for that group.")
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
    ap.add_argument("--out", default="genesis_contact_list_parity_all_presets.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(args.cell, args.num_envs, args.num_steps, args.warmup, args.seed)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
