"""Equivalence check: fused Genesis contact capture vs the per-group path.

``GenesisContactBatch`` computes every contact group's ``found``/``force``
frame in one fused pass; ``GenesisContactSensor.capture_substep`` is the
original per-group computation, kept intact. Both read the same
generation-cached contact list, so on a live env we can compare them
directly per step:

1. Step the env normally (the production path captures via the batch).
2. For each sensor: save its ring references, run the per-group
   ``capture_substep`` on the SAME cache generation, compare its newest
   frame against the batch's, then restore the rings.

``found`` must be BIT-identical (same boolean formula per column).
``force`` runs through one larger einsum, so only the in-kernel
reduction scheduling may differ — compared to 1e-5 of the frame scale.

Usage:
    jaxpy -m jaxrlworld.scripts.diag.gates.check_genesis_contact_batching
"""

from __future__ import annotations

import argparse
import importlib

import torch


def _build_env(preset: str, num_envs: int):
    from jaxrlworld.rl.runners import BaseRunner

    if ":" in preset:
        mod_path, cls_name = preset.split(":", 1)
    else:
        mod_path, cls_name = (
            "jaxrlworld.rl.configs.presets.go2.genesis.gait_conditioned",
            "Go2GaitConditionedGenesisConfig",
        )
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type="genesis", num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="go2_gait")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = _build_env(args.preset, args.num_envs)
    sensors = env.contact_manager._sensors
    if not sensors:
        raise RuntimeError(f"Preset {args.preset!r} has no genesis contact sensors to compare.")

    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    failures = 0
    for step in range(args.steps):
        env.step(0.1 * torch.randn_like(action))

        for name, sensor in sensors.items():
            batch_found = sensor.read_found().clone()
            batch_force = sensor.read_force().clone() if sensor._track_force else None

            # Re-run the per-group capture on the same generation (the
            # shared list read is cached), then restore the rings so the
            # env continues from the production state.
            saved_found, saved_force = sensor._found_hist, sensor._force_hist
            sensor.capture_substep()
            ref_found = sensor.read_found().clone()
            ref_force = sensor.read_force().clone() if sensor._track_force else None
            sensor._found_hist, sensor._force_hist = saved_found, saved_force

            if not torch.equal(batch_found, ref_found):
                failures += 1
                print(f"[step {step:3d}] {name}: found differs")
            if batch_force is not None:
                scale = ref_force.abs().max().clamp_min(1e-6)
                diff = (batch_force - ref_force).abs().max() / scale
                if diff.item() > 1e-5:
                    failures += 1
                    print(f"[step {step:3d}] {name}: force rel diff {diff.item():.3e}")

        if failures == 0 and step % 10 == 0:
            print(f"[step {step:3d}] all groups match the per-group reference")

    if failures:
        print(f"\nFAIL — {failures} mismatches over {args.steps} steps")
        return 1
    print(f"\nPASS — fused capture matches the per-group path, {len(sensors)} groups x {args.steps} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
