"""Does the compiled Genesis contact substep match the eager one?

``GenesisContactBatch`` runs the per-substep contact capture, history
rings and contact-timing update as one function of tensors, through
``torch.compile`` when ``EnvConfig.compile_contact_kernels`` is set.
This drives a live env and, on every substep, runs the eager
``_substep_impl`` on cloned rings / timing buffers next to the compiled
one on the production buffers, from the same engine reads:

- ``found`` rings and all five timing buffers must be bit-identical
  (boolean logic, ``where`` and ``+ dt`` — nothing a fused kernel can
  reassociate).
- the link-frame force rings are compared to 1e-5 of the frame scale:
  the fused einsum / quaternion rotation may contract multiply-adds and
  reorder the reduction.

Usage (GPU box, from the SimForge root):
    jaxpy -m jaxrlworld.scripts.diag.gates.check_genesis_contact_compile
    jaxpy -m jaxrlworld.scripts.diag.gates.check_genesis_contact_compile --preset \\
        jaxrlworld.rl.configs.presets.go2.genesis.gait_conditioned:Go2GaitConditionedGenesisConfig
"""

from __future__ import annotations

import argparse
import importlib
from types import SimpleNamespace

import torch

from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager
from jaxrlworld.rl.envs.managers.genesis.contact_sensor import GenesisContactBatch

_DEFAULT_PRESET = "jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe:K1G1RecipeConfig"


def _build_env(preset: str, num_envs: int):
    from jaxrlworld.rl.runners import BaseRunner

    mod_path, cls_name = preset.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type="genesis", num_envs=num_envs).build()
    if not cfgs.env.compile_contact_kernels:
        raise RuntimeError("compile_contact_kernels is off for this preset; nothing to compare.")
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=_DEFAULT_PRESET)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = _build_env(args.preset, args.num_envs)
    cm = env.contact_manager
    if not cm._sensors:
        raise RuntimeError(f"Preset {args.preset!r} has no genesis contact sensors to compare.")
    batch = cm._get_batch()
    if batch._substep is batch._substep_impl:
        raise RuntimeError("The batch is not compiled; compile_contact_kernels did not take effect.")

    fields = BaseContactManager._TIMING_FIELDS
    stats = {"substeps": 0, "found_mismatch": 0, "timing_mismatch": 0, "force_max_rel": 0.0, "force_fail": 0}

    def checked_capture_and_advance(timing, dt: float) -> None:
        inputs = batch._gather_inputs()
        link_a, link_b, force, n_live, quats, found_hists, force_hists = inputs
        ref_timing = SimpleNamespace(**{f: getattr(timing, f).clone() for f in fields})
        ref_found, ref_force = GenesisContactBatch._substep_impl(
            batch,
            link_a,
            link_b,
            force,
            n_live,
            quats,
            [h.clone() for h in found_hists],
            [h.clone() for h in force_hists],
            ref_timing,
            dt,
        )
        new_found, new_force = batch._substep(*inputs, timing, dt)
        batch._store_rings(new_found, new_force)

        stats["substeps"] += 1
        for a, b in zip(ref_found, new_found):
            if not torch.equal(a, b):
                stats["found_mismatch"] += 1
        for f in fields:
            if not torch.equal(getattr(ref_timing, f), getattr(timing, f)):
                stats["timing_mismatch"] += 1
        for a, b in zip(ref_force, new_force):
            scale = a[:, 0].abs().max().clamp_min(1e-6)
            rel = ((a[:, 0] - b[:, 0]).abs().max() / scale).item()
            stats["force_max_rel"] = max(stats["force_max_rel"], rel)
            if rel > 1e-5:
                stats["force_fail"] += 1

    batch.capture_and_advance = checked_capture_and_advance

    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    widths: set[int] = set()
    for step in range(args.steps):
        env.step(0.1 * torch.randn_like(action))
        widths.add(int(batch._gather_inputs()[0].shape[1]))
        if step == 0:
            # The contact-list width varies per substep and is declared
            # dynamic; after the first compile nothing may recompile.
            torch._dynamo.config.error_on_recompile = True
        if step % 10 == 0:
            print(
                f"[step {step:3d}] substeps={stats['substeps']} found_mismatch={stats['found_mismatch']} "
                f"timing_mismatch={stats['timing_mismatch']} force_max_rel={stats['force_max_rel']:.2e}"
            )

    found_bad, timing_bad, force_bad = stats["found_mismatch"], stats["timing_mismatch"], stats["force_fail"]
    failed = found_bad or timing_bad or force_bad
    print()
    print(f"substeps compared: {stats['substeps']}  groups: {len(cm._sensors)}")
    print(f"contact-list widths seen: {sorted(widths)}  (no recompile raised)")
    print(f"found rings   : {'bit-identical' if not found_bad else f'{found_bad} MISMATCHES'}")
    print(f"timing buffers: {'bit-identical' if not timing_bad else f'{timing_bad} MISMATCHES'}")
    print(
        f"force rings   : max rel diff {stats['force_max_rel']:.2e} ({'ok' if not force_bad else f'{force_bad} over 1e-5'})"
    )
    if failed:
        print("\nFAIL")
        return 1
    print("\nPASS — compiled contact substep matches eager")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
