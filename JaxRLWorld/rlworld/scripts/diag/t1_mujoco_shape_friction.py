"""Dump per-env per-geom friction (slide/torsional/rolling) in mjlab.

mjlab's startup-mode ``foot_friction_spin`` / ``foot_friction_roll``
DR terms use ``mujoco.randomize_friction`` which rewrites
MuJoCo's ``geom_friction[:, geom_ids, axis]`` per world. This script
builds the env (so startup DR fires), then prints the per-env values
for the foot collision geoms (from
``T1Config.foot_geom_names_mjlab``) so we can verify randomization
actually fired and that non-foot geoms are untouched.
"""
from __future__ import annotations

import mujoco

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner


def _geom_id_by_name(mj, name: str) -> int:
    """Accepts either bare name or ``<entity>/name`` form."""
    for candidate in (name, f"robot/{name}"):
        gid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, candidate)
        if gid >= 0:
            return gid
    return -1


def main() -> None:
    cfgs = T1GetupConfig(sim_type="mujoco", num_envs=4).build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    sm = env.scene_manager
    mj = sm.mj_model
    m = sm.model  # mujoco-warp Model (per-env geom_friction lives here)

    r = cfgs.runner.robot if hasattr(cfgs, "runner") and hasattr(cfgs.runner, "robot") else None
    foot_names = r.foot_geom_names_mjlab if r is not None else (
        "left_foot1_collision", "left_foot2_collision", "left_foot3_collision", "left_foot4_collision",
        "right_foot1_collision", "right_foot2_collision", "right_foot3_collision", "right_foot4_collision",
    )

    foot_ids = []
    for n in foot_names:
        gid = _geom_id_by_name(mj, n)
        if gid < 0:
            print(f"  WARN: geom {n!r} not found")
            continue
        foot_ids.append((n, gid))

    import numpy as np
    import torch
    import warp as wp
    # mujoco-warp stores per-env geom_friction. Shape: (nworld, ngeom, 3).
    # Newer warp versions' to_torch hit an is_cpu attr error on torch.device,
    # so go numpy → torch directly.
    arr = m.geom_friction
    if isinstance(arr, torch.Tensor):
        gf = arr
    elif isinstance(arr, np.ndarray):
        gf = torch.from_numpy(arr)
    else:
        # wp.array — use .numpy() round-trip
        gf = torch.from_numpy(arr.numpy())
    print(f"\nmujoco-warp geom_friction shape: {tuple(gf.shape)}  (nworld, ngeom, 3)\n")

    print("=== Foot geoms — friction[:, geom, :] per env ===")
    print(f"{'Geom':<30} {'GeomID':<7} env{'slide':>9} {'torsional':>12} {'rolling':>12}")
    for name, gid in foot_ids:
        for env_i in range(gf.shape[0]):
            tri = gf[env_i, gid].cpu().tolist()
            tag = name if env_i == 0 else ""
            gid_s = str(gid) if env_i == 0 else ""
            print(f"{tag:<30} {gid_s:<7} env{env_i}  "
                  f"{tri[0]:>9.5f} {tri[1]:>12.6f} {tri[2]:>12.6f}")
        print()

    # Sanity non-foot geom
    all_gids = {gid for _, gid in foot_ids}
    non_foot = next((g for g in range(gf.shape[1]) if g not in all_gids), None)
    if non_foot is not None:
        nf_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, non_foot) or "?"
        print(f"=== Non-foot geom {non_foot} ({nf_name!r}) — should be untouched ===")
        for env_i in range(gf.shape[0]):
            tri = gf[env_i, non_foot].cpu().tolist()
            print(f"  env{env_i}: slide={tri[0]:.5f} torsional={tri[1]:.6f} rolling={tri[2]:.6f}")


if __name__ == "__main__":
    main()
