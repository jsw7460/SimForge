"""Dump per-world per-shape geom friction (slide/torsional/rolling).

The T1 getup startup-mode DR terms ``foot_friction_spin`` /
``foot_friction_roll`` randomize Newton's ``shape_material_mu`` /
``shape_material_mu_torsional`` / ``shape_material_mu_rolling`` for
the foot collision shapes. This script builds the env (so startup DR
fires), then prints the per-env values for the foot shapes so we can
verify:

  - each env actually sees a different value (log_uniform over range),
  - non-foot shapes were NOT touched,
  - the expected ranges match the config.
"""
from __future__ import annotations

import warp as wp

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.envs.utils.newton.body_cache import get_cache
from rlworld.rl.envs.utils.newton.label import leaf_name
from rlworld.rl.runners import BaseRunner


def _attr(view, m, name: str):
    return wp.to_torch(view.get_attribute(name, m))


def main() -> None:
    cfgs = T1GetupConfig(sim_type="newton", num_envs=4).build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    sm = env.scene_manager
    m = sm.model
    view = sm.robot_view

    # Resolve foot shape indices via the same path as the DR term
    cache = get_cache(env)
    body_idx = cache.get_body_indices(env.env_config.robot.foot_body_pattern_newton) \
        if hasattr(env, "env_config") else cache.get_body_indices(r"(left|right)_foot_link")
    foot_shape_idx = []
    for bi in (body_idx.tolist() if hasattr(body_idx, "tolist") else body_idx):
        foot_shape_idx.extend(int(si) for si in m.body_shapes[int(bi)])

    slide = _attr(view, m, "shape_material_mu")          # (W, 1, shapes)
    tor   = _attr(view, m, "shape_material_mu_torsional")
    rol   = _attr(view, m, "shape_material_mu_rolling")

    # ArticulationView attrs come out as (num_envs, num_articulations, per_arti)
    # — squeeze the middle dim so indexing is [env, shape].
    if slide.ndim == 3:
        slide = slide[:, 0, :]
        tor   = tor[:, 0, :]
        rol   = rol[:, 0, :]

    shapes_per_arti = slide.shape[1]
    print(f"\nShape tensor shape after squeeze: {tuple(slide.shape)}  (num_envs, shapes_per_arti={shapes_per_arti})")
    print(f"Foot shape indices from body_shapes: {foot_shape_idx}\n")

    # Keep only foot indices that fall within the view's shape space.
    valid_foot = [s for s in foot_shape_idx if s < shapes_per_arti]
    if len(valid_foot) < len(foot_shape_idx):
        print(f"  WARN: {len(foot_shape_idx) - len(valid_foot)} foot shape indices out of range "
              f"(model body_shapes may index global space incl. ground plane)")

    print("=== Foot shapes — slide μ (axis 0, DR range (0.8, 1.5) uniform) ===")
    for si in valid_foot:
        print(f"  shape {si:<4}: {slide[:, si].cpu().tolist()}")

    print("\n=== Foot shapes — torsional μ (axis 1, DR range (1e-4, 2e-2) log_uniform) ===")
    for si in valid_foot:
        print(f"  shape {si:<4}: {tor[:, si].cpu().tolist()}")

    print("\n=== Foot shapes — rolling μ (axis 2, DR range (1e-5, 5e-3) log_uniform) ===")
    for si in valid_foot:
        print(f"  shape {si:<4}: {rol[:, si].cpu().tolist()}")

    # Sanity non-foot.
    non_foot = [s for s in range(shapes_per_arti) if s not in valid_foot][:1]
    if non_foot:
        s = non_foot[0]
        print(f"\n=== Non-foot shape {s} (should be untouched by DR) ===")
        print(f"  slide:     {slide[:, s].cpu().tolist()}")
        print(f"  torsional: {tor[:, s].cpu().tolist()}")
        print(f"  rolling:   {rol[:, s].cpu().tolist()}")


if __name__ == "__main__":
    main()
