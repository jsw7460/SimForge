"""How much collision work does a Genesis env carry per step: geoms, broadphase pairs, contacts, constraints.

The GPU time of a Genesis step is dominated by kernels whose cost grows
with the number of contacts and constraint rows (collision constraint
assembly, the Hessian assembly, the narrowphase). Two frameworks running
the same engine on the same robot can therefore differ in step time
purely through what their scenes make the collider do. This builds the
env, steps it with zero actions (standing on the ground) and prints the
rigid options the scene resolved to, the collision geometry (geom count
and kinds, vertex count), and per-step averages of broadphase pairs,
contacts and constraint rows over the envs.

    jaxpy -m jaxrlworld.scripts.diag.perf.genesis_contact_census --num-envs 4096
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
from genesis.utils.misc import qd_to_torch

from jaxrlworld.scripts.diag.perf.env_step_breakdown import build_env

_OPTION_FIELDS = (
    "dt",
    "integrator",
    "constraint_solver",
    "iterations",
    "tolerance",
    "ls_iterations",
    "ls_tolerance",
    "max_collision_pairs",
    "enable_collision",
    "enable_self_collision",
    "enable_multi_contact",
    "enable_mujoco_compatibility",
    "use_gjk_collision",
    "contact_pruning_tolerance",
    "batch_dofs_info",
    "batch_links_info",
    "constraint_timeconst",
    "noslip_iterations",
)


def census(scene, steps: int, step_fn) -> str:
    solver = scene.rigid_solver
    options = solver._options
    lines = ["rigid options: " + ", ".join(f"{name}={getattr(options, name)}" for name in _OPTION_FIELDS)]
    kinds = Counter(str(geom.type).rsplit(".", 1)[-1] for entity in scene.entities for geom in entity.geoms)
    lines.append(
        f"collision geometry: {solver.n_geoms} geoms ({dict(kinds)}), {solver.n_verts} verts, "
        f"{solver.n_links} links, {solver.n_dofs} dofs, n_envs {solver.n_envs}"
    )
    broad, contacts, rows = [], [], []
    for _ in range(steps):
        step_fn()
        broad.append(qd_to_torch(solver.collider.collider_state.n_broad_pairs).float())
        contacts.append(qd_to_torch(solver.collider.collider_state.n_contacts).float())
        rows.append(qd_to_torch(solver.constraint_solver.constraint_state.n_constraints).float())
    for name, values in (("broadphase pairs", broad), ("contacts", contacts), ("constraint rows", rows)):
        stacked = torch.stack(values)
        lines.append(
            f"{name:>16} per env: mean {stacked.mean():7.2f}  max {stacked.max():6.0f}  "
            f"(over {steps} steps, read after each control step)"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="jaxrlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--no-stagger", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = ap.parse_args()

    env = build_env(args.config, "genesis", args.num_envs, args.set)
    env.reset()
    if not args.no_stagger:
        env.termination_manager.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    for _ in range(args.warmup):
        env.step(actions)
    print(f"[census] {args.config} num_envs={args.num_envs}")
    print(census(env.scene_manager.scene, args.steps, lambda: env.step(actions)))


if __name__ == "__main__":
    main()
