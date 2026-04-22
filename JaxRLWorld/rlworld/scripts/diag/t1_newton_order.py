"""Dump canonical action/obs joint order for Newton T1 getup.

Prints:
  1. ``act_manager.actuated_joint_names`` — the canonical action/obs
     joint order that the policy sees. Must match across sims for
     cross-sim transfer to work.
  2. ``indexing.sim_indices`` / ``newton_q_indices`` /
     ``newton_qd_indices`` — where those canonical slots land in the
     sim-native joint/q/qd arrays.
  3. Observation term layout per group (actor / critic) — ordered list
     of (term_name, shape) so we can confirm joint-related terms
     (``dof_pos``, ``dof_vel``, ``raw_actions``) sit at the same
     offsets inside the concatenated obs vector across sims.
"""
from __future__ import annotations

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner


def _dump_obs_groups(env) -> None:
    obs_mgr = env.obs_manager
    for group_name in obs_mgr.group_names:
        group = obs_mgr.get_group(group_name)
        print(f"\n  Group {group_name!r}:")
        total = 0
        for i, term_name in enumerate(group.term_names):
            term = group.terms[term_name]
            dim = term.last_obs.shape[-1] if hasattr(term, "last_obs") and term.last_obs is not None else "?"
            print(f"    {i:<2} {term_name:<30} dim={dim} [{total}..{total + (dim if isinstance(dim, int) else 0)})")
            if isinstance(dim, int):
                total += dim
        print(f"    TOTAL: {total}")


def main() -> None:
    cfgs = T1GetupConfig(sim_type="newton", num_envs=4).build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env

    print("\n=== Newton T1 — canonical action order ===")
    am = env.act_manager
    for i, name in enumerate(am.actuated_joint_names):
        print(f"  {i:<3} {name}")

    print("\n=== Newton T1 — indexing (canonical -> sim) ===")
    idx = am._indexing
    print(f"  sim_indices:       {idx.sim_indices.cpu().tolist()}")
    if hasattr(idx, "newton_q_indices") and idx.newton_q_indices is not None:
        print(f"  newton_q_indices:  {idx.newton_q_indices.cpu().tolist()}")
    if hasattr(idx, "newton_qd_indices") and idx.newton_qd_indices is not None:
        print(f"  newton_qd_indices: {idx.newton_qd_indices.cpu().tolist()}")

    print("\n=== Newton T1 — observation group layout ===")
    # Drive one step so obs terms have populated shapes
    import torch
    env.reset()
    zero_act = torch.zeros(env.num_envs, am.total_action_dim, device=env.device)
    env.step(zero_act)
    _dump_obs_groups(env)


if __name__ == "__main__":
    main()
