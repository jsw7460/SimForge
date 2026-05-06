"""Dump canonical action/obs joint order for Mjlab T1 getup. (See Newton twin.)"""

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
    cfgs = T1GetupConfig(sim_type="mujoco", num_envs=4).build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env

    print("\n=== Mjlab T1 — canonical action order ===")
    am = env.act_manager
    for i, name in enumerate(am.actuated_joint_names):
        print(f"  {i:<3} {name}")

    print("\n=== Mjlab T1 — indexing (canonical -> sim) ===")
    idx = am._indexing
    print(f"  sim_indices: {idx.sim_indices.cpu().tolist()}")

    print("\n=== Mjlab T1 — observation group layout ===")
    import torch

    env.reset()
    zero_act = torch.zeros(env.num_envs, am.total_action_dim, device=env.device)
    env.step(zero_act)
    _dump_obs_groups(env)


if __name__ == "__main__":
    main()
