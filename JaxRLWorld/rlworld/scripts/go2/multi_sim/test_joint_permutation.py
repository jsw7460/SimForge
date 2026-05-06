"""Joint permutation correctness test for Go2 MultiSimWorld.

Validates that joint permutation logic correctly aligns action and
observation ordering across Genesis, Newton, and MuJoCo for Go2.

Tests:
  1. Joint name listing — compare bare names across simulators
  2. Permutation index verification — round-trip consistency
  3. Action permutation — same named-joint gets same action value
  4. Observation permutation — dof_pos, dof_vel, prev_processed_actions
     are correctly reordered to canonical order
  5. End-to-end step — identical canonical action produces consistent
     per-joint responses across simulators
  6. Permutation round-trip — action→obs consistency

Usage:
    python -m rlworld.scripts.go2.multi_sim.test_joint_permutation
"""

import random
import sys
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

NUM_ENVS = 2
SEP = "=" * 80
THIN = "-" * 80
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

num_passed = 0
num_failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global num_passed, num_failed
    if condition:
        num_passed += 1
        print(f"  [{PASS}] {name}")
    else:
        num_failed += 1
        print(f"  [{FAIL}] {name}")
    if detail:
        print(f"         {detail}")


def _build_common_obs_terms():
    from rlworld.rl.configs.observations import ObservationTermConfig
    from rlworld.rl.envs.mdp.observations.common.proprioception import (
        base_ang_vel, command, projected_gravity, dof_pos, dof_vel, prev_processed_actions,
    )

    return [
        ObservationTermConfig(func=base_ang_vel, scale=1.0),
        ObservationTermConfig(func=projected_gravity, scale=1.0),
        ObservationTermConfig(func=command, scale=1.0),
        ObservationTermConfig(func=dof_pos, scale=1.0),
        ObservationTermConfig(func=dof_vel, scale=1.0),
        ObservationTermConfig(func=prev_processed_actions, scale=1.0),
    ]


def _create_envs():
    from rlworld.rl.runners import BaseRunner

    obs = _build_common_obs_terms()

    print("  Creating Genesis env...", end="", flush=True)
    from rlworld.rl.configs.presets.go2_flat.mlp import get_config
    from rlworld.rl.configs.common_config_classes import disable_corruption
    g_cfg = get_config(sim="genesis")
    g_cfg.env.num_envs = NUM_ENVS
    g_cfg.observation.obs_group = {"actor": obs, "critic": obs}
    disable_corruption(g_cfg.observation)
    g_env = BaseRunner._create_env_from_config(g_cfg)
    print(" done")

    print("  Creating Newton env...", end="", flush=True)
    n_cfg = get_config(sim="newton")
    n_cfg.env.num_envs = NUM_ENVS
    n_cfg.observation.obs_group = {"actor": obs, "critic": obs}
    disable_corruption(n_cfg.observation)
    n_env = BaseRunner._create_env_from_config(n_cfg)
    print(" done")

    print("  Creating MuJoCo env...", end="", flush=True)
    m_cfg = get_config(sim="mujoco")
    m_cfg.env.num_envs = NUM_ENVS
    m_cfg.observation.obs_group = {"actor": obs, "critic": obs}
    disable_corruption(m_cfg.observation)
    m_env = BaseRunner._create_env_from_config(m_cfg)
    print(" done")

    return g_env, n_env, m_env


def _bare(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def main():
    print(SEP)
    print("  Joint Permutation Correctness Test — Go2")
    print(SEP)

    print("\n[Setup] Creating environments...")
    g_env, n_env, m_env = _create_envs()

    g_names = list(g_env.act_manager.actuated_joint_names)
    n_names = list(n_env.act_manager.actuated_joint_names)
    m_names = list(m_env.act_manager.actuated_joint_names)

    g_bare = [_bare(n) for n in g_names]
    n_bare = [_bare(n) for n in n_names]
    m_bare = [_bare(n) for n in m_names]

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Joint name sets match
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{THIN}")
    print("  TEST 1: Joint name sets")
    print(THIN)

    check("Genesis-Newton same joint set", set(g_bare) == set(n_bare))
    check("Genesis-MuJoCo same joint set", set(g_bare) == set(m_bare))
    check("All 3 have 12 joints", len(g_bare) == 12 and len(n_bare) == 12 and len(m_bare) == 12,
          f"G={len(g_bare)}, N={len(n_bare)}, M={len(m_bare)}")

    ordering_differs_gn = g_bare != n_bare
    ordering_differs_gm = g_bare != m_bare
    ordering_differs_nm = n_bare != m_bare
    print(f"\n  Joint ordering differs G-N: {ordering_differs_gn}")
    print(f"  Joint ordering differs G-M: {ordering_differs_gm}")
    print(f"  Joint ordering differs N-M: {ordering_differs_nm}")

    if ordering_differs_gn:
        diffs = [(i, g, n) for i, (g, n) in enumerate(zip(g_bare, n_bare)) if g != n]
        print(f"\n  First 5 ordering differences (Genesis vs Newton):")
        for i, g, n in diffs[:5]:
            print(f"    [{i:2d}] Genesis: {g:35s}  Newton: {n}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: _JointPermutation index verification
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{THIN}")
    print("  TEST 2: Permutation index correctness")
    print(THIN)

    from rlworld.rl.envs.multi_sim_world import _JointPermutation, MultiSimWorld

    canonical_names = g_names
    num_actions = g_env.num_actions
    can_bare = [_bare(n) for n in canonical_names]

    for sim_name, env, sim_joint_names in [
        ("Genesis", g_env, g_names),
        ("Newton", n_env, n_names),
        ("MuJoCo", m_env, m_names),
    ]:
        joint_slices = MultiSimWorld._find_joint_obs_slices(env, num_actions)
        obs_dims = env.obs_manager.calculate_obs_dim()

        jp = _JointPermutation(
            canonical_names=canonical_names,
            sim_names=sim_joint_names,
            obs_group_joint_slices=joint_slices,
            obs_group_dims=obs_dims,
            device=env.device,
        )

        sim_bare = [_bare(n) for n in sim_joint_names]

        perm_correct = True
        for s_idx in range(num_actions):
            c_idx = jp.action_perm[s_idx].item()
            if can_bare[c_idx] != sim_bare[s_idx]:
                perm_correct = False
                print(f"    action_perm ERROR at s={s_idx}: "
                      f"canonical[{c_idx}]={can_bare[c_idx]} != sim[{s_idx}]={sim_bare[s_idx]}")
                break

        check(f"{sim_name}: action_perm maps canonical->sim correctly", perm_correct)

        for group in obs_dims:
            dim = obs_dims[group]
            test_vec = torch.arange(dim, dtype=torch.float32, device=env.device).unsqueeze(0)
            permuted = jp.permute_obs({group: test_vec})[group]
            if jp.is_identity:
                check(f"{sim_name}/{group}: identity permutation (no reorder needed)",
                      torch.equal(test_vec, permuted))
            else:
                non_joint_ok = True
                joint_ok = True

                all_joint_positions = set()
                for start, end in joint_slices.get(group, []):
                    for p in range(start, end):
                        all_joint_positions.add(p)

                for pos in range(dim):
                    if pos not in all_joint_positions:
                        if permuted[0, pos].item() != pos:
                            non_joint_ok = False
                            break

                for start, end in joint_slices.get(group, []):
                    for c in range(num_actions):
                        src_pos = permuted[0, start + c].long().item()
                        src_joint_idx = src_pos - start
                        if can_bare[c] != sim_bare[src_joint_idx]:
                            joint_ok = False
                            break

                check(f"{sim_name}/{group}: non-joint obs positions unchanged", non_joint_ok)
                check(f"{sim_name}/{group}: joint obs positions correctly reordered", joint_ok)

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Action permutation functional test
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{THIN}")
    print("  TEST 3: Action permutation — named-joint gets correct value")
    print(THIN)

    for sim_name, env, sim_joint_names in [
        ("Newton", n_env, n_names),
        ("MuJoCo", m_env, m_names),
    ]:
        joint_slices_s = MultiSimWorld._find_joint_obs_slices(env, num_actions)
        obs_dims_s = env.obs_manager.calculate_obs_dim()

        jp = _JointPermutation(
            canonical_names=canonical_names,
            sim_names=sim_joint_names,
            obs_group_joint_slices=joint_slices_s,
            obs_group_dims=obs_dims_s,
            device=env.device,
        )

        canonical_actions = torch.zeros(1, num_actions, device=env.device)
        value_map = {}
        for c_idx, c_name in enumerate(can_bare):
            val = float((c_idx + 1) * 100)
            canonical_actions[0, c_idx] = val
            value_map[c_name] = val

        sim_actions = jp.permute_actions(canonical_actions)

        all_correct = True
        for s_idx in range(num_actions):
            expected = value_map[_bare(sim_joint_names[s_idx])]
            actual = sim_actions[0, s_idx].item()
            if abs(expected - actual) > 1e-6:
                all_correct = False
                print(f"    MISMATCH at sim[{s_idx}]={_bare(sim_joint_names[s_idx])}: "
                      f"expected={expected}, got={actual}")

        check(f"{sim_name}: each sim joint receives its correct canonical action value", all_correct)

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Observation permutation functional test
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{THIN}")
    print("  TEST 4: Observation permutation — dof_pos reordered correctly")
    print(THIN)

    g_env.reset()
    n_env.reset()
    m_env.reset()

    from rlworld.rl.envs.mdp.observations.common.proprioception import dof_pos as dof_pos_fn

    g_dof = dof_pos_fn(g_env).detach()
    n_dof = dof_pos_fn(n_env).detach()
    m_dof = dof_pos_fn(m_env).detach()

    g_joint_vals = {g_bare[j]: g_dof[0, j].item() for j in range(num_actions)}
    n_joint_vals = {n_bare[j]: n_dof[0, j].item() for j in range(num_actions)}
    m_joint_vals = {m_bare[j]: m_dof[0, j].item() for j in range(num_actions)}

    gn_match = all(abs(g_joint_vals[k] - n_joint_vals[k]) < 0.05 for k in g_joint_vals)
    gm_match = all(abs(g_joint_vals[k] - m_joint_vals[k]) < 0.05 for k in g_joint_vals)

    check("Genesis-Newton: same initial dof_pos per joint name", gn_match)
    check("Genesis-MuJoCo: same initial dof_pos per joint name", gm_match)

    if not gn_match:
        for k in sorted(g_joint_vals):
            gv, nv = g_joint_vals[k], n_joint_vals[k]
            if abs(gv - nv) > 1e-4:
                print(f"    {k}: Genesis={gv:.6f} Newton={nv:.6f} diff={abs(gv-nv):.6f}")

    if not gm_match:
        for k in sorted(g_joint_vals):
            gv, mv = g_joint_vals[k], m_joint_vals[k]
            if abs(gv - mv) > 1e-4:
                print(f"    {k}: Genesis={gv:.6f} MuJoCo={mv:.6f} diff={abs(gv-mv):.6f}")

    for sim_name, env, sim_joint_names, raw_dof in [
        ("Newton", n_env, n_names, n_dof),
        ("MuJoCo", m_env, m_names, m_dof),
    ]:
        joint_slices_s = MultiSimWorld._find_joint_obs_slices(env, num_actions)
        obs_dims_s = env.obs_manager.calculate_obs_dim()

        jp = _JointPermutation(
            canonical_names=canonical_names,
            sim_names=sim_joint_names,
            obs_group_joint_slices=joint_slices_s,
            obs_group_dims=obs_dims_s,
            device=env.device,
        )

        sim_obs = env.obs_manager.get_observation()
        canonical_obs = jp.permute_obs(sim_obs)

        # dof_pos is the 4th term (index 3): after base_ang_vel(3) + projected_gravity(3) + command(3)
        dof_pos_start = 3 + 3 + 3  # = 9
        dof_pos_end = dof_pos_start + num_actions  # = 21

        permuted_dof = canonical_obs["actor"][0, dof_pos_start:dof_pos_end]

        g_dof_canonical = g_dof[0]

        all_correct = True
        for c_idx in range(num_actions):
            c_name = can_bare[c_idx]
            expected = g_joint_vals[c_name]
            actual = permuted_dof[c_idx].item()
            if abs(expected - actual) > 0.05:
                all_correct = False
                print(f"    {c_name}: expected={expected:.6f} got={actual:.6f} "
                      f"diff={abs(expected - actual):.6f}")

        check(f"{sim_name}: permuted dof_pos matches Genesis canonical order (per joint name)", all_correct)

    # ══════════════════════════════════════════════════════════════════
    # TEST 5: MultiSimWorld end-to-end
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{THIN}")
    print("  TEST 5: MultiSimWorld end-to-end step")
    print(THIN)

    multi_env = MultiSimWorld([g_env, n_env, m_env])
    multi_env.reset()

    canonical_actions = torch.zeros(NUM_ENVS * 3, num_actions, device=g_env.device)
    for j in range(num_actions):
        canonical_actions[:, j] = float(j + 1) * 0.01

    obs, rew, term, trunc, info = multi_env.step(canonical_actions)

    check("MultiSimWorld step returns correct num_envs",
          obs["actor"].shape[0] == NUM_ENVS * 3,
          f"shape={obs['actor'].shape}")

    dof_pos_start = 9
    dof_pos_end = dof_pos_start + num_actions

    g_slice = obs["actor"][:NUM_ENVS, dof_pos_start:dof_pos_end]
    n_slice = obs["actor"][NUM_ENVS:2*NUM_ENVS, dof_pos_start:dof_pos_end]
    m_slice = obs["actor"][2*NUM_ENVS:, dof_pos_start:dof_pos_end]

    print(f"\n  dof_pos values at canonical position 0 ({can_bare[0]}):")
    print(f"    Genesis: {g_slice[0, 0].item():.6f}")
    print(f"    Newton:  {n_slice[0, 0].item():.6f}")
    print(f"    MuJoCo:  {m_slice[0, 0].item():.6f}")

    print(f"  dof_pos values at canonical position 6 ({can_bare[6]}):")
    print(f"    Genesis: {g_slice[0, 6].item():.6f}")
    print(f"    Newton:  {n_slice[0, 6].item():.6f}")
    print(f"    MuJoCo:  {m_slice[0, 6].item():.6f}")

    ppa_start = dof_pos_start + num_actions + num_actions
    ppa_end = ppa_start + num_actions

    g_ppa = obs["actor"][:NUM_ENVS, ppa_start:ppa_end]
    n_ppa = obs["actor"][NUM_ENVS:2*NUM_ENVS, ppa_start:ppa_end]
    m_ppa = obs["actor"][2*NUM_ENVS:, ppa_start:ppa_end]

    gn_ppa_diff = (g_ppa - n_ppa).abs().max().item()
    gm_ppa_diff = (g_ppa - m_ppa).abs().max().item()
    nm_ppa_diff = (n_ppa - m_ppa).abs().max().item()

    print(f"\n  prev_processed_actions max diff (should be ~0 after permutation):")
    print(f"    |G-N|: {gn_ppa_diff:.6f}")
    print(f"    |G-M|: {gm_ppa_diff:.6f}")
    print(f"    |N-M|: {nm_ppa_diff:.6f}")

    check("Newton-MuJoCo prev_processed_actions match after permutation",
          nm_ppa_diff < 1e-3, f"max_diff={nm_ppa_diff:.6f}")

    # ══════════════════════════════════════════════════════════════════
    # TEST 6: Permutation round-trip
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{THIN}")
    print("  TEST 6: Permutation round-trip (action->obs consistency)")
    print(THIN)

    for sim_name, env, sim_joint_names in [
        ("Newton", n_env, n_names),
        ("MuJoCo", m_env, m_names),
    ]:
        joint_slices_s = MultiSimWorld._find_joint_obs_slices(env, num_actions)
        obs_dims_s = env.obs_manager.calculate_obs_dim()

        jp = _JointPermutation(
            canonical_names=canonical_names,
            sim_names=sim_joint_names,
            obs_group_joint_slices=joint_slices_s,
            obs_group_dims=obs_dims_s,
            device=env.device,
        )

        can_act = torch.arange(1, num_actions + 1, dtype=torch.float32, device=env.device).unsqueeze(0) * 0.001
        sim_act = jp.permute_actions(can_act)

        recovered = torch.zeros_like(can_act)
        sim_bare_names = [_bare(n) for n in sim_joint_names]
        can_bare_names = [_bare(n) for n in canonical_names]
        can_idx = {name: i for i, name in enumerate(can_bare_names)}

        for s in range(num_actions):
            c = can_idx[sim_bare_names[s]]
            recovered[0, c] = sim_act[0, s]

        roundtrip_ok = torch.allclose(can_act, recovered, atol=1e-6)
        check(f"{sim_name}: action round-trip (canonical->sim->canonical) exact", roundtrip_ok,
              f"max_diff={(can_act - recovered).abs().max().item():.8f}")

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  RESULTS: {num_passed} passed, {num_failed} failed")
    print(SEP)

    if num_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
