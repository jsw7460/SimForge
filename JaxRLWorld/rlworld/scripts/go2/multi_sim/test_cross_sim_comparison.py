"""Cross-simulator comparison test for Go2.

Creates Genesis, Newton, MuJoCo environments with:
  - Identical common observation terms (no noise)
  - Each simulator's OWN reward functions (mapped to semantic equivalents)

Then compares observation values, rewards, and trajectory divergence.

Usage:
    python -m rlworld.scripts.go2.multi_sim.test_cross_sim_comparison
"""

import random
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

NUM_ENVS = 4

# =====================================================================
# Common observation terms (RobotData protocol — works on all sims)
# =====================================================================

def _build_common_actor_obs_terms():
    from rlworld.rl.configs.observations import ObservationTermConfig
    from rlworld.rl.envs.mdp.observations.common.proprioception import (
        base_ang_vel, projected_gravity, dof_pos, dof_vel, prev_processed_actions,
    )
    from rlworld.rl.envs.mdp.observations.genesis.exteroception import command

    return [
        ("base_ang_vel",           ObservationTermConfig(func=base_ang_vel, scale=1.0)),
        ("projected_gravity",      ObservationTermConfig(func=projected_gravity, scale=1.0)),
        ("command",                ObservationTermConfig(func=command, scale=1.0)),
        ("dof_pos",                ObservationTermConfig(func=dof_pos, scale=1.0)),
        ("dof_vel",                ObservationTermConfig(func=dof_vel, scale=1.0)),
        ("prev_processed_actions", ObservationTermConfig(func=prev_processed_actions, scale=1.0)),
    ]


# =====================================================================
# Reward term mapping per simulator
# =====================================================================

FEET_LINKS_GENESIS = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


def _genesis_reward_terms():
    from rlworld.rl.configs.rewards import RewardTermConfig
    from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
    from rlworld.rl.envs.mdp.rewards.genesis import mjlab_rewards as rf_g

    return OrderedDict({
        "track_lin_vel": RewardTermConfig(
            rf_common.track_lin_vel, weight=2.0,
            params={"std": 0.5, "penalize_z": True},
        ),
        "track_ang_vel": RewardTermConfig(
            rf_common.track_ang_vel, weight=2.0,
            params={"std": 0.707, "penalize_xy": True},
        ),
        "flat_orientation": RewardTermConfig(
            rf_common.flat_orientation, weight=1.0,
            params={"std": 0.447},
        ),
        "processed_action_rate_l2": RewardTermConfig(
            rf_g.processed_action_rate_l2_mjlab, weight=0.1,
        ),
        "variable_posture": RewardTermConfig(
            rf_g.variable_posture, weight=1.0,
            params={
                "std_standing": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
                },
                "std_walking": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "std_running": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "walking_threshold": 0.05, "running_threshold": 1.5,
            },
        ),
        "joint_pos_limits": RewardTermConfig(
            rf_g.joint_pos_limits_mjlab, weight=1.0,
        ),
        "feet_clearance": RewardTermConfig(
            rf_g.feet_clearance_mjlab, weight=2.0,
            params={"feet_links": FEET_LINKS_GENESIS, "target_height": 0.1, "command_threshold": 0.05},
        ),
        "feet_swing_height": RewardTermConfig(
            rf_g.feet_swing_height_mjlab, weight=0.25,
            params={"feet_links": FEET_LINKS_GENESIS, "target_height": 0.1, "command_threshold": 0.05},
        ),
        "feet_slip": RewardTermConfig(
            rf_g.feet_slip_mjlab, weight=0.1,
            params={"feet_links": FEET_LINKS_GENESIS, "command_threshold": 0.05},
        ),
        "soft_landing": RewardTermConfig(
            rf_g.soft_landing_mjlab, weight=1e-5,
            params={"feet_links": FEET_LINKS_GENESIS, "command_threshold": 0.05},
        ),
    })


def _newton_reward_terms():
    from rlworld.rl.configs.rewards import RewardTermConfig
    from rlworld.rl.configs.robots.go2 import Go2Config
    from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
    from rlworld.rl.envs.mdp.rewards.newton import mjlab_rewards as rf_n

    robot = Go2Config()
    feet = robot.prefixed_foot_names

    return OrderedDict({
        "track_lin_vel": RewardTermConfig(
            rf_common.track_lin_vel, weight=2.0,
            params={"std": 0.5, "penalize_z": True},
        ),
        "track_ang_vel": RewardTermConfig(
            rf_common.track_ang_vel, weight=2.0,
            params={"std": 0.707, "penalize_xy": True},
        ),
        "flat_orientation": RewardTermConfig(
            rf_common.flat_orientation, weight=1.0,
            params={"std": 0.447},
        ),
        "processed_action_rate_l2": RewardTermConfig(
            rf_n.processed_action_rate_l2_mjlab, weight=0.1,
        ),
        "variable_posture": RewardTermConfig(
            rf_n.variable_posture, weight=1.0,
            params={
                "std_standing": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
                },
                "std_walking": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "std_running": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "walking_threshold": 0.05, "running_threshold": 1.5,
            },
        ),
        "joint_pos_limits": RewardTermConfig(
            rf_n.joint_pos_limits_mjlab, weight=1.0,
        ),
        "feet_clearance": RewardTermConfig(
            rf_n.feet_clearance_mjlab, weight=2.0,
            params={"feet_bodies": feet, "target_height": 0.1, "command_threshold": 0.05},
        ),
        "feet_swing_height": RewardTermConfig(
            rf_n.feet_swing_height_mjlab, weight=0.25,
            params={"feet_bodies": feet, "target_height": 0.1, "command_threshold": 0.05},
        ),
        "feet_slip": RewardTermConfig(
            rf_n.feet_slip_mjlab, weight=0.1,
            params={"feet_bodies": feet, "command_threshold": 0.05},
        ),
        "soft_landing": RewardTermConfig(
            rf_n.soft_landing_mjlab, weight=1e-5,
            params={"feet_bodies": feet, "command_threshold": 0.05},
        ),
    })


def _mujoco_reward_terms():
    import math
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from rlworld.rl.configs.rewards import RewardTermConfig
    from rlworld.rl.envs.mdp.rewards.common import reward_terms as rf_common
    from rlworld.rl.envs.mdp.rewards.mujoco import reward_terms as rf_m

    site_names = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")

    return OrderedDict({
        "track_lin_vel": RewardTermConfig(
            rf_common.track_lin_vel, weight=2.0,
            params={"std": 0.5, "penalize_z": True},
        ),
        "track_ang_vel": RewardTermConfig(
            rf_common.track_ang_vel, weight=2.0,
            params={"std": 0.707, "penalize_xy": True},
        ),
        "flat_orientation": RewardTermConfig(
            rf_m.flat_orientation, weight=1.0,
            params={"std": 0.447},
        ),
        "processed_action_rate_l2": RewardTermConfig(
            rf_m.raw_action_rate_l2, weight=0.1,
        ),
        "variable_posture": RewardTermConfig(
            rf_m.variable_posture, weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=(".*",)),
                "std_standing": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
                },
                "std_walking": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "std_running": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "walking_threshold": 0.05, "running_threshold": 1.5,
            },
        ),
        "joint_pos_limits": RewardTermConfig(
            rf_m.joint_pos_limits, weight=1.0,
        ),
        "feet_clearance": RewardTermConfig(
            rf_m.feet_clearance, weight=2.0,
            params={"asset_cfg": SceneEntityCfg(name="robot", site_names=site_names),
                     "target_height": 0.1, "command_threshold": 0.05},
        ),
        "feet_swing_height": RewardTermConfig(
            rf_m.feet_swing_height, weight=0.25,
            params={"sensor_name": "feet_ground_contact",
                     "asset_cfg": SceneEntityCfg(name="robot", site_names=site_names),
                     "target_height": 0.1, "command_threshold": 0.05},
        ),
        "feet_slip": RewardTermConfig(
            rf_m.feet_slip, weight=0.1,
            params={"sensor_name": "feet_ground_contact",
                     "asset_cfg": SceneEntityCfg(name="robot", site_names=site_names),
                     "command_threshold": 0.05},
        ),
        "soft_landing": RewardTermConfig(
            rf_m.soft_landing, weight=1e-5,
            params={"sensor_name": "feet_ground_contact", "command_threshold": 0.05},
        ),
    })


# =====================================================================
# Semantic reward names (the intersection across all 3 sims)
# =====================================================================

REWARD_NAMES = [
    "track_lin_vel", "track_ang_vel", "flat_orientation", "processed_action_rate_l2",
    "variable_posture", "joint_pos_limits",
    "feet_clearance", "feet_swing_height", "feet_slip", "soft_landing",
]


# =====================================================================
# Environment creation
# =====================================================================

def _create_genesis_env():
    from rlworld.rl.configs.presets.go2_flat.mlp import get_config
    from rlworld.rl.runners import BaseRunner

    cfg = get_config(sim="genesis")
    cfg.env.num_envs = NUM_ENVS
    obs = [t for _, t in _build_common_actor_obs_terms()]
    cfg.observation.obs_group = {"actor": obs, "critic": obs}
    cfg.observation.enable_noise = False
    cfg.reward.reward_terms = _genesis_reward_terms()
    return BaseRunner._create_env_from_config(cfg)


def _create_newton_env():
    from rlworld.rl.configs.presets.go2_flat.mlp import get_config
    from rlworld.rl.runners import BaseRunner

    cfg = get_config(sim="newton")
    cfg.env.num_envs = NUM_ENVS
    obs = [t for _, t in _build_common_actor_obs_terms()]
    cfg.observation.obs_group = {"actor": obs, "critic": obs}
    cfg.observation.enable_noise = False
    cfg.reward.reward_terms = _newton_reward_terms()
    return BaseRunner._create_env_from_config(cfg)


def _create_mujoco_env():
    from rlworld.rl.configs.presets.go2_flat.mlp import get_config
    from rlworld.rl.runners import BaseRunner

    cfg = get_config(sim="mujoco")
    cfg.env.num_envs = NUM_ENVS
    obs = [t for _, t in _build_common_actor_obs_terms()]
    cfg.observation.obs_group = {"actor": obs, "critic": obs}
    cfg.observation.enable_noise = False
    cfg.reward.reward_terms = _mujoco_reward_terms()
    return BaseRunner._create_env_from_config(cfg)


# =====================================================================
# Helpers
# =====================================================================

OBS_NAMES = [name for name, _ in _build_common_actor_obs_terms()]

SEP = "=" * 90
THIN = "-" * 90
SIM_NAMES = ["Genesis", "Newton", "MuJoCo"]


def _compute_obs_raw(env) -> Dict[str, torch.Tensor]:
    results = {}
    for name, term in _build_common_actor_obs_terms():
        params = term.params or {}
        results[name] = term.func(env, **params).detach().clone()
    return results


def _fmt(v: float, width: int = 10) -> str:
    if abs(v) < 1e-6:
        return f"{'~0':>{width}}"
    return f"{v:>+{width}.6f}"


def _print_obs_table(title: str, obs_per_sim: Dict[str, Dict[str, torch.Tensor]]):
    print(f"\n{THIN}")
    print(f"  {title}")
    print(THIN)

    for tn in OBS_NAMES:
        shape = list(obs_per_sim[SIM_NAMES[0]][tn].shape)
        print(f"\n  [{tn}]  shape={shape}")
        for sn in SIM_NAMES:
            v = obs_per_sim[sn][tn]
            print(f"    {sn:>8s}:  mean={_fmt(v.mean().item())}  "
                  f"std={v.std().item():8.6f}  "
                  f"min={_fmt(v.min().item())}  max={_fmt(v.max().item())}")
        for i in range(3):
            for j in range(i+1, 3):
                a, b = obs_per_sim[SIM_NAMES[i]][tn], obs_per_sim[SIM_NAMES[j]][tn]
                d = (a - b).abs()
                print(f"    |{SIM_NAMES[i][:1]}-{SIM_NAMES[j][:1]}|: "
                      f"mean={d.mean().item():.6f}  max={d.max().item():.6f}")


def _print_reward_table(title: str, rew_per_sim: Dict[str, Dict[str, torch.Tensor]]):
    print(f"\n{THIN}")
    print(f"  {title}")
    print(THIN)

    for tn in REWARD_NAMES:
        print(f"\n  [{tn}]")
        for sn in SIM_NAMES:
            v = rew_per_sim[sn].get(tn)
            if v is None:
                print(f"    {sn:>8s}:  (not available)")
                continue
            print(f"    {sn:>8s}:  mean={_fmt(v.mean().item())}  "
                  f"per_env=[{', '.join(f'{x.item():.4f}' for x in v)}]")
        available = [sn for sn in SIM_NAMES if tn in rew_per_sim[sn]]
        for i in range(len(available)):
            for j in range(i+1, len(available)):
                a = rew_per_sim[available[i]][tn]
                b = rew_per_sim[available[j]][tn]
                d = (a - b).abs()
                print(f"    |{available[i][:1]}-{available[j][:1]}|: "
                      f"mean={d.mean().item():.6f}  max={d.max().item():.6f}")


# =====================================================================
# Main
# =====================================================================

def main():
    print(SEP)
    print(f"  Cross-Simulator Comparison: Go2  (num_envs={NUM_ENVS})")
    print(SEP)

    print("\n[1/6] Creating environments...")
    print("  Genesis...", end="", flush=True)
    g_env = _create_genesis_env()
    print(" done")
    print("  Newton...", end="", flush=True)
    n_env = _create_newton_env()
    print(" done")
    print("  MuJoCo...", end="", flush=True)
    m_env = _create_mujoco_env()
    print(" done")

    envs = OrderedDict({"Genesis": g_env, "Newton": n_env, "MuJoCo": m_env})

    print("\n[2/6] Dimension check:")
    for sn, env in envs.items():
        d = env.calculate_obs_dim()
        print(f"  {sn:>8s}: obs={d}, act={env.num_actions}, n_env={env.num_envs}")
    dims = [e.calculate_obs_dim() for e in envs.values()]
    acts = [e.num_actions for e in envs.values()]
    assert all(d == dims[0] for d in dims), f"Obs mismatch: {dims}"
    assert all(a == acts[0] for a in acts), f"Act mismatch: {acts}"
    print("  -> All match!")

    print("\n[3/6] Initial state after reset")
    for env in envs.values():
        env.reset()

    init_obs = {sn: _compute_obs_raw(env) for sn, env in envs.items()}
    _print_obs_table("INITIAL OBSERVATIONS (after reset)", init_obs)

    print(f"\n[4/6] Step with ZERO actions")
    zero_step = {}
    for sn, env in envs.items():
        z = torch.zeros(NUM_ENVS, env.num_actions, device=env.device)
        obs, rew, term, trunc, info = env.step(z)
        zero_step[sn] = {"obs_raw": _compute_obs_raw(env), "rew": rew, "info": info}

    _print_obs_table("OBSERVATIONS AFTER ZERO-ACTION STEP", {sn: d["obs_raw"] for sn, d in zero_step.items()})

    zero_rew = {}
    for sn in SIM_NAMES:
        rpt = zero_step[sn]["info"].get("rewards_per_type", {})
        zero_rew[sn] = {k: rpt[k].detach().clone() for k in rpt}
    _print_reward_table("REWARDS AFTER ZERO-ACTION STEP (from env.step)", zero_rew)

    print(f"\n  [TOTAL REWARD]")
    for sn in SIM_NAMES:
        r = zero_step[sn]["rew"]
        print(f"    {sn:>8s}: {r.mean().item():+.6f}  [{', '.join(f'{x.item():.4f}' for x in r)}]")

    print(f"\n[5/6] Step with IDENTICAL random actions")
    rand_act = torch.randn(NUM_ENVS, acts[0], device=g_env.device) * 0.1
    rand_act = rand_act.clamp(-1.0, 1.0)
    print(f"  actions: mean={rand_act.mean().item():.4f} std={rand_act.std().item():.4f}")

    rand_step = {}
    for sn, env in envs.items():
        a = rand_act.to(env.device)
        obs, rew, term, trunc, info = env.step(a)
        rand_step[sn] = {"obs_raw": _compute_obs_raw(env), "rew": rew, "info": info}

    _print_obs_table("OBSERVATIONS AFTER RANDOM-ACTION STEP", {sn: d["obs_raw"] for sn, d in rand_step.items()})

    rand_rew = {}
    for sn in SIM_NAMES:
        rpt = rand_step[sn]["info"].get("rewards_per_type", {})
        rand_rew[sn] = {k: rpt[k].detach().clone() for k in rpt}
    _print_reward_table("REWARDS AFTER RANDOM-ACTION STEP", rand_rew)

    print(f"\n  [TOTAL REWARD]")
    for sn in SIM_NAMES:
        r = rand_step[sn]["rew"]
        print(f"    {sn:>8s}: {r.mean().item():+.6f}  [{', '.join(f'{x.item():.4f}' for x in r)}]")

    NUM_STEPS = 10
    print(f"\n[6/6] {NUM_STEPS}-step trajectory (same actions per step)")

    for env in envs.values():
        env.reset()

    all_actions = [torch.randn(NUM_ENVS, acts[0], device=g_env.device) * 0.05 for _ in range(NUM_STEPS)]

    traj_obs = {sn: {tn: [] for tn in OBS_NAMES} for sn in SIM_NAMES}
    traj_rew = {sn: [] for sn in SIM_NAMES}
    traj_rew_terms = {sn: {tn: [] for tn in REWARD_NAMES} for sn in SIM_NAMES}

    for si in range(NUM_STEPS):
        for sn, env in envs.items():
            a = all_actions[si].to(env.device)
            obs, rew, term, trunc, info = env.step(a)
            raw = _compute_obs_raw(env)
            for tn in OBS_NAMES:
                traj_obs[sn][tn].append(raw[tn])
            traj_rew[sn].append(rew.detach().clone())
            rpt = info.get("rewards_per_type", {})
            for tn in REWARD_NAMES:
                traj_rew_terms[sn][tn].append(
                    rpt[tn].detach().clone() if tn in rpt else torch.zeros(NUM_ENVS, device=env.device)
                )

    print(f"\n{THIN}")
    print(f"  OBS DIVERGENCE PER STEP (mean |Genesis-X|, env-averaged)")
    print(THIN)
    short_obs = [n[:10] for n in OBS_NAMES]
    print(f"  {'step':>4s}", end="")
    for s in short_obs:
        print(f"  {s:>12s}  ", end="")
    print()
    print(f"  {'':>4s}", end="")
    for _ in OBS_NAMES:
        print(f"  {'G-N':>5s} {'G-M':>5s}  ", end="")
    print()

    for si in range(NUM_STEPS):
        print(f"  {si:>4d}", end="")
        for tn in OBS_NAMES:
            gn = (traj_obs["Genesis"][tn][si] - traj_obs["Newton"][tn][si]).abs().mean().item()
            gm = (traj_obs["Genesis"][tn][si] - traj_obs["MuJoCo"][tn][si]).abs().mean().item()
            print(f"  {gn:5.3f} {gm:5.3f}  ", end="")
        print()

    print(f"\n{THIN}")
    print(f"  REWARD DIVERGENCE PER STEP (mean |Genesis-X|, env-averaged)")
    print(THIN)
    short_rew = [n[:14] for n in REWARD_NAMES]
    print(f"  {'step':>4s}", end="")
    for s in short_rew:
        print(f"  {s:>16s}  ", end="")
    print(f"  {'TOTAL':>16s}")
    print(f"  {'':>4s}", end="")
    for _ in REWARD_NAMES:
        print(f"  {'G-N':>7s} {'G-M':>7s}  ", end="")
    print(f"  {'G-N':>7s} {'G-M':>7s}")

    for si in range(NUM_STEPS):
        print(f"  {si:>4d}", end="")
        for tn in REWARD_NAMES:
            gn = (traj_rew_terms["Genesis"][tn][si] - traj_rew_terms["Newton"][tn][si]).abs().mean().item()
            gm = (traj_rew_terms["Genesis"][tn][si] - traj_rew_terms["MuJoCo"][tn][si]).abs().mean().item()
            print(f"  {gn:7.4f} {gm:7.4f}  ", end="")
        gr = traj_rew["Genesis"][si]
        nr = traj_rew["Newton"][si]
        mr = traj_rew["MuJoCo"][si]
        print(f"  {(gr-nr).abs().mean().item():7.4f} {(gr-mr).abs().mean().item():7.4f}")

    print(f"\n  Cumulative reward over {NUM_STEPS} steps:")
    for sn in SIM_NAMES:
        total = sum(r.mean().item() for r in traj_rew[sn])
        print(f"    {sn:>8s}: {total:+.4f}")

    print(f"\n{SEP}")
    print(f"  SUMMARY")
    print(SEP)

    print(f"\n  Obs dims: {dims[0]}")
    print(f"  Act dim:  {acts[0]}")

    print(f"\n  Avg obs divergence ({NUM_STEPS} steps):")
    for tn in OBS_NAMES:
        gn = np.mean([(traj_obs["Genesis"][tn][i] - traj_obs["Newton"][tn][i]).abs().mean().item() for i in range(NUM_STEPS)])
        gm = np.mean([(traj_obs["Genesis"][tn][i] - traj_obs["MuJoCo"][tn][i]).abs().mean().item() for i in range(NUM_STEPS)])
        nm = np.mean([(traj_obs["Newton"][tn][i] - traj_obs["MuJoCo"][tn][i]).abs().mean().item() for i in range(NUM_STEPS)])
        print(f"    {tn:>25s}:  G-N={gn:.6f}  G-M={gm:.6f}  N-M={nm:.6f}")

    print(f"\n  Avg reward divergence ({NUM_STEPS} steps):")
    for tn in REWARD_NAMES:
        gn = np.mean([(traj_rew_terms["Genesis"][tn][i] - traj_rew_terms["Newton"][tn][i]).abs().mean().item() for i in range(NUM_STEPS)])
        gm = np.mean([(traj_rew_terms["Genesis"][tn][i] - traj_rew_terms["MuJoCo"][tn][i]).abs().mean().item() for i in range(NUM_STEPS)])
        nm = np.mean([(traj_rew_terms["Newton"][tn][i] - traj_rew_terms["MuJoCo"][tn][i]).abs().mean().item() for i in range(NUM_STEPS)])
        print(f"    {tn:>25s}:  G-N={gn:.6f}  G-M={gm:.6f}  N-M={nm:.6f}")

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    main()
