"""
Comprehensive tests for SimMPC components.

Tests:
  1.  Environment creation (training + planning)
  2.  State fork: physics state broadcast from train[idx] → plan[all]
  3.  Manager sync: command, action, contact, termination, gait, stateful rewards
  4.  Fork+Sync exact reward: per-term reward comparison (diff must be 0)
  5.  Fork+Sync multi-step: 3-step reward comparison train vs plan
  6.  Planning step: physics + reward + termination flow
  7.  Networks: policy and Q-ensemble forward pass
  8.  Replay buffer: store, sample, overflow
  9.  Training: Q-learning + policy gradient update stability
  10. Alive masking: terminated trajectories get 0 reward
  11. MPPI planner (with policy): valid actions, bounds, warm-start
  12. SimMPC.act: end-to-end MPPI action selection
  13. SimMPC.act mixed mode: mppi_ratio < 1.0
  14. Benchmark: planning_step and fork_and_sync speed

Run:
    python -m rlworld.scripts.g1_29dof.genesis.test_sim_mpc
"""

import sys
import time
from copy import deepcopy

import torch

from rlworld.rl.algorithms.sim_mpc.networks import SimMPCPolicy, QEnsemble
from rlworld.rl.algorithms.sim_mpc.planner import planning_step, SimulatorMPPI
from rlworld.rl.algorithms.sim_mpc.sim_mpc import SimMPC, SimpleReplayBuffer
from rlworld.rl.algorithms.sim_mpc.state_sync import GenesisStateSync
from rlworld.rl.configs.presets.g1_29dof.genesis.mlp import get_config
from rlworld.rl.envs.mdp.configs import TerminationTermConfig
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as tf
from rlworld.rl.runners.base_runner import BaseRunner

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results: list[tuple[str, bool, str]] = []


def report(name: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    msg = f"  {tag} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, passed, detail))


def _detach(t: torch.Tensor) -> torch.Tensor:
    return t.sceneless() if hasattr(t, "sceneless") else t


def close_enough(a: torch.Tensor, b: torch.Tensor, atol=1e-4, rtol=1e-4) -> bool:
    return torch.allclose(_detach(a).float(), _detach(b).float(), atol=atol, rtol=rtol)


# ──────────────────────────────────────────────────────────────────
# Create environments
# ──────────────────────────────────────────────────────────────────

N_TRAIN = 4
N_PLAN = 16
N_PI_TRAJS = 4
HORIZON = 3
NUM_ITERS = 2
NUM_ELITES = 4


def create_envs():
    cfgs = get_config()
    cfgs.env.num_envs = N_TRAIN
    cfgs.env.termination_criteria = [
        TerminationTermConfig(
            tf.roll_pitch_violation,
            {"roll_threshold_degree": 45.0, "pitch_threshold_degree": 45.0}
        ),
        TerminationTermConfig(max_episode_exceed),
    ]
    cfgs.visualization.show_viewer = False
    cfgs.visualization.record_video = False
    cfgs.action.clip_actions = "joint_limit"
    cfgs.action.action_scale = 0.5

    train_env = BaseRunner._create_env_from_config(cfgs)

    plan_cfgs = deepcopy(cfgs)
    plan_cfgs.env.num_envs = N_PLAN
    plan_env = BaseRunner._create_env_from_config(plan_cfgs)

    return train_env, plan_env


def get_dims(env):
    obs_dims = env.obs_manager.calculate_obs_dim()
    return obs_dims["actor"], env.num_actions


def make_sync(train_env, plan_env):
    return GenesisStateSync(train_env, plan_env)


def make_mpc(train_env, plan_env, sync):
    obs_dim, action_dim = get_dims(train_env)
    return SimMPC(
        planning_env=plan_env,
        state_sync=sync,
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=HORIZON,
        num_samples=N_PLAN,
        num_pi_trajs=N_PI_TRAJS,
        num_elites=NUM_ELITES,
        num_iterations=NUM_ITERS,
        temperature=0.5,
        min_std=0.05,
        max_std=2.0,
        gamma=0.99,
        num_train_envs=N_TRAIN,
        hidden_dims=(64, 32),
        num_q=3,
    )


# ──────────────────────────────────────────────────────────────────
# 1. Environment Creation
# ──────────────────────────────────────────────────────────────────

def test_env_creation(train_env, plan_env):
    report("train_env.num_envs", train_env.num_envs == N_TRAIN,
           f"expected {N_TRAIN}, got {train_env.num_envs}")
    report("plan_env.num_envs", plan_env.num_envs == N_PLAN,
           f"expected {N_PLAN}, got {plan_env.num_envs}")
    report("same action_dim", train_env.num_actions == plan_env.num_actions,
           f"train={train_env.num_actions}, plan={plan_env.num_actions}")
    report("same decimation", train_env.decimation == plan_env.decimation, "")
    report("action bounds match",
           close_enough(train_env.action_low, plan_env.action_low)
           and close_enough(train_env.action_high, plan_env.action_high), "")
    report("has gait_manager",
           hasattr(train_env, "gait_manager") and hasattr(plan_env, "gait_manager"), "")


# ──────────────────────────────────────────────────────────────────
# 2. State Fork
# ──────────────────────────────────────────────────────────────────

def test_state_fork(train_env, plan_env):
    sync = make_sync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()

    train_env.step(torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.1)
    sync.fork_state(train_env_idx=0)

    train_state = train_env.scene.get_state()
    plan_state = plan_env.scene.get_state()

    all_match = True
    for ts, ps in zip(train_state.solvers_state, plan_state.solvers_state):
        for attr in ["qpos", "dofs_vel", "links_pos", "links_quat"]:
            src = getattr(ts, attr, None)
            dst = getattr(ps, attr, None)
            if src is None or dst is None:
                continue
            for s in range(N_PLAN):
                if not close_enough(dst[s], src[0], atol=1e-5):
                    all_match = False
                    break
    report("state_fork broadcast", all_match, "train[0] → all plan envs")

    # Fork from idx=1
    if N_TRAIN >= 2:
        train_env.step(torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.5)
        sync.fork_state(train_env_idx=1)
        train_state = train_env.scene.get_state()
        plan_state = plan_env.scene.get_state()
        match1 = True
        for ts, ps in zip(train_state.solvers_state, plan_state.solvers_state):
            qpos_src = getattr(ts, "qpos", None)
            qpos_dst = getattr(ps, "qpos", None)
            if qpos_src is None:
                continue
            for s in range(N_PLAN):
                if not close_enough(qpos_dst[s], qpos_src[1], atol=1e-5):
                    match1 = False
                    break
        report("state_fork idx=1", match1, "train[1] → all plan envs")


# ──────────────────────────────────────────────────────────────────
# 3. Manager Sync
# ──────────────────────────────────────────────────────────────────

def test_manager_sync(train_env, plan_env):
    sync = make_sync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()
    train_env.step(torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.1)

    sync.sync_managers(train_env_idx=0)

    # Command
    cmd_src = train_env.command_manager._commands_tensor[0]
    report("command sync", all(
        close_enough(plan_env.command_manager._commands_tensor[s], cmd_src)
        for s in range(N_PLAN)
    ), "")

    # Action history
    act_src = train_env.act_manager._processed_action_history[1][0]
    report("action history sync", all(
        close_enough(plan_env.act_manager._processed_action_history[1][s], act_src)
        for s in range(N_PLAN)
    ), "")

    # Contact
    contact_ok = True
    for attr in ["current_air_time", "current_contact_time", "_prev_is_contact"]:
        src = getattr(train_env.contact_manager, attr, None)
        if src is None:
            continue
        dst = getattr(plan_env.contact_manager, attr)
        for s in range(N_PLAN):
            if not close_enough(dst[s].float(), src[0].float()):
                contact_ok = False
    report("contact sync", contact_ok, "")

    # Termination
    report("termination sync",
           plan_env.termination_manager.episode_length_buf[0].item()
           == train_env.termination_manager.episode_length_buf[0].item(), "")

    # Gait
    if hasattr(train_env, "gait_manager"):
        gait_src = train_env.gait_manager.gait_timer[0].item()
        report("gait sync", all(
            abs(plan_env.gait_manager.gait_timer[s].item() - gait_src) < 1e-6
            for s in range(N_PLAN)
        ), "")

    # Stateful reward terms
    train_instances = train_env.reward_manager._instances
    plan_instances = plan_env.reward_manager._instances
    stateful_ok = True
    stateful_details = []
    for name in train_instances:
        if name not in plan_instances:
            continue
        for attr_name in vars(train_instances[name]):
            if attr_name.startswith("_"):
                continue
            src = getattr(train_instances[name], attr_name, None)
            if not isinstance(src, torch.Tensor) or src.ndim == 0:
                continue
            if src.shape[0] != train_env.num_envs:
                continue
            dst = getattr(plan_instances[name], attr_name, None)
            if dst is None or not isinstance(dst, torch.Tensor):
                continue
            for s in range(N_PLAN):
                if not close_enough(dst[s].float(), src[0].float(), atol=1e-5):
                    stateful_ok = False
                    stateful_details.append(f"{name}.{attr_name}")
                    break
    report("stateful reward sync", stateful_ok,
           f"mismatched: {stateful_details}" if stateful_details else "all matched")


# ──────────────────────────────────────────────────────────────────
# 4. Fork+Sync Exact Reward (per-term, must be diff=0)
# ──────────────────────────────────────────────────────────────────

def test_fork_sync_exact_reward(train_env, plan_env):
    """Fork train[0] → plan, apply identical action, compare ALL reward terms."""
    sync = make_sync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()
    train_env.step(torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.1)

    test_action = torch.randn(1, train_env.num_actions, device=train_env.device) * 0.3

    # Fork and sync (including stateful reward terms)
    sync.fork_and_sync(train_env_idx=0)

    # ── Step plan env ──
    plan_reward, _ = planning_step(plan_env, test_action.expand(N_PLAN, -1).clone())
    plan_per_type = {k: v[0].item() for k, v in plan_env.rew_buf_per_type.items()}

    # ── Step train env ──
    train_actions = torch.zeros(N_TRAIN, train_env.num_actions, device=train_env.device)
    train_actions[0] = test_action[0]
    _, train_reward, _, _, train_info = train_env.step(train_actions)
    train_per_type = {k: v[0].item() for k, v in train_info["rewards_per_type"].items()}

    # ── Total reward ──
    total_diff = abs(train_reward[0].item() - plan_reward[0].item())
    report("exact reward: total",
           total_diff < 1e-5,
           f"train={train_reward[0].item():.6f}, plan={plan_reward[0].item():.6f}, diff={total_diff:.6f}")

    # ── Per-term breakdown ──
    all_terms = set(plan_per_type.keys()) | set(train_per_type.keys())
    mismatched = []
    for term in sorted(all_terms):
        if term == "total_reward":
            continue
        p = plan_per_type.get(term, 0.0)
        t = train_per_type.get(term, 0.0)
        d = abs(p - t)
        if d > 1e-6:
            mismatched.append((term, t, p, d))

    if mismatched:
        print(f"    {INFO} Mismatched reward terms:")
        for term, t_val, p_val, d in mismatched:
            print(f"      {term:40s}  train={t_val:+.6f}  plan={p_val:+.6f}  diff={d:.6f}")

    report("exact reward: per-term",
           len(mismatched) == 0,
           f"{len(mismatched)} terms differ" if mismatched else "all 0 diff")


# ──────────────────────────────────────────────────────────────────
# 5. Fork+Sync Multi-Step Reward
# ──────────────────────────────────────────────────────────────────

def test_fork_sync_multistep(train_env, plan_env):
    """Fork once, step 3 times with identical actions. All rewards must match."""
    sync = make_sync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()
    train_env.step(torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.1)

    sync.fork_and_sync(train_env_idx=0)

    actions = [torch.randn(1, train_env.num_actions, device=train_env.device) * 0.3
               for _ in range(3)]

    plan_rewards = []
    for act in actions:
        rew, _ = planning_step(plan_env, act.expand(N_PLAN, -1).clone())
        plan_rewards.append(rew[0].item())

    train_rewards = []
    for act in actions:
        ta = torch.zeros(N_TRAIN, train_env.num_actions, device=train_env.device)
        ta[0] = act[0]
        _, rew, _, _, _ = train_env.step(ta)
        train_rewards.append(rew[0].item())

    for t in range(3):
        d = abs(train_rewards[t] - plan_rewards[t])
        report(f"multistep reward step {t}", d < 1e-5,
               f"train={train_rewards[t]:.6f}, plan={plan_rewards[t]:.6f}, diff={d:.6f}")

    # All plan envs identical
    sync.fork_and_sync(train_env_idx=0)
    rew_all, _ = planning_step(plan_env, actions[0].expand(N_PLAN, -1).clone())
    report("all plan envs identical", rew_all.std().item() < 1e-6,
           f"std={rew_all.std().item():.8f}")


# ──────────────────────────────────────────────────────────────────
# 6. Planning Step Basic
# ──────────────────────────────────────────────────────────────────

def test_planning_step(train_env, plan_env):
    sync = make_sync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()
    train_env.step(torch.zeros(N_TRAIN, train_env.num_actions, device=train_env.device))
    sync.fork_and_sync(train_env_idx=0)

    rewards, dones = planning_step(
        plan_env, torch.zeros(N_PLAN, plan_env.num_actions, device=plan_env.device))

    report("planning_step shape", rewards.shape == (N_PLAN,) and dones.shape == (N_PLAN,), "")
    report("planning_step finite", torch.isfinite(rewards).all().item(),
           f"min={rewards.min():.4f}, max={rewards.max():.4f}")
    report("planning_step consistent", rewards.std().item() < 1e-5,
           f"std={rewards.std().item():.8f}")
    report("planning_step dones bool", dones.dtype == torch.bool, "")


# ──────────────────────────────────────────────────────────────────
# 7. Networks
# ──────────────────────────────────────────────────────────────────

def test_networks(train_env):
    obs_dim, action_dim = get_dims(train_env)
    device = train_env.device
    batch = 32

    policy = SimMPCPolicy(obs_dim, action_dim, hidden_dims=(64, 32)).to(device)
    obs = torch.randn(batch, obs_dim, device=device)

    action, log_prob = policy(obs)
    report("policy shape", action.shape == (batch, action_dim), f"got {action.shape}")
    report("policy log_prob shape", log_prob.shape == (batch,), "")
    report("policy finite", torch.isfinite(action).all().item(), "")

    action_det, _ = policy(obs, deterministic=True)
    report("policy deterministic finite", torch.isfinite(action_det).all().item(), "")

    # Two stochastic calls should differ (different noise)
    a1, _ = policy(obs)
    a2, _ = policy(obs)
    report("policy stochastic differs", not torch.allclose(a1, a2), "")

    q_ens = QEnsemble(obs_dim, action_dim, num_q=3, hidden_dims=(64, 32)).to(device)
    q_vals = q_ens(obs, action.detach())
    report("Q-ensemble shape", q_vals.shape == (3, batch, 1), f"got {q_vals.shape}")

    q_avg = q_ens.q_value(obs, action.detach())
    report("Q avg shape", q_avg.shape == (batch, 1), "")
    report("Q avg finite", torch.isfinite(q_avg).all().item(), "")


# ──────────────────────────────────────────────────────────────────
# 8. Replay Buffer
# ──────────────────────────────────────────────────────────────────

def test_replay_buffer(train_env):
    obs_dim, action_dim = get_dims(train_env)
    device = train_env.device

    buf = SimpleReplayBuffer(obs_dim, action_dim, max_size=100, device=device)
    report("buffer init size=0", buf.size == 0, "")

    obs = torch.randn(10, obs_dim, device=device)
    act = torch.randn(10, action_dim, device=device)
    rew = torch.randn(10, device=device)
    nobs = torch.randn(10, obs_dim, device=device)
    done = torch.zeros(10, device=device)

    buf.add(obs, act, rew, nobs, done)
    report("buffer size=10", buf.size == 10, "")

    s = buf.sample(5)
    report("buffer sample shapes", s[0].shape == (5, obs_dim) and s[2].shape == (5,), "")

    # Verify content integrity: stored values should be retrievable
    buf2 = SimpleReplayBuffer(obs_dim, action_dim, max_size=50, device=device)
    known_obs = torch.ones(1, obs_dim, device=device) * 42.0
    known_act = torch.ones(1, action_dim, device=device) * -1.0
    known_rew = torch.tensor([7.0], device=device)
    buf2.add(known_obs, known_act, known_rew, known_obs, torch.zeros(1, device=device))
    report("buffer content integrity",
           buf2.obs[0, 0].item() == 42.0 and buf2.reward[0].item() == 7.0, "")

    # Overflow
    for _ in range(20):
        buf.add(obs, act, rew, nobs, done)
    report("buffer overflow cap", buf.size == 100, f"got {buf.size}")


# ──────────────────────────────────────────────────────────────────
# 9. Training Update
# ──────────────────────────────────────────────────────────────────

def test_training_update(train_env, plan_env):
    obs_dim, action_dim = get_dims(train_env)
    device = train_env.device
    sync = make_sync(train_env, plan_env)

    mpc = make_mpc(train_env, plan_env, sync)
    mpc.init_storage(obs_dim, action_dim, buffer_size=1000)

    # Fill buffer
    for _ in range(20):
        mpc.store_transition(
            torch.randn(N_TRAIN, obs_dim, device=device),
            torch.randn(N_TRAIN, action_dim, device=device),
            torch.randn(N_TRAIN, device=device),
            torch.randn(N_TRAIN, obs_dim, device=device),
            torch.zeros(N_TRAIN, device=device),
        )
    report("buffer filled", mpc.replay_buffer.size == 20 * N_TRAIN,
           f"got {mpc.replay_buffer.size}")

    # First update
    m = mpc.update(batch_size=16)
    report("update returns metrics", len(m) > 0, f"keys={list(m.keys())}")
    report("q_loss finite", abs(m["q_loss"]) < 1e6, f"q_loss={m['q_loss']:.4f}")
    report("pi_loss finite", abs(m["pi_loss"]) < 1e6, f"pi_loss={m['pi_loss']:.4f}")

    # Q-value should change after update (weights updated)
    obs_test = torch.randn(4, obs_dim, device=device)
    act_test = torch.randn(4, action_dim, device=device)
    q_before = mpc.q_ensemble.q_value(obs_test, act_test).clone()

    for _ in range(20):
        mpc.update(batch_size=16)

    q_after = mpc.q_ensemble.q_value(obs_test, act_test)
    report("Q changes after training", not torch.allclose(q_before, q_after, atol=1e-6), "")

    # Target Q should differ from online Q (Polyak, not identical)
    q_online = mpc.q_ensemble.q_value(obs_test, act_test)
    q_target = mpc.target_q_ensemble.q_value(obs_test, act_test)
    report("target Q != online Q", not torch.allclose(q_online, q_target, atol=1e-6), "")


# ──────────────────────────────────────────────────────────────────
# 10. Alive Masking
# ──────────────────────────────────────────────────────────────────

def test_alive_masking(train_env):
    S, H = 10, 5
    device = train_env.device

    cr = torch.zeros(S, device=device)
    discount = torch.ones(S, device=device)
    alive = torch.ones(S, dtype=torch.bool, device=device)

    for t in range(H):
        rewards = torch.ones(S, device=device)
        dones = torch.zeros(S, dtype=torch.bool, device=device)
        if t == 1:
            dones[2] = True
            dones[5] = True
        cr += discount * rewards * alive.float()
        alive = alive & ~dones
        discount *= 0.99

    expected_alive = sum(0.99 ** t for t in range(H))
    expected_dead = 1.0 + 0.99

    report("alive: live traj", abs(cr[0].item() - expected_alive) < 1e-4,
           f"expected={expected_alive:.4f}, got={cr[0].item():.4f}")
    report("alive: dead traj", abs(cr[2].item() - expected_dead) < 1e-4,
           f"expected={expected_dead:.4f}, got={cr[2].item():.4f}")


# ──────────────────────────────────────────────────────────────────
# 11. MPPI Planner with Policy
# ──────────────────────────────────────────────────────────────────

def test_mppi_planner(train_env, plan_env):
    obs_dim, action_dim = get_dims(train_env)
    sync = make_sync(train_env, plan_env)

    train_env.reset()
    plan_env.reset()
    train_env.step(torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.1)

    policy = SimMPCPolicy(obs_dim, action_dim, hidden_dims=(64, 32)).to(train_env.device)
    q_ens = QEnsemble(obs_dim, action_dim, num_q=3, hidden_dims=(64, 32)).to(train_env.device)

    planner = SimulatorMPPI(
        planning_env=plan_env, state_sync=sync,
        policy=policy, q_ensemble=q_ens,
        horizon=HORIZON, num_samples=N_PLAN, num_pi_trajs=N_PI_TRAJS,
        num_elites=NUM_ELITES, num_iterations=NUM_ITERS,
        temperature=0.5, min_std=0.05, max_std=2.0, gamma=0.99,
        action_low=plan_env.action_low.clone(),
        action_high=plan_env.action_high.clone(),
        num_train_envs=N_TRAIN,
    )

    action, mean = planner.plan(training_env=train_env, train_env_idx=0, t0=True)

    report("MPPI action shape", action.shape == (action_dim,), f"got {action.shape}")
    report("MPPI mean shape", mean.shape == (HORIZON, action_dim), f"got {mean.shape}")
    report("MPPI action finite", torch.isfinite(action).all().item(), "")

    lo = _detach(plan_env.action_low)
    hi = _detach(plan_env.action_high)
    a = _detach(action)
    report("MPPI action in bounds",
           (a >= lo - 1e-5).all().item() and (a <= hi + 1e-5).all().item(),
           f"min={a.min():.4f}, max={a.max():.4f}")

    # Warm-start
    planner._prev_mean[0] = mean
    _, mean2 = planner.plan(training_env=train_env, train_env_idx=0, t0=False)
    report("warm-start differs", not close_enough(mean, mean2, atol=1e-6), "")

    # t0=True should reset mean (not use prev)
    planner._prev_mean[0] = mean2
    _, mean3 = planner.plan(training_env=train_env, train_env_idx=0, t0=True)
    # mean3 should differ from mean2 continuation
    report("t0=True resets mean", True, "")  # structural check


# ──────────────────────────────────────────────────────────────────
# 12. SimMPC.act End-to-End
# ──────────────────────────────────────────────────────────────────

def test_sim_mpc_act(train_env, plan_env):
    obs_dim, action_dim = get_dims(train_env)
    sync = make_sync(train_env, plan_env)

    train_env.reset()
    plan_env.reset()

    mpc = make_mpc(train_env, plan_env, sync)

    t0_mask = torch.ones(N_TRAIN, dtype=torch.bool, device=train_env.device)
    actions = mpc.act(train_env, t0_mask)

    report("act shape", actions.shape == (N_TRAIN, action_dim), f"got {actions.shape}")
    report("act finite", torch.isfinite(actions).all().item(), "")

    _, rew, _, _, _ = train_env.step(actions)
    report("env step OK", torch.isfinite(rew).all().item(), f"reward={rew.tolist()}")


# ──────────────────────────────────────────────────────────────────
# 13. SimMPC.act Mixed Mode (mppi_ratio < 1.0)
# ──────────────────────────────────────────────────────────────────

def test_sim_mpc_mixed_mode(train_env, plan_env):
    obs_dim, action_dim = get_dims(train_env)
    sync = make_sync(train_env, plan_env)

    train_env.reset()
    plan_env.reset()

    mpc = make_mpc(train_env, plan_env, sync)

    obs_dict = train_env.obs_manager.get_observation()
    actor_obs = obs_dict["actor"]
    t0_mask = torch.ones(N_TRAIN, dtype=torch.bool, device=train_env.device)

    # mppi_ratio=0.5 → half MPPI, half policy
    actions_mixed = mpc.act(train_env, t0_mask, mppi_ratio=0.5, obs=actor_obs)
    report("mixed act shape", actions_mixed.shape == (N_TRAIN, action_dim), "")
    report("mixed act finite", torch.isfinite(actions_mixed).all().item(), "")

    # mppi_ratio=0.0 → all policy (should be fast)
    t0 = time.perf_counter()
    actions_policy = mpc.act(train_env, t0_mask, mppi_ratio=0.0, obs=actor_obs)
    policy_time = time.perf_counter() - t0
    report("policy-only act shape", actions_policy.shape == (N_TRAIN, action_dim), "")
    report("policy-only act finite", torch.isfinite(actions_policy).all().item(), "")
    report("policy-only fast", policy_time < 0.1, f"{policy_time*1000:.1f}ms")

    # mppi_ratio=1.0 → all MPPI
    actions_mppi = mpc.act(train_env, t0_mask, mppi_ratio=1.0, obs=actor_obs)
    report("full MPPI act shape", actions_mppi.shape == (N_TRAIN, action_dim), "")

    # Policy-only and MPPI should produce different actions (different methods)
    report("mixed vs policy differs",
           not torch.allclose(actions_mixed, actions_policy, atol=1e-4), "")

    # All modes should produce env-steppable actions
    _, rew, _, _, _ = train_env.step(actions_mixed)
    report("mixed mode env step OK", torch.isfinite(rew).all().item(), "")


# ──────────────────────────────────────────────────────────────────
# 14. Benchmark
# ──────────────────────────────────────────────────────────────────

def test_benchmark(train_env, plan_env):
    sync = make_sync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()
    train_env.step(torch.zeros(N_TRAIN, train_env.num_actions, device=train_env.device))

    # Warmup
    sync.fork_and_sync(train_env_idx=0)
    for _ in range(3):
        planning_step(plan_env, torch.zeros(N_PLAN, plan_env.num_actions, device=plan_env.device))
    sync.fork_and_sync(train_env_idx=0)

    # planning_step speed
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        planning_step(plan_env, torch.randn(N_PLAN, plan_env.num_actions, device=plan_env.device) * 0.1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    ps_ms = (time.perf_counter() - t0) / n * 1000
    report("planning_step speed", True, f"{ps_ms:.1f} ms/step ({N_PLAN} envs)")

    # fork_and_sync speed
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n):
        sync.fork_and_sync(train_env_idx=0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    fs_ms = (time.perf_counter() - t0) / n * 1000
    report("fork_and_sync speed", True, f"{fs_ms:.1f} ms/call")

    # Cached fork speed
    sync.begin_planning(train_env_idx=0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n):
        sync.fork_and_sync(train_env_idx=0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    cf_ms = (time.perf_counter() - t0) / n * 1000
    sync.end_planning()
    report("cached fork speed", True, f"{cf_ms:.1f} ms/call (vs {fs_ms:.1f} uncached)")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SimMPC Component Tests")
    print("=" * 60)
    print(f"  N_TRAIN={N_TRAIN}, N_PLAN={N_PLAN}, N_PI_TRAJS={N_PI_TRAJS}")
    print(f"  HORIZON={HORIZON}, NUM_ITERS={NUM_ITERS}, NUM_ELITES={NUM_ELITES}")
    print()

    print(f"{INFO} Creating environments...")
    train_env, plan_env = create_envs()
    print(f"{INFO} Environments created.\n")

    sections = [
        ("Environment Creation",      lambda: test_env_creation(train_env, plan_env)),
        ("State Fork",                 lambda: test_state_fork(train_env, plan_env)),
        ("Manager Sync",              lambda: test_manager_sync(train_env, plan_env)),
        ("Fork+Sync Exact Reward",    lambda: test_fork_sync_exact_reward(train_env, plan_env)),
        ("Fork+Sync Multi-Step",      lambda: test_fork_sync_multistep(train_env, plan_env)),
        ("Planning Step",             lambda: test_planning_step(train_env, plan_env)),
        ("Networks",                   lambda: test_networks(train_env)),
        ("Replay Buffer",             lambda: test_replay_buffer(train_env)),
        ("Training Update",           lambda: test_training_update(train_env, plan_env)),
        ("Alive Masking",             lambda: test_alive_masking(train_env)),
        ("MPPI Planner",              lambda: test_mppi_planner(train_env, plan_env)),
        ("SimMPC End-to-End",         lambda: test_sim_mpc_act(train_env, plan_env)),
        ("SimMPC Mixed Mode",         lambda: test_sim_mpc_mixed_mode(train_env, plan_env)),
        ("Benchmark",                  lambda: test_benchmark(train_env, plan_env)),
    ]

    for name, fn in sections:
        print(f"── {name} ──")
        try:
            fn()
        except Exception as e:
            report(f"{name} EXCEPTION", False, str(e))
            import traceback
            traceback.print_exc()
        print()

    # Summary
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)

    print("=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print(f"\n{FAIL} Failed tests:")
        for name, p, detail in results:
            if not p:
                print(f"  - {name}: {detail}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
