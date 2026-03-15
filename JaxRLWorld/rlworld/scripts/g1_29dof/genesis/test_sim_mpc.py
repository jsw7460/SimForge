"""
Comprehensive tests for SimMPC components.

Tests:
  1. Environment creation (training + planning)
  2. State fork: physics state broadcast from train[idx] → plan[all]
  3. Manager sync: command, action, contact, termination, gait
  4. Reward consistency: planning_step vs World.step for identical actions
  5. Planning step: physics + reward + termination flow
  6. Networks: policy and Q-ensemble forward pass
  7. MPPI planner (with policy warm-start): produces valid actions
  8. Replay buffer: store and sample transitions
  9. Training: Q-learning + policy gradient update
  10. SimMPC.act: end-to-end action selection
  11. Alive masking, warm-start, benchmark

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
    """Detach Genesis scene-bound tensor for cross-scene operations."""
    return t.sceneless() if hasattr(t, "sceneless") else t


def close_enough(a: torch.Tensor, b: torch.Tensor, atol=1e-4, rtol=1e-4) -> bool:
    return torch.allclose(_detach(a).float(), _detach(b).float(), atol=atol, rtol=rtol)


# ──────────────────────────────────────────────────────────────────
# Create environments
# ──────────────────────────────────────────────────────────────────

N_TRAIN = 2       # Training envs
N_PLAN = 16       # Planning envs (small for fast test)
N_PI_TRAJS = 4    # Policy trajectories
HORIZON = 3
NUM_ITERS = 2
NUM_ELITES = 4


def create_envs():
    """Create training and planning environments."""
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

    return train_env, plan_env, cfgs


def get_obs_and_action_dim(train_env):
    obs_dims = train_env.obs_manager.calculate_obs_dim()
    return obs_dims["actor"], train_env.num_actions


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────

def test_env_creation(train_env, plan_env):
    """Test that both environments are created correctly."""
    report("train_env.num_envs", train_env.num_envs == N_TRAIN,
           f"expected {N_TRAIN}, got {train_env.num_envs}")
    report("plan_env.num_envs", plan_env.num_envs == N_PLAN,
           f"expected {N_PLAN}, got {plan_env.num_envs}")
    report("same action_dim", train_env.num_actions == plan_env.num_actions,
           f"train={train_env.num_actions}, plan={plan_env.num_actions}")
    report("same decimation", train_env.decimation == plan_env.decimation,
           f"train={train_env.decimation}, plan={plan_env.decimation}")
    report("action bounds match",
           close_enough(train_env.action_low, plan_env.action_low)
           and close_enough(train_env.action_high, plan_env.action_high), "")
    report("has gait_manager",
           hasattr(train_env, "gait_manager") and hasattr(plan_env, "gait_manager"), "")


def test_state_fork(train_env, plan_env):
    sync = GenesisStateSync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()

    random_actions = torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.1
    train_env.step(random_actions)
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
            expected = src[0]
            for s in range(N_PLAN):
                if not close_enough(dst[s], expected, atol=1e-5):
                    all_match = False
                    break
    report("state_fork broadcast", all_match, "train[0] → all plan envs")


def test_manager_sync(train_env, plan_env):
    sync = GenesisStateSync(train_env, plan_env)

    train_env.command_manager._commands_tensor[0] = torch.tensor(
        [0.5, -0.3, 0.2], device=train_env.device
    )[:train_env.command_manager._commands_tensor.shape[1]]
    sync.sync_managers(train_env_idx=0)

    cmd_src = train_env.command_manager._commands_tensor[0]
    cmd_match = all(
        close_enough(plan_env.command_manager._commands_tensor[s], cmd_src)
        for s in range(N_PLAN)
    )
    report("command_manager sync", cmd_match, "commands broadcast")

    act_src = train_env.act_manager._prev_processed_actions[0]
    act_match = all(
        close_enough(plan_env.act_manager._prev_processed_actions[s], act_src)
        for s in range(N_PLAN)
    )
    report("action_manager sync", act_match, "prev_processed_actions broadcast")

    if hasattr(train_env, "gait_manager"):
        gait_src = train_env.gait_manager.gait_timer[0].item()
        gait_match = all(
            abs(plan_env.gait_manager.gait_timer[s].item() - gait_src) < 1e-6
            for s in range(N_PLAN)
        )
        report("gait_manager sync", gait_match, "gait_timer broadcast")


def test_planning_step(train_env, plan_env):
    sync = GenesisStateSync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()

    actions = torch.zeros(N_TRAIN, train_env.num_actions, device=train_env.device)
    train_env.step(actions)
    sync.fork_and_sync(train_env_idx=0)

    zero_actions = torch.zeros(N_PLAN, plan_env.num_actions, device=plan_env.device)
    rewards, dones = planning_step(plan_env, zero_actions)

    report("planning_step rewards shape", rewards.shape == (N_PLAN,), f"got {rewards.shape}")
    report("planning_step dones shape", dones.shape == (N_PLAN,), f"got {dones.shape}")
    report("planning_step rewards finite", torch.isfinite(rewards).all().item(),
           f"min={rewards.min():.4f}, max={rewards.max():.4f}")
    report("planning_step reward consistency", rewards.std().item() < 1e-3,
           f"std={rewards.std().item():.6f}")


def test_reward_matches_world_step(train_env, plan_env):
    sync = GenesisStateSync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()

    warmup = torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.05
    train_env.step(warmup)
    sync.fork_and_sync(train_env_idx=0)

    fixed_action = torch.randn(1, train_env.num_actions, device=train_env.device) * 0.2
    plan_actions = fixed_action.expand(N_PLAN, -1).clone()
    plan_reward, _ = planning_step(plan_env, plan_actions)

    train_actions = torch.zeros(N_TRAIN, train_env.num_actions, device=train_env.device)
    train_actions[0] = fixed_action[0]
    _, train_reward, _, _, _ = train_env.step(train_actions)

    diff = abs(train_reward[0].item() - plan_reward[0].item())
    report("planning_step vs World.step reward", diff < 0.1,
           f"world={train_reward[0].item():.4f}, plan={plan_reward[0].item():.4f}, diff={diff:.4f}")


def test_fork_sync_exact_reward(train_env, plan_env):
    """Strict test: fork train[0] → plan, apply identical action, compare rewards.

    Compares total reward AND per-term reward breakdown to identify
    which reward term (if any) causes a discrepancy.
    """
    sync = GenesisStateSync(train_env, plan_env)

    # Reset and warm up to get non-trivial state
    train_env.reset()
    plan_env.reset()
    warmup = torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.1
    train_env.step(warmup)

    # Generate a fixed action
    test_action = torch.randn(1, train_env.num_actions, device=train_env.device) * 0.3

    # Fork to plan env
    sync.fork_and_sync(train_env_idx=0)

    # ── Step plan env ──
    plan_act = test_action.expand(N_PLAN, -1).clone()
    plan_reward, _ = planning_step(plan_env, plan_act)
    plan_reward_per_type = {
        k: v[0].item() for k, v in plan_env.rew_buf_per_type.items()
    }

    # ── Step train env ──
    train_actions = torch.zeros(N_TRAIN, train_env.num_actions, device=train_env.device)
    train_actions[0] = test_action[0]
    _, train_reward, _, _, train_info = train_env.step(train_actions)
    train_reward_per_type = {
        k: v[0].item() for k, v in train_info["rewards_per_type"].items()
    }

    # ── Compare total reward ──
    total_diff = abs(train_reward[0].item() - plan_reward[0].item())
    report(
        "fork_sync total reward",
        total_diff < 0.05,
        f"train={train_reward[0].item():.6f}, plan={plan_reward[0].item():.6f}, diff={total_diff:.6f}"
    )

    # ── Compare per-term reward breakdown ──
    all_terms = set(plan_reward_per_type.keys()) | set(train_reward_per_type.keys())
    mismatched_terms = []
    for term in sorted(all_terms):
        if term == "total_reward":
            continue
        p = plan_reward_per_type.get(term, 0.0)
        t = train_reward_per_type.get(term, 0.0)
        diff = abs(p - t)
        if diff > 1e-6:
            mismatched_terms.append((term, t, p, diff))

    if mismatched_terms:
        print(f"    {INFO} Per-term reward breakdown (mismatched only):")
        for term, t_val, p_val, diff in mismatched_terms:
            print(f"      {term:40s}  train={t_val:+.6f}  plan={p_val:+.6f}  diff={diff:.6f}")
    else:
        print(f"    {INFO} All reward terms match exactly.")

    report(
        "fork_sync per-term match",
        len(mismatched_terms) == 0,
        f"{len(mismatched_terms)} terms differ" if mismatched_terms else "all terms match"
    )

    # ── Multi-step: steps 1 and 2 ──
    train_env.reset()
    plan_env.reset()
    warmup2 = torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.1
    train_env.step(warmup2)
    sync.fork_and_sync(train_env_idx=0)

    action_sequence = [
        torch.randn(1, train_env.num_actions, device=train_env.device) * 0.3
        for _ in range(3)
    ]

    plan_rewards = []
    for act in action_sequence:
        plan_act = act.expand(N_PLAN, -1).clone()
        rew, _ = planning_step(plan_env, plan_act)
        plan_rewards.append(rew[0].item())

    train_rewards = []
    for act in action_sequence:
        ta = torch.zeros(N_TRAIN, train_env.num_actions, device=train_env.device)
        ta[0] = act[0]
        _, rew, _, _, _ = train_env.step(ta)
        train_rewards.append(rew[0].item())

    for t in range(3):
        diff = abs(train_rewards[t] - plan_rewards[t])
        report(
            f"fork_sync reward step {t}",
            diff < 0.05,
            f"train={train_rewards[t]:.6f}, plan={plan_rewards[t]:.6f}, diff={diff:.6f}"
        )

    # All plan envs should agree (same forked state + same action)
    sync.fork_and_sync(train_env_idx=0)
    same_act = action_sequence[0].expand(N_PLAN, -1).clone()
    rew_all, _ = planning_step(plan_env, same_act)
    report(
        "fork_sync all plan envs identical",
        rew_all.std().item() < 1e-5,
        f"std={rew_all.std().item():.8f}"
    )


def test_networks(train_env):
    """Test policy and Q-ensemble forward pass."""
    obs_dim, action_dim = get_obs_and_action_dim(train_env)
    device = train_env.device
    batch = 32

    # Policy
    policy = SimMPCPolicy(obs_dim, action_dim, hidden_dims=(64, 32)).to(device)
    obs = torch.randn(batch, obs_dim, device=device)

    action, log_prob = policy(obs)
    report("policy output shape", action.shape == (batch, action_dim), f"got {action.shape}")
    report("policy log_prob shape", log_prob.shape == (batch,), f"got {log_prob.shape}")
    report("policy output finite", torch.isfinite(action).all().item(), "")

    action_det, _ = policy(obs, deterministic=True)
    report("policy deterministic output finite", torch.isfinite(action_det).all().item(), "")

    # Q-ensemble
    q_ens = QEnsemble(obs_dim, action_dim, num_q=3, hidden_dims=(64, 32)).to(device)
    q_vals = q_ens(obs, action.detach())
    report("Q-ensemble output shape", q_vals.shape == (3, batch, 1), f"got {q_vals.shape}")

    q_avg = q_ens.q_value(obs, action.detach())
    report("Q-ensemble avg shape", q_avg.shape == (batch, 1), f"got {q_avg.shape}")
    report("Q-ensemble avg finite", torch.isfinite(q_avg).all().item(), "")


def test_replay_buffer(train_env):
    """Test replay buffer store and sample."""
    obs_dim, action_dim = get_obs_and_action_dim(train_env)
    device = train_env.device

    buf = SimpleReplayBuffer(obs_dim, action_dim, max_size=100, device=device)
    report("replay buffer initial size", buf.size == 0, f"got {buf.size}")

    # Add batch
    obs = torch.randn(10, obs_dim, device=device)
    action = torch.randn(10, action_dim, device=device)
    reward = torch.randn(10, device=device)
    next_obs = torch.randn(10, obs_dim, device=device)
    done = torch.zeros(10, device=device)

    buf.add(obs, action, reward, next_obs, done)
    report("replay buffer after add", buf.size == 10, f"got {buf.size}")

    # Sample
    s_obs, s_act, s_rew, s_next, s_done = buf.sample(5)
    report("replay buffer sample obs shape", s_obs.shape == (5, obs_dim), f"got {s_obs.shape}")
    report("replay buffer sample rew shape", s_rew.shape == (5,), f"got {s_rew.shape}")

    # Overflow
    for _ in range(20):
        buf.add(obs, action, reward, next_obs, done)
    report("replay buffer overflow", buf.size == 100, f"got {buf.size}")


def test_training_update(train_env):
    """Test Q-learning + policy gradient update."""
    obs_dim, action_dim = get_obs_and_action_dim(train_env)
    device = train_env.device
    sync = GenesisStateSync(train_env, train_env)  # dummy, not used in update

    mpc = SimMPC(
        planning_env=train_env,  # dummy for update test
        state_sync=sync,
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=2,
        num_samples=N_TRAIN,
        num_pi_trajs=1,
        num_elites=1,
        num_iterations=1,
        temperature=0.5,
        min_std=0.05,
        max_std=2.0,
        gamma=0.99,
        num_train_envs=N_TRAIN,
        hidden_dims=(64, 32),
        num_q=3,
    )
    mpc.init_storage(obs_dim, action_dim, buffer_size=1000)

    # Fill buffer with random data
    for _ in range(10):
        obs = torch.randn(N_TRAIN, obs_dim, device=device)
        action = torch.randn(N_TRAIN, action_dim, device=device)
        reward = torch.randn(N_TRAIN, device=device)
        next_obs = torch.randn(N_TRAIN, obs_dim, device=device)
        done = torch.zeros(N_TRAIN, device=device)
        mpc.store_transition(obs, action, reward, next_obs, done)

    report("buffer filled", mpc.replay_buffer.size == 20, f"got {mpc.replay_buffer.size}")

    # Update
    metrics = mpc.update(batch_size=16)
    report("update returns metrics", len(metrics) > 0, f"keys={list(metrics.keys())}")
    report("q_loss finite", "q_loss" in metrics and abs(metrics["q_loss"]) < 1e6,
           f"q_loss={metrics.get('q_loss', 'N/A')}")
    report("pi_loss finite", "pi_loss" in metrics and abs(metrics["pi_loss"]) < 1e6,
           f"pi_loss={metrics.get('pi_loss', 'N/A')}")

    # Run several updates to check stability
    for _ in range(10):
        m = mpc.update(batch_size=16)
    report("10 updates stable", "q_loss" in m and abs(m["q_loss"]) < 1e6, "")


def test_alive_masking(train_env):
    S, H = 10, 5
    device = train_env.device

    cumulative_reward = torch.zeros(S, device=device)
    discount = torch.ones(S, device=device)
    alive = torch.ones(S, dtype=torch.bool, device=device)

    for t in range(H):
        rewards = torch.ones(S, device=device)
        dones = torch.zeros(S, dtype=torch.bool, device=device)
        if t == 1:
            dones[2] = True
            dones[5] = True
        cumulative_reward += discount * rewards * alive.float()
        alive = alive & ~dones
        discount *= 0.99

    expected_alive = sum(0.99 ** t for t in range(H))
    expected_dead = 1.0 + 0.99

    report("alive masking: live trajectory",
           abs(cumulative_reward[0].item() - expected_alive) < 1e-4,
           f"expected={expected_alive:.4f}, got={cumulative_reward[0].item():.4f}")
    report("alive masking: dead trajectory",
           abs(cumulative_reward[2].item() - expected_dead) < 1e-4,
           f"expected={expected_dead:.4f}, got={cumulative_reward[2].item():.4f}")


def test_mppi_planner_with_policy(train_env, plan_env):
    """Test MPPI planner with policy warm-start and terminal Q-value."""
    obs_dim, action_dim = get_obs_and_action_dim(train_env)
    sync = GenesisStateSync(train_env, plan_env)

    train_env.reset()
    plan_env.reset()

    warmup = torch.randn(N_TRAIN, train_env.num_actions, device=train_env.device) * 0.1
    train_env.step(warmup)

    policy = SimMPCPolicy(obs_dim, action_dim, hidden_dims=(64, 32)).to(train_env.device)
    q_ens = QEnsemble(obs_dim, action_dim, num_q=3, hidden_dims=(64, 32)).to(train_env.device)

    planner = SimulatorMPPI(
        planning_env=plan_env,
        state_sync=sync,
        policy=policy,
        q_ensemble=q_ens,
        horizon=HORIZON,
        num_samples=N_PLAN,
        num_pi_trajs=N_PI_TRAJS,
        num_elites=NUM_ELITES,
        num_iterations=NUM_ITERS,
        temperature=0.5,
        min_std=0.05,
        max_std=2.0,
        gamma=0.99,
        action_low=plan_env.action_low.clone(),
        action_high=plan_env.action_high.clone(),
        num_train_envs=N_TRAIN,
    )

    action, new_mean = planner.plan(training_env=train_env, train_env_idx=0, t0=True)

    report("MPPI action shape", action.shape == (action_dim,), f"got {action.shape}")
    report("MPPI new_mean shape", new_mean.shape == (HORIZON, action_dim),
           f"got {new_mean.shape}")
    report("MPPI action finite", torch.isfinite(action).all().item(), "")

    action_low = plan_env.action_low
    action_high = plan_env.action_high
    in_bounds = (_detach(action) >= _detach(action_low) - 1e-5).all() and \
                (_detach(action) <= _detach(action_high) + 1e-5).all()
    report("MPPI action within bounds", in_bounds,
           f"min={action.min().item():.4f}, max={action.max().item():.4f}")

    # Warm-start test
    planner._prev_mean[0] = new_mean
    action2, mean2 = planner.plan(training_env=train_env, train_env_idx=0, t0=False)
    means_differ = not close_enough(new_mean, mean2, atol=1e-6)
    report("warm-start: means differ across steps", means_differ, "")


def test_sim_mpc_act(train_env, plan_env):
    """Test end-to-end SimMPC.act()."""
    obs_dim, action_dim = get_obs_and_action_dim(train_env)
    sync = GenesisStateSync(train_env, plan_env)

    train_env.reset()
    plan_env.reset()

    mpc = SimMPC(
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

    t0_mask = torch.ones(N_TRAIN, dtype=torch.bool, device=train_env.device)
    actions = mpc.act(train_env, t0_mask)

    report("SimMPC.act output shape", actions.shape == (N_TRAIN, action_dim),
           f"got {actions.shape}")
    report("SimMPC.act output finite", torch.isfinite(actions).all().item(), "")

    obs, rew, term, trunc, info = train_env.step(actions)
    report("training env step with MPC actions", torch.isfinite(rew).all().item(),
           f"reward={rew.tolist()}")


def test_benchmark(train_env, plan_env):
    sync = GenesisStateSync(train_env, plan_env)
    train_env.reset()
    plan_env.reset()

    warmup = torch.zeros(N_TRAIN, train_env.num_actions, device=train_env.device)
    train_env.step(warmup)
    sync.fork_and_sync(train_env_idx=0)

    # Warmup GPU
    for _ in range(3):
        actions = torch.zeros(N_PLAN, plan_env.num_actions, device=plan_env.device)
        planning_step(plan_env, actions)
    sync.fork_and_sync(train_env_idx=0)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    n_steps = 10
    for _ in range(n_steps):
        actions = torch.randn(N_PLAN, plan_env.num_actions, device=plan_env.device) * 0.1
        planning_step(plan_env, actions)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    ms_per_step = (elapsed / n_steps) * 1000
    report("benchmark: planning_step speed", True,
           f"{ms_per_step:.1f} ms/step ({N_PLAN} envs)")


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
    train_env, plan_env, cfgs = create_envs()
    print(f"{INFO} Environments created.\n")

    print("── Environment Creation ──")
    test_env_creation(train_env, plan_env)
    print()

    print("── State Fork ──")
    test_state_fork(train_env, plan_env)
    print()

    print("── Manager Sync ──")
    test_manager_sync(train_env, plan_env)
    print()

    print("── Planning Step ──")
    test_planning_step(train_env, plan_env)
    test_reward_matches_world_step(train_env, plan_env)
    print()

    print("── Fork+Sync Exact Reward ──")
    test_fork_sync_exact_reward(train_env, plan_env)
    print()

    print("── Networks ──")
    test_networks(train_env)
    print()

    print("── Replay Buffer ──")
    test_replay_buffer(train_env)
    print()

    print("── Training Update ──")
    test_training_update(train_env)
    print()

    print("── Alive Masking ──")
    test_alive_masking(train_env)
    print()

    print("── MPPI Planner (with policy) ──")
    test_mppi_planner_with_policy(train_env, plan_env)
    print()

    print("── SimMPC End-to-End ──")
    test_sim_mpc_act(train_env, plan_env)
    print()

    print("── Benchmark ──")
    test_benchmark(train_env, plan_env)
    print()

    # ── Summary ──
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
