"""
SimulatorMPPI: MPPI planning with real simulator rollouts.

Mirrors TDMPC2's MPPI structure:
  - Policy generates num_pi_trajs candidate trajectories
  - Noise sampling generates (num_samples - num_pi_trajs) candidates
  - All candidates rolled out in actual simulator
  - Trajectory value = sum of real rewards + terminal Q-value
  - CEM/MPPI update selects best action
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from rlworld.rl.envs.genesis.genesis_env import GenesisEnv
    from .state_sync import GenesisStateSync
    from .networks import SimMPCPolicy, QEnsemble


def planning_step(
    plan_env: GenesisEnv,
    actions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lightweight World.step() for planning: physics + reward + termination.

    No observations, events, or command updates.

    Args:
        plan_env: The planning environment (S parallel envs).
        actions: [S, action_dim] — candidate actions.

    Returns:
        rewards: [S] — reward for each candidate.
        dones: [S] — whether each trajectory terminated.
    """
    # Process and apply actions
    processed = plan_env.act_manager.process_actions(actions)
    plan_env.act_manager.apply_actions(processed)

    # Physics (decimated)
    for _ in range(plan_env.decimation):
        plan_env.scene_manager.step()

    plan_env._invalidate_cache()

    # Contact update (needed for contact-based rewards)
    plan_env.contact_manager.advance()

    # Reward computation
    # Pre-reward hook (e.g., gait_manager.advance() in LocomotionEnv)
    plan_env._pre_reward_hook()

    plan_env.rew_buf[:] = 0.0
    plan_env.reward_manager.set_rewards(
        reward_buffer=plan_env.rew_buf,
        episode_sums=plan_env.episode_sums,
        reward_buffer_per_type=plan_env.rew_buf_per_type,
    )

    # Pre-termination hook
    plan_env._pre_termination_hook()

    # Termination check
    terminated, truncated = plan_env.termination_manager.check_termination()
    dones = terminated | truncated

    # Reset done environments (so subsequent horizon steps have valid physics)
    done_ids = dones.nonzero(as_tuple=False).flatten()
    if len(done_ids) > 0:
        plan_env._reset_idx(done_ids)

    # Advance action history (for action_rate reward on next step)
    plan_env.act_manager.advance()
    # Advance termination manager (episode_length_buf += 1)
    plan_env.termination_manager.advance()

    return plan_env.rew_buf.clone(), dones


def _get_obs_for_planning(plan_env: GenesisEnv) -> torch.Tensor:
    """Get observation from planning env (for policy/Q-network).

    Lightweight: only computes actor obs, no history update.
    """
    plan_env.obs_manager.process_observations(update_history=False)
    obs_dict = plan_env.obs_manager.obs_dict
    return obs_dict["actor"]


class SimulatorMPPI:
    """MPPI planner using real simulator rollouts + learned policy warm-start."""

    def __init__(
        self,
        planning_env: GenesisEnv,
        state_sync: GenesisStateSync,
        policy: SimMPCPolicy,
        q_ensemble: QEnsemble,
        horizon: int,
        num_samples: int,
        num_pi_trajs: int,
        num_elites: int,
        num_iterations: int,
        temperature: float,
        min_std: float,
        max_std: float,
        gamma: float,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        num_train_envs: int,
    ):
        self.planning_env = planning_env
        self.state_sync = state_sync
        self.policy = policy
        self.q_ensemble = q_ensemble
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_pi_trajs = num_pi_trajs
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.min_std = min_std
        self.max_std = max_std
        self.gamma = gamma
        self.action_low = action_low   # [action_dim]
        self.action_high = action_high  # [action_dim]
        self.device = planning_env.device
        self.use_terminal_q = False  # Enable after Q-network is trained

        action_dim = action_low.shape[0]

        # Warm-start mean: [N_train, horizon, action_dim]
        self._prev_mean = torch.zeros(
            num_train_envs, horizon, action_dim, device=self.device
        )

    @torch.no_grad()
    def _sample_policy_trajectories(self) -> torch.Tensor:
        """Autoregressive policy rollout using the actual simulator.

        Like TDMPC2's pi(z) → a, z = dynamics(z, a) → pi(z) → ...
        but using real simulator steps instead of learned dynamics.

        Uses the first num_pi_trajs envs of the planning env (all start
        from the same forked state). The remaining envs are stepped too
        (unavoidable with batched simulator), but their results are ignored.

        The planning env must already be forked. State will be modified
        and must be re-forked before CEM rollout.

        Returns:
            pi_actions: [horizon, num_pi_trajs, action_dim]
        """
        H = self.horizon
        S = self.num_samples
        num_pi = self.num_pi_trajs
        action_dim = self.action_low.shape[0]

        pi_actions = []
        for t in range(H):
            # Get current obs from simulator state
            obs = _get_obs_for_planning(self.planning_env)  # [S, obs_dim]
            obs_pi = obs[:num_pi]  # [num_pi, obs_dim]

            # Policy forward: stochastic → each of num_pi gets different action
            action, _ = self.policy(obs_pi, deterministic=False)
            action = action.clamp(self.action_low, self.action_high)
            pi_actions.append(action)

            # Step simulator to advance state for next horizon step
            if t < H - 1:
                full_action = torch.zeros(S, action_dim, device=self.device)
                full_action[:num_pi] = action
                planning_step(self.planning_env, full_action)
        return torch.stack(pi_actions, dim=0)  # [H, num_pi, action_dim]

    @torch.no_grad()
    def plan(
        self,
        training_env: GenesisEnv,
        train_env_idx: int,
        t0: bool,
        eval_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run MPPI planning for a single training environment.

        Mirrors TDMPC2's plan_mppi_inner:
          1. Sample policy trajectories (warm-start)
          2. CEM iterations: noise sample + simulator rollout + elite selection
          3. Terminal Q-value added to trajectory returns
          4. Gumbel-softmax action selection

        Args:
            training_env: The training environment.
            train_env_idx: Index of the training env to plan for.
            t0: Whether this is the first step of an episode.
            eval_mode: If True, no exploration noise on final action.

        Returns:
            action: [action_dim] — selected action.
            new_mean: [horizon, action_dim] — updated mean for warm-start.
        """
        H = self.horizon
        S = self.num_samples
        num_pi = self.num_pi_trajs
        num_noise = S - num_pi
        action_dim = self.action_low.shape[0]

        # Cache training env state once (avoid repeated get_state in CEM loop)
        self.state_sync.begin_planning(train_env_idx)

        # ── Sample policy trajectories ──
        self.state_sync.fork_and_sync(train_env_idx)
        pi_actions = self._sample_policy_trajectories()  # [H, num_pi, action_dim]

        # ── Warm-start mean ──
        if t0:
            mean = torch.zeros(H, action_dim, device=self.device)
        else:
            prev = self._prev_mean[train_env_idx]  # [H, action_dim]
            mean = torch.zeros(H, action_dim, device=self.device)
            mean[:-1] = prev[1:]  # shift left, last step = 0

        std = self.max_std * torch.ones(H, action_dim, device=self.device)

        # ── CEM iterations ──
        for iteration in range(self.num_iterations):
            # Re-fork state (planning env was modified by rollout)
            self.state_sync.fork_and_sync(train_env_idx)

            # Sample noise actions: [H, num_noise, action_dim]
            noise = torch.randn(H, num_noise, action_dim, device=self.device)
            noise_actions = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(
                self.action_low, self.action_high
            )

            # Combine: [H, S, action_dim]
            actions = torch.cat([pi_actions, noise_actions], dim=1)

            # ── Rollout with alive masking ──
            cumulative_reward = torch.zeros(S, device=self.device)
            discount = torch.ones(S, device=self.device)
            alive = torch.ones(S, dtype=torch.bool, device=self.device)

            for t in range(H):
                rewards, dones = planning_step(self.planning_env, actions[t])
                cumulative_reward += discount * rewards * alive.float()
                alive = alive & ~dones
                discount *= self.gamma

            # ── Terminal Q-value (only when Q-network is trained) ──
            if self.use_terminal_q:
                obs_terminal = _get_obs_for_planning(self.planning_env)  # [S, obs_dim]
                terminal_action, _ = self.policy(obs_terminal, deterministic=True)
                terminal_q = self.q_ensemble.q_value(obs_terminal, terminal_action).squeeze(-1)
                cumulative_reward += discount * terminal_q * alive.float()

            # ── Elite selection + MPPI update ──
            _, elite_idx = cumulative_reward.topk(self.num_elites)
            elite_actions = actions[:, elite_idx]  # [H, num_elites, action_dim]
            elite_values = cumulative_reward[elite_idx]  # [num_elites]

            # Softmax scoring
            score = torch.exp(
                self.temperature * (elite_values - elite_values.max())
            )
            score = score / (score.sum() + 1e-9)  # [num_elites]

            # Weighted mean and std
            mean = (score[None, :, None] * elite_actions).sum(dim=1)  # [H, action_dim]
            residual = elite_actions - mean.unsqueeze(1)
            std = (
                (score[None, :, None] * residual.square()).sum(dim=1)
                / (score.sum() + 1e-9)
            ).sqrt().clamp(self.min_std, self.max_std)

        # ── Final action selection via Gumbel-Softmax ──
        gumbel = -torch.log(-torch.log(torch.rand_like(score) + 1e-8) + 1e-8)
        selected = (score.log() + gumbel).argmax()
        action = elite_actions[0, selected]  # [action_dim]

        # Add exploration noise if not eval
        if not eval_mode:
            exploration_noise = torch.randn(action_dim, device=self.device) * std[0]
            action = (action + exploration_noise).clamp(self.action_low, self.action_high)

        self.state_sync.end_planning()
        return action, mean
