"""
SimMPC: Model Predictive Control using real simulator rollouts.

Mirrors TDMPC2 architecture but replaces learned dynamics with actual simulator:
  - Policy (MLP): obs → action, used for MPPI warm-start + deployed to real world
  - Q-ensemble: (obs, action) → value, used for terminal value + policy training
  - MPPI planner: policy trajs + noise trajs → simulator rollout → CEM selection
  - Training: TD3-style Q-learning + deterministic policy gradient
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F

from .networks import SimMPCPolicy, QEnsemble
from .planner import SimulatorMPPI
from .state_sync import GenesisStateSync

if TYPE_CHECKING:
    from rlworld.rl.envs.genesis.genesis_env import GenesisEnv


class SimMPC:
    """SimMPC: MPPI with real simulator + learned policy & Q-network."""

    def __init__(
        self,
        planning_env: GenesisEnv,
        state_sync: GenesisStateSync,
        obs_dim: int,
        action_dim: int,
        # Planning
        horizon: int,
        num_samples: int,
        num_pi_trajs: int,
        num_elites: int,
        num_iterations: int,
        temperature: float,
        min_std: float,
        max_std: float,
        gamma: float,
        num_train_envs: int,
        # Training
        lr: float = 3e-4,
        pi_lr: float = 3e-4,
        tau: float = 0.005,
        # Networks
        hidden_dims: tuple = (512, 256),
        num_q: int = 5,
        squash_policy: bool = False,
    ):
        self.planning_env = planning_env
        self.state_sync = state_sync
        self.device = planning_env.device
        self.tau = tau
        self.gamma = gamma

        # Compatibility with BaseRunner (sets alg._last_dones)
        self._last_dones = None

        # Action bounds
        action_low = planning_env.action_low.clone()
        action_high = planning_env.action_high.clone()

        # ── Networks ──
        self.policy = SimMPCPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            squash=squash_policy,
        ).to(self.device)

        self.q_ensemble = QEnsemble(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_q=num_q,
            hidden_dims=hidden_dims,
        ).to(self.device)

        self.target_q_ensemble = copy.deepcopy(self.q_ensemble)
        for p in self.target_q_ensemble.parameters():
            p.requires_grad_(False)

        # ── Optimizers ──
        self.pi_optimizer = torch.optim.Adam(self.policy.parameters(), lr=pi_lr)
        self.q_optimizer = torch.optim.Adam(self.q_ensemble.parameters(), lr=lr)

        # ── Planner ──
        self.planner = SimulatorMPPI(
            planning_env=planning_env,
            state_sync=state_sync,
            policy=self.policy,
            q_ensemble=self.q_ensemble,
            horizon=horizon,
            num_samples=num_samples,
            num_pi_trajs=num_pi_trajs,
            num_elites=num_elites,
            num_iterations=num_iterations,
            temperature=temperature,
            min_std=min_std,
            max_std=max_std,
            gamma=gamma,
            action_low=action_low,
            action_high=action_high,
            num_train_envs=num_train_envs,
        )

        # ── Replay buffer ──
        self.replay_buffer: Optional[SimpleReplayBuffer] = None

    def init_storage(self, obs_dim: int, action_dim: int, buffer_size: int):
        """Initialize replay buffer."""
        self.replay_buffer = SimpleReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_size=buffer_size,
            device=self.device,
        )

    @torch.no_grad()
    def act(
        self,
        training_env: GenesisEnv,
        t0_mask: torch.Tensor,
        eval_mode: bool = False,
    ) -> torch.Tensor:
        """Select actions for all training envs via sequential MPPI.

        Args:
            training_env: The training environment.
            t0_mask: [N] bool tensor — True if episode just started.
            eval_mode: If True, no exploration noise.

        Returns:
            actions: [N, action_dim] — selected actions.
        """
        N = training_env.num_envs
        actions = []

        for i in range(N):
            action, new_mean = self.planner.plan(
                training_env=training_env,
                train_env_idx=i,
                t0=t0_mask[i].item() if isinstance(t0_mask, torch.Tensor) else bool(t0_mask[i]),
                eval_mode=eval_mode,
            )
            self.planner._prev_mean[i] = new_mean
            actions.append(action)

        return torch.stack(actions)

    def store_transition(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """Store transition in replay buffer."""
        if self.replay_buffer is not None:
            self.replay_buffer.add(obs, action, reward, next_obs, done)

    def update(self, batch_size: int) -> Dict[str, float]:
        """TD3-style update: Q-learning + deterministic policy gradient.

        Returns:
            metrics dict with losses.
        """
        if self.replay_buffer is None or self.replay_buffer.size < batch_size:
            return {}

        obs, action, reward, next_obs, done = self.replay_buffer.sample(batch_size)

        # ── Q update ──
        with torch.no_grad():
            next_action, _ = self.policy(next_obs, deterministic=True)
            target_q = self.target_q_ensemble.q_value(next_obs, next_action).squeeze(-1)
            target = reward + self.gamma * (1.0 - done) * target_q

        # Each Q in ensemble predicts target
        q_pred = self.q_ensemble(obs, action)  # [num_q, batch, 1]
        q_loss = sum(
            F.mse_loss(q_pred[i].squeeze(-1), target)
            for i in range(len(self.q_ensemble.nets))
        )

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ── Policy update (deterministic policy gradient) ──
        pi_action, _ = self.policy(obs)
        q_val = self.q_ensemble.q_value(obs, pi_action)
        pi_loss = -q_val.mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        # ── Polyak target update ──
        with torch.no_grad():
            for p, tp in zip(self.q_ensemble.parameters(), self.target_q_ensemble.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "q_loss": q_loss.item(),
            "pi_loss": pi_loss.item(),
            "q_mean": q_pred.mean().item(),
            "target_mean": target.mean().item(),
        }

    def save_train_state(self, checkpoint_dir: str) -> dict:
        """Save policy, Q-ensemble, and optimizers."""
        torch.save({
            "policy": self.policy.state_dict(),
            "q_ensemble": self.q_ensemble.state_dict(),
            "target_q_ensemble": self.target_q_ensemble.state_dict(),
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
        }, os.path.join(checkpoint_dir, "sim_mpc.pt"))
        return {}

    def load_train_state(self, checkpoint_dir: str, metadata: dict) -> None:
        """Load policy, Q-ensemble, and optimizers."""
        state = torch.load(
            os.path.join(checkpoint_dir, "sim_mpc.pt"),
            map_location=self.device,
        )
        self.policy.load_state_dict(state["policy"])
        self.q_ensemble.load_state_dict(state["q_ensemble"])
        self.target_q_ensemble.load_state_dict(state["target_q_ensemble"])
        self.pi_optimizer.load_state_dict(state["pi_optimizer"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])


class SimpleReplayBuffer:
    """Simple flat replay buffer for (obs, action, reward, next_obs, done)."""

    def __init__(self, obs_dim: int, action_dim: int, max_size: int, device: torch.device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = torch.zeros(max_size, obs_dim, device=device)
        self.action = torch.zeros(max_size, action_dim, device=device)
        self.reward = torch.zeros(max_size, device=device)
        self.next_obs = torch.zeros(max_size, obs_dim, device=device)
        self.done = torch.zeros(max_size, device=device)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """Add batch of transitions. obs: [batch, obs_dim], etc."""
        batch = obs.shape[0]
        idx = torch.arange(self.ptr, self.ptr + batch, device=self.device) % self.max_size

        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx] = reward.flatten()
        self.next_obs[idx] = next_obs
        self.done[idx] = done.float().flatten()

        self.ptr = (self.ptr + batch) % self.max_size
        self.size = min(self.size + batch, self.max_size)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.action[idx],
            self.reward[idx],
            self.next_obs[idx],
            self.done[idx],
        )
