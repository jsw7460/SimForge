"""
PyTorch policy and Q-network for SimMPC.

Policy: obs → action (Gaussian, optionally squashed)
Q-network: (obs, action) → scalar value (ensemble)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimMPCPolicy(nn.Module):
    """Gaussian policy: obs → action.

    When squash=True, output is tanh-squashed to [-1, 1].
    When squash=False, output is raw Gaussian (clipped by MPPI).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256),
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        squash: bool = False,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.squash = squash
        self.action_dim = action_dim

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [batch, obs_dim]
            deterministic: if True, return mean (no sampling)

        Returns:
            action: [batch, action_dim]
            log_prob: [batch] (for entropy regularization)
        """
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean) if self.squash else mean
            return action, torch.zeros(obs.shape[0], device=obs.device)

        # Reparameterized sample
        noise = torch.randn_like(mean)
        raw_action = mean + std * noise

        if self.squash:
            action = torch.tanh(raw_action)
            # Log-prob with tanh correction
            log_prob = (
                -0.5 * noise.square().sum(-1)
                - log_std.sum(-1)
                - (1 - action.square() + 1e-6).log().sum(-1)
            )
        else:
            action = raw_action
            log_prob = -0.5 * noise.square().sum(-1) - log_std.sum(-1)

        return action, log_prob


class QNetwork(nn.Module):
    """Single Q-network: (obs, action) → scalar."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        layers = []
        in_dim = obs_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns: [batch, 1]"""
        return self.net(torch.cat([obs, action], dim=-1))


class QEnsemble(nn.Module):
    """Ensemble of Q-networks."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_q: int = 5,
        hidden_dims: Tuple[int, ...] = (512, 256),
    ):
        super().__init__()
        self.nets = nn.ModuleList([
            QNetwork(obs_dim, action_dim, hidden_dims) for _ in range(num_q)
        ])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns: [num_q, batch, 1]"""
        return torch.stack([q(obs, action) for q in self.nets], dim=0)

    def q_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Average Q-value across ensemble. Returns: [batch, 1]"""
        return self.forward(obs, action).mean(dim=0)
