from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# ============================================================================
# Abstract Base Class
# ============================================================================


class DynamicsModel(nn.Module, ABC):
    """
    Abstract base class for dynamics models.

    All dynamics models should predict next state given current state and action.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state.

        Args:
            x: (batch, state_dim) - Current state
            action: (batch, action_dim) - Action

        Returns:
            next_x: (batch, state_dim) - Predicted next state
        """
        pass

    def predict_delta(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict state change (delta).

        Args:
            x: (batch, state_dim)
            action: (batch, action_dim)

        Returns:
            delta: (batch, state_dim)
        """
        next_x = self.forward(x, action)
        return next_x - x

    def rollout(self, initial_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Perform multi-step rollout.

        Args:
            initial_state: (batch, state_dim)
            actions: (batch, horizon, action_dim)

        Returns:
            states: (batch, horizon+1, state_dim) - Including initial state
        """
        batch_size, horizon, _ = actions.shape
        states = [initial_state]

        x = initial_state
        for t in range(horizon):
            x = self.forward(x, actions[:, t])
            states.append(x)

        return torch.stack(states, dim=1)  # (batch, horizon+1, state_dim)
